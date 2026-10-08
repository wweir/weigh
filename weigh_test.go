package weigh_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/wweir/weigh"
)

// fake is a vLLM-shaped backend whose tokenizer is one token per rune, so the answer boundary
// holds by construction and the slot ids are predictable.
type fake struct {
	// root is what /v1/models reports as the served checkpoint. Empty omits the field, which is
	// how a backend with nothing to compare is exercised.
	root string
}

func (f *fake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/version":
		writeJSON(w, map[string]any{"version": "0.27.1"})
	case "/v1/models":
		model := map[string]any{"id": "served-model", "max_model_len": 4096}
		if f.root != "" {
			model["root"] = f.root
		}
		writeJSON(w, map[string]any{"data": []any{model}})
	case "/tokenize":
		var body struct {
			Prompt string `json:"prompt"`
		}
		decode(r, &body)
		ids := make([]int, 0, len(body.Prompt))
		for _, char := range body.Prompt {
			ids = append(ids, 100+int(char))
		}
		writeJSON(w, map[string]any{"tokens": ids, "count": len(ids), "max_model_len": 4096})
	case "/v1/completions":
		var body map[string]any
		decode(r, &body)
		pinned, _ := body["logprob_token_ids"].([]any)
		top := map[string]any{}
		for index, entry := range pinned {
			slot, _ := entry.(float64)
			top[fmt.Sprintf("token_id:%d", int(slot))] = -0.5 * float64(index+1)
		}
		writeJSON(w, map[string]any{"choices": []any{
			map[string]any{"logprobs": map[string]any{"top_logprobs": []any{top}}},
		}})
	default:
		http.NotFound(w, r)
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func decode(r *http.Request, out any) {
	_ = json.NewDecoder(r.Body).Decode(out)
}

func digest(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

func start(t *testing.T, backend *fake) string {
	t.Helper()
	server := httptest.NewServer(backend)
	t.Cleanup(server.Close)
	return server.URL
}

func open(t *testing.T, cfg weigh.Config) (*weigh.Client, *weigh.Metadata) {
	t.Helper()
	if cfg.Timeout == 0 {
		cfg.Timeout = 5 * time.Second
	}
	client, metadata, err := weigh.New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return client, metadata
}

// TestDecideScoresTheCallersOwnPrompt is the library contract in one case: a caller with no
// local checkpoint renders an arbitrary prompt and gets a number attributed to exactly those
// bytes. The backend reports a served root, and that must not be compared against a directory
// this client never had.
func TestDecideScoresTheCallersOwnPrompt(t *testing.T) {
	client, metadata := open(t, weigh.Config{
		URLs:    []string{start(t, &fake{root: "a-checkpoint-this-library-never-read"})},
		Backend: weigh.BackendVLLM,
		Readout: weigh.ReadoutExactSlot,
		// No Source: this process renders its own prompt.
	})

	if metadata.PromptTemplate != "" {
		t.Fatalf("an unread checkpoint must leave the template unresolved, got %q", metadata.PromptTemplate)
	}
	if metadata.PromptTemplateSource != "" {
		t.Fatalf("an unresolved template has no source, got %q", metadata.PromptTemplateSource)
	}
	if metadata.TokenizerMatches != nil {
		t.Fatal("no checkpoint was named, so there is nothing a served tokenizer could have been checked against")
	}

	// Deliberately not the direct-options contract: this library renders no chat template.
	const prompt = "whatever wrapper this caller wants\nA or B\n"
	scored, err := client.Decide(context.Background(), "row-1", prompt, []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}

	if got, want := *scored.PromptSHA256, digest(prompt); got != want {
		t.Fatalf("PromptSHA256 = %s; want the digest of the caller's own bytes (%s)", got, want)
	}
	if len(scored.Probabilities) != 2 || len(scored.OptionIDs) != 2 {
		t.Fatalf("got %d probabilities and %d labels for 2 options", len(scored.Probabilities), len(scored.OptionIDs))
	}
	if scored.OptionIDs[0] != "yes" || scored.OptionIDs[1] != "no" {
		t.Fatalf("labels were not echoed in slot order: %v", scored.OptionIDs)
	}
	// The fake scores slot 0 above slot 1, so the subset softmax must keep that order.
	if scored.Probabilities[0] <= scored.Probabilities[1] {
		t.Fatalf("slot order was not preserved: %v", scored.Probabilities)
	}
	if scored.Modality != "text" || !scored.ServerTokenized {
		t.Fatalf("modality provenance = %q, server_tokenized = %v", scored.Modality, scored.ServerTokenized)
	}
	if scored.Readout == "" {
		t.Fatal("a readout that ran must describe the request it read")
	}
	if metadata.ServingConfig != weigh.ServingVLLMTokenIDs {
		t.Fatalf("serving config = %q; want the exact-slot recipe %q", metadata.ServingConfig, weigh.ServingVLLMTokenIDs)
	}
}

// TestANamedFamilyIsRecordedWithoutACheckpoint: naming a family needs no file, so it must come
// back and stay renderable. Dropping it would leave Metadata.PromptTemplate empty, and the
// assembly path would then refuse a family the caller had already configured.
func TestANamedFamilyIsRecordedWithoutACheckpoint(t *testing.T) {
	client, metadata := open(t, weigh.Config{
		URLs:     []string{start(t, &fake{root: "a-checkpoint-this-library-never-read"})},
		Backend:  weigh.BackendVLLM,
		Readout:  weigh.ReadoutExactSlot,
		Template: weigh.TemplateChoice{Fixed: weigh.Chatml},
	})
	if metadata.PromptTemplate != weigh.Chatml {
		t.Fatalf("template = %q; want the configured %q", metadata.PromptTemplate, weigh.Chatml)
	}

	row, err := weigh.ValidateRow(map[string]any{
		"id":       "row-1",
		"state":    "the sky is blue",
		"question": "Is the sky blue?",
		"options": []any{
			map[string]any{"id": "yes", "description": "yes"},
			map[string]any{"id": "no", "description": "no"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := weigh.RenderDirectOptions(metadata.PromptTemplate, row)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Decide(context.Background(), row.ID, rendered, []string{"yes", "no"}); err != nil {
		t.Fatal(err)
	}
}

// TestAnUnusableTemplateChoiceIsRefused: a template choice that cannot be honoured is reported,
// not dropped. Silence here leaves a caller assembling against the wrong wrapper, or against none.
func TestAnUnusableTemplateChoiceIsRefused(t *testing.T) {
	cases := []struct {
		name string
		tune func(*weigh.Config)
		want string
	}{
		{"unknown family", func(c *weigh.Config) { c.Template = weigh.TemplateChoice{Fixed: "no-such-family"} }, "unknown template"},
		{"auto without a checkpoint", func(c *weigh.Config) { c.Template = weigh.TemplateChoice{Auto: true} }, "names none"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := weigh.Config{
				URLs:    []string{start(t, &fake{})},
				Backend: weigh.BackendVLLM,
				Readout: weigh.ReadoutExactSlot,
				Timeout: 5 * time.Second,
			}
			tc.tune(&cfg)
			if _, _, err := weigh.New(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("New() error = %v; want one mentioning %q", err, tc.want)
			}
		})
	}
}

// TestConfiguredAssemblyRendersAndScores shows the configured path: the caller picks the template
// family, renders the published payload with this package's primitives, and scores the result.
func TestConfiguredAssemblyRendersAndScores(t *testing.T) {
	client, metadata := open(t, weigh.Config{
		URLs:     []string{start(t, &fake{})},
		Backend:  weigh.BackendVLLM,
		Readout:  weigh.ReadoutExactSlot,
		Template: weigh.TemplateChoice{Fixed: weigh.Chatml},
		Source:   t.TempDir(),
	})
	if metadata.PromptTemplate != weigh.Chatml {
		t.Fatalf("template = %q; want the explicitly requested %q", metadata.PromptTemplate, weigh.Chatml)
	}

	row, err := weigh.ValidateRow(map[string]any{
		"id":       "row-1",
		"state":    "the sky is blue",
		"question": "Is the sky blue?",
		"options": []any{
			map[string]any{"id": "yes", "description": "yes"},
			map[string]any{"id": "no", "description": "no"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	rendered, err := weigh.RenderDirectOptions(metadata.PromptTemplate, row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(rendered, "<|im_start|>system\n"+weigh.DirectSystem) {
		t.Fatalf("the ChatML family must open with the system turn: %q", rendered)
	}
	// The letters are the slot binding the readout will use, so they must be in the payload.
	if !strings.Contains(rendered, `"letter": "A"`) || !strings.Contains(rendered, `"letter": "B"`) {
		t.Fatalf("the payload must bind options to slot letters: %q", rendered)
	}

	scored, err := client.Decide(context.Background(), row.ID, rendered, []string{row.Options[0].ID, row.Options[1].ID})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := *scored.PromptSHA256, digest(rendered); got != want {
		t.Fatalf("PromptSHA256 = %s; want the digest of the rendered prompt (%s)", got, want)
	}
	if scored.Probabilities[0] <= scored.Probabilities[1] {
		t.Fatalf("slot order was not preserved: %v", scored.Probabilities)
	}
}

// TestMediaIsRefusedWithoutAProbedRoute pins the library's share of the media contract: the
// refusal is a ContractError the caller can branch on, not a number read under a route that
// never ran.
func TestMediaIsRefusedWithoutAProbedRoute(t *testing.T) {
	client, _ := open(t, weigh.Config{
		URLs:    []string{start(t, &fake{})},
		Backend: weigh.BackendVLLM,
		Readout: weigh.ReadoutExactSlot,
		// Media stays at its zero value: this deployment never probed an image.
	})

	payload, err := weigh.MediaUserPayload("the sky is blue", "Is the sky blue?", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.DecideMedia(context.Background(), "row-1", weigh.DirectSystem, payload,
		[]weigh.Media{{Reference: "data:image/png;base64,iVBORw0KGgo="}}, []string{"yes", "no"})
	var contract *weigh.ContractError
	if !errors.As(err, &contract) {
		t.Fatalf("want a *weigh.ContractError, got %v", err)
	}
}

// TestTemplatePrimitivesAreCallersFacing: a caller that renders its own prompt still needs to know
// which wrapper the checkpoint declares, and the answer must not be a guess.
func TestTemplatePrimitives(t *testing.T) {
	if family, ok := weigh.DetectTemplate("<|im_start|>system\nx<|im_end|>"); !ok || family != weigh.Chatml {
		t.Fatalf("DetectTemplate(ChatML) = %q, %v", family, ok)
	}
	if _, ok := weigh.DetectTemplate("a shape no build here renders"); ok {
		t.Fatal("an unknown chat_template must not be detected as a family")
	}
	if _, ok := weigh.DetectTemplate("{{ raise 'System role not supported' }}<start_of_turn>"); ok {
		t.Fatal("a template that refuses the system role must be refused, not detected")
	}

	// A checkpoint with no chat_template keeps the historical default, and says so.
	resolved, err := weigh.ResolveTemplate(weigh.TemplateChoice{Auto: true}, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Template != weigh.Gemma4 || resolved.Source != "default:no-chat-template" {
		t.Fatalf("ResolveTemplate on an empty dir = %q/%q", resolved.Template, resolved.Source)
	}

	choice, err := weigh.ParseTemplateChoice("auto")
	if err != nil || !choice.Auto {
		t.Fatalf("ParseTemplateChoice(auto) = %+v, %v", choice, err)
	}
	if _, err := weigh.ParseTemplateChoice("no-such-family"); err == nil {
		t.Fatal("an unknown family name must be refused")
	}
	if _, err := weigh.RenderPrompt("", "system", "user"); err == nil {
		t.Fatal("rendering an unset family must be an error, not a panic")
	}
}
