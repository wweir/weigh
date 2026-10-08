package readout

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/media"
	"github.com/wweir/weigh/internal/template"
	"github.com/wweir/weigh/internal/tokenize"
)

func TestSoftmaxNormalisesAndRefusesUnusableInput(t *testing.T) {
	values, err := Softmax([]float64{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 0.5 || values[1] != 0.5 {
		t.Fatalf("softmax = %v", values)
	}
	if _, err := Softmax([]float64{1}); err == nil {
		t.Error("one score is not a distribution")
	}
	if _, err := Softmax([]float64{1, 2, 3}); err != nil {
		t.Fatalf("three slots must be fine: %v", err)
	}
}

// fakeBackend is a vLLM-shaped backend whose tokenizer is one token per rune, so the answer
// boundary holds by construction and the slot ids are predictable.
type fakeBackend struct {
	version string
	root    string
	// omitSlot, when set, is left out of the pinned completion response, which is how a
	// contract violation is tested.
	omitSlot uint32
	// ignorePins makes the build answer the top-N route without token-id keys.
	ignorePins bool
	// topNOnly narrows the top-N response to the first slot, forcing the fallback.
	topNOnly bool
	// mediaTextKeys makes the chat route answer under decoded-text keys instead of token ids.
	mediaTextKeys bool
	// mediaPinsRefused refuses the pinned chat request while answering the unpinned one with
	// token-id keys: the combination that proves the best-effort media route.
	mediaPinsRefused bool
	// mediaTopNOnly narrows the unpinned chat answer to the first slot.
	mediaTopNOnly bool

	mu         sync.Mutex
	generative int
}

func (f *fakeBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/version":
		writeJSON(w, map[string]any{"version": f.version})
	case r.URL.Path == "/v1/models":
		writeJSON(w, map[string]any{"data": []any{
			map[string]any{"id": "served-model", "root": f.root, "max_model_len": 4096},
		}})
	case r.URL.Path == "/tokenize":
		var body struct {
			Prompt string `json:"prompt"`
		}
		decodeBody(r, &body)
		ids := runeIDs(body.Prompt)
		writeJSON(w, map[string]any{"tokens": ids, "count": len(ids), "max_model_len": 4096})
	case r.URL.Path == "/v1/completions":
		var body map[string]any
		decodeBody(r, &body)
		pinned, _ := body["logprob_token_ids"].([]any)
		top := map[string]any{}
		switch {
		case f.ignorePins:
			// A build that accepts `logprob_token_ids` and ignores it answers under decoded-text
			// keys, which is exactly what the probe looks for and what this service cannot match
			// without a local tokenizer.
			top["A"] = -0.5
			top["B"] = -1.0
		case len(pinned) > 0:
			for index, entry := range pinned {
				slot := uint32(toNumber(entry))
				if f.omitSlot != 0 && slot == f.omitSlot {
					continue
				}
				top[fmt.Sprintf("token_id:%d", slot)] = -0.5 * float64(index+1)
			}
		default:
			limit := 2
			if f.topNOnly {
				limit = 1
			}
			for index := 0; index < limit; index++ {
				top[fmt.Sprintf("token_id:%d", 100+'A'+index)] = -0.5 * float64(index+1)
			}
		}
		writeJSON(w, map[string]any{"choices": []any{
			map[string]any{"logprobs": map[string]any{"top_logprobs": []any{top}}},
		}})
	case r.URL.Path == "/v1/chat/completions":
		var body map[string]any
		decodeBody(r, &body)
		pinned, _ := body["logprob_token_ids"].([]any)
		if f.mediaPinsRefused && len(pinned) > 0 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"logprob_token_ids is not supported"}`))
			return
		}
		// The real chat shape, measured on vLLM 0.27.1: the scored position is nested under
		// `logprobs.content[]`, and `top_logprobs` is a *list* of entries carrying a `token`
		// field — not the `token_id:<id>`-keyed object `/v1/completions` returns.
		tops := []any{}
		switch {
		case f.mediaTextKeys:
			tops = append(tops, map[string]any{"token": "A", "logprob": -0.5})
		default:
			slots := pinned
			if len(slots) == 0 {
				limit := 2
				if f.mediaTopNOnly {
					limit = 1
				}
				slots = make([]any, 0, limit)
				for index := 0; index < limit; index++ {
					slots = append(slots, float64(100+'A'+index))
				}
			}
			for index, entry := range slots {
				tops = append(tops, map[string]any{
					"token":   fmt.Sprintf("token_id:%d", toNumber(entry)),
					"logprob": -0.5 * float64(index+1),
				})
			}
		}
		// A build that ignores `return_tokens_as_token_ids` answers the sampled token in decoded
		// text too, so the fixture does the same rather than being internally inconsistent.
		sampled := "token_id:7"
		if f.mediaTextKeys {
			sampled = "The"
		}
		writeJSON(w, map[string]any{
			"choices": []any{map[string]any{"logprobs": map[string]any{"content": []any{
				map[string]any{"token": sampled, "logprob": -0.1, "top_logprobs": tops},
			}}}},
			"usage": map[string]any{"prompt_tokens": 1234},
		})
	case r.URL.Path == "/generative_scoring":
		f.mu.Lock()
		f.generative++
		f.mu.Unlock()
		var body struct {
			LabelTokenIDs []int `json:"label_token_ids"`
		}
		decodeBody(r, &body)
		// The wanted slot is first in the label list; return a score that makes it identifiable.
		score := 0.25 * float64(len(body.LabelTokenIDs))
		writeJSON(w, map[string]any{"data": []any{map[string]any{"score": score}}})
	default:
		http.NotFound(w, r)
	}
}

// runeIDs tokenizes one token per rune. It is what makes the answer boundary hold by
// construction and every slot id predictable, so every fake tokenizer here uses it.
func runeIDs(text string) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		ids = append(ids, 100+int(r))
	}
	return ids
}

// generativeCalls reads the counter under the lock: the handler runs on the server's goroutine.
func (f *fakeBackend) generativeCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.generative
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func decodeBody(r *http.Request, out any) {
	_ = json.NewDecoder(r.Body).Decode(out)
}

func toNumber(value any) int {
	number, _ := value.(float64)
	return int(number)
}

// modelDir returns a checkpoint directory whose base name matches the served root, so the
// checkpoint guard passes. It carries no tokenizer_config.json, which is what makes these cases
// exercise the `--template auto` default.
func modelDir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "test")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	return dir
}

func startFake(t *testing.T, fake *fakeBackend) string {
	t.Helper()
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)
	return server.URL
}

func testConfig(url, dir string) Config {
	return Config{
		URLs:     []string{url},
		Backend:  backend.ChoiceVLLM,
		Readout:  backend.ReadoutExactSlot,
		Template: template.Choice{Auto: true},
		Timeout:  5 * time.Second,
		Source:   dir,
	}
}

func TestDecideTierAReadsThePinnedSlots(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/test"}
	dir := modelDir(t)
	client, metadata, err := New(context.Background(), testConfig(startFake(t, fake), dir))
	if err != nil {
		t.Fatal(err)
	}
	if !metadata.TierA {
		t.Fatal("tier A was not recorded")
	}
	if metadata.TokenizerMatches == nil || !*metadata.TokenizerMatches {
		t.Fatalf("checkpoint guard did not record a match: %v", metadata.TokenizerMatches)
	}

	scored, err := client.Decide(context.Background(), "row-1", "hello", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	if len(scored.OptionLogprobs) != 2 || scored.OptionLogprobs[0] != -0.5 || scored.OptionLogprobs[1] != -1.0 {
		t.Fatalf("option logprobs = %v", scored.OptionLogprobs)
	}
	if scored.Probabilities[0] <= scored.Probabilities[1] {
		t.Fatalf("probabilities = %v, want the first slot ahead", scored.Probabilities)
	}
	if scored.Readout != "vLLM /v1/completions logprob_token_ids: exact answer slots keyed by token_id, no top-N dependence and no fallback" {
		t.Fatalf("readout = %q", scored.Readout)
	}
	if scored.Modality != "text" || !scored.ServerTokenized {
		t.Fatalf("provenance = %s/%v", scored.Modality, scored.ServerTokenized)
	}
	if scored.PromptSHA256 == nil || scored.InputIDsSHA256 == nil || scored.InputTokens == nil {
		t.Fatal("text provenance digests are missing")
	}
	if scored.FallbackUsed {
		t.Fatal("tier A must not report a fallback")
	}
	if scored.AnswerTokenIDs[0] != 100+'A' || scored.AnswerTokenIDs[1] != 100+'B' {
		t.Fatalf("answer token ids = %v", scored.AnswerTokenIDs)
	}
}

func TestDecideTopNIsRefusedWhenTheBuildKeysByText(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/test", ignorePins: true}
	dir := modelDir(t)
	cfg := testConfig(startFake(t, fake), dir)
	cfg.Readout = backend.ReadoutTopN
	client, _, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Decide(context.Background(), "row-1", "hello", []string{"yes", "no"})
	if err == nil || !strings.Contains(err.Error(), "decoded token text") {
		t.Fatalf("err = %v, want the decoded-text refusal", err)
	}
}

func TestDecideTopNFallsBackToGenerativeScoring(t *testing.T) {
	fake := &fakeBackend{version: "0.25.0", root: "/models/test", topNOnly: true}
	dir := modelDir(t)
	cfg := testConfig(startFake(t, fake), dir)
	cfg.Readout = backend.ReadoutTopN
	client, _, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	scored, err := client.Decide(context.Background(), "row-1", "hello", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	if !scored.FallbackUsed {
		t.Fatal("a slot outside the top-N must report the fallback")
	}
	if scored.FallbackSeconds <= 0 {
		t.Fatal("fallback time was not recorded")
	}
	if !strings.Contains(scored.Readout, "/generative_scoring") {
		t.Fatalf("readout = %q", scored.Readout)
	}
	total := 0.0
	for _, value := range scored.Probabilities {
		total += value
	}
	if total < 0.999 || total > 1.001 {
		t.Fatalf("probabilities do not sum to one: %v", scored.Probabilities)
	}
	if fake.generativeCalls() != 2 {
		t.Fatalf("generative_scoring calls = %d, want one per slot", fake.generativeCalls())
	}
}

func TestMissingPinnedSlotIsRefused(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/test", omitSlot: 100 + 'B'}
	dir := modelDir(t)
	client, _, err := New(context.Background(), testConfig(startFake(t, fake), dir))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Decide(context.Background(), "row-1", "hello", []string{"yes", "no"})
	if err == nil || !strings.Contains(err.Error(), "omitted requested slot") {
		t.Fatalf("err = %v, want the missing-slot refusal", err)
	}
}

func TestExactSlotStartupGateRefusesABuildThatIgnoresPins(t *testing.T) {
	fake := &fakeBackend{version: "0.25.0", root: "/models/test", ignorePins: true}
	dir := modelDir(t)
	_, _, err := New(context.Background(), testConfig(startFake(t, fake), dir))
	if err == nil || !strings.Contains(err.Error(), "did not honor `logprob_token_ids`") {
		t.Fatalf("err = %v, want the exact-slot refusal", err)
	}
}

// The chat template is read from --model, so a checkpoint whose directory name differs from the
// served root is refused unless the operator says the mounts are the same checkpoint.
func TestCheckpointMismatchIsRefused(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/other"}
	dir := modelDir(t)
	_, _, err := New(context.Background(), testConfig(startFake(t, fake), dir))
	if err == nil || !strings.Contains(err.Error(), "checkpoint mismatch") {
		t.Fatalf("err = %v, want the checkpoint refusal", err)
	}

	cfg := testConfig(startFake(t, fake), dir)
	cfg.AllowTokenizerMismatch = true
	if _, _, err := New(context.Background(), cfg); err != nil {
		t.Fatalf("--allow-tokenizer-mismatch must relax the guard: %v", err)
	}
}

// Tokenization is a precondition of serving, not a property of one readout: an SGLang endpoint
// that does not expose its tokenize route must fail at startup even though its own readout
// route is exact-slot by construction.
func TestSGLangWithoutTokenizeIsRefusedAtStartup(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/get_model_info":
			writeJSON(w, map[string]any{"model_path": "/models/test", "context_length": 4096})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	dir := modelDir(t)
	cfg := testConfig(server.URL, dir)
	cfg.Backend = backend.ChoiceSGLang
	// SGLang's adapter is exact-slot by construction; naming the tier keeps ReadoutAuto (which a
	// client must not leave undecided) out of the picture.
	cfg.Readout = backend.ReadoutExactSlot
	_, _, err := New(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "/v1/tokenize") {
		t.Fatalf("err = %v, want the missing-tokenize refusal", err)
	}
}

func TestFleetMustAgreeOnTheVersion(t *testing.T) {
	dir := modelDir(t)
	first := startFake(t, &fakeBackend{version: "0.26.0", root: "/models/test"})
	second := startFake(t, &fakeBackend{version: "0.26.1", root: "/models/test"})
	cfg := testConfig(first, dir)
	cfg.URLs = []string{first, second}
	_, _, err := New(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "Fleet mismatch") {
		t.Fatalf("err = %v, want the fleet refusal", err)
	}
}

func TestSGLangGenerateReadout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/get_model_info":
			writeJSON(w, map[string]any{"model_path": "/models/test", "context_length": 4096, "version": "0.4.0"})
		case "/v1/models":
			writeJSON(w, map[string]any{"data": []any{map[string]any{"id": "served-model"}}})
		case "/v1/tokenize":
			var body struct {
				Prompt string `json:"prompt"`
			}
			decodeBody(r, &body)
			ids := runeIDs(body.Prompt)
			writeJSON(w, map[string]any{"tokens": ids, "count": len(ids), "max_model_len": 4096})
		case "/generate":
			var body struct {
				TokenIDsLogprob []int `json:"token_ids_logprob"`
			}
			decodeBody(r, &body)
			position := make([]any, 0, len(body.TokenIDsLogprob))
			for index, slot := range body.TokenIDsLogprob {
				position = append(position, []any{-0.5 * float64(index+1), slot, "x"})
			}
			writeJSON(w, map[string]any{
				"meta_info": map[string]any{"output_token_ids_logprobs": []any{position}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	dir := modelDir(t)
	cfg := testConfig(server.URL, dir)
	cfg.Backend = backend.ChoiceSGLang
	cfg.Readout = backend.ReadoutExactSlot
	client, metadata, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Backend != backend.KindSGLang || !metadata.TierA {
		t.Fatalf("metadata = %s/%v", metadata.Backend, metadata.TierA)
	}
	scored, err := client.Decide(context.Background(), "row-1", "hello", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	if scored.OptionLogprobs[0] != -0.5 || scored.OptionLogprobs[1] != -1.0 {
		t.Fatalf("option logprobs = %v", scored.OptionLogprobs)
	}
	if !strings.Contains(scored.Readout, "SGLang native /generate") {
		t.Fatalf("readout = %q", scored.Readout)
	}
}

// mediaImage parses one image the way the request path does, so the readout is exercised with the
// value a real request would produce rather than a hand-built stand-in.
func mediaImage(t *testing.T) []media.Media {
	t.Helper()
	content, err := media.ReadContent(map[string]any{
		"role": "user",
		"content": []any{map[string]any{
			"type":      "image_url",
			"image_url": map[string]any{"url": media.ProbeImageDataURI},
		}},
	}, 0, media.Limits{AllowMedia: true, MaxImages: 1, MaxMediaBytes: 4096})
	if err != nil {
		t.Fatal(err)
	}
	return content.Media
}

func mediaClient(t *testing.T, fake *fakeBackend, policy MediaPolicy) (*Client, *Metadata) {
	t.Helper()
	dir := modelDir(t)
	cfg := testConfig(startFake(t, fake), dir)
	cfg.Media = policy
	client, metadata, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return client, metadata
}

func TestDecideMediaExactSlotReadsThePinnedSlots(t *testing.T) {
	client, metadata := mediaClient(t, &fakeBackend{version: "0.26.0", root: "/models/test"}, MediaExactSlot)
	if metadata.MediaSupport != backend.MediaChatExactSlot || metadata.MediaServingConfig != ServingVLLMMediaTokenIDs {
		t.Fatalf("media = %s/%s", metadata.MediaSupport, metadata.MediaServingConfig)
	}

	scored, err := client.DecideMedia(context.Background(), "row-1", "judge it", `{"evidence": "x"}`, mediaImage(t), []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	if scored.OptionLogprobs[0] != -0.5 || scored.OptionLogprobs[1] != -1.0 {
		t.Fatalf("option logprobs = %v", scored.OptionLogprobs)
	}
	if scored.Modality != "text+image" || !scored.ServerTokenized {
		t.Fatalf("provenance = %s/%v", scored.Modality, scored.ServerTokenized)
	}
	// A media prompt is tokenized by the backend, so the prompt digests do not exist and the
	// digest of the request body takes their place.
	if scored.PromptSHA256 != nil || scored.InputIDsSHA256 != nil || scored.RequestSHA256 == nil {
		t.Fatalf("digests = %v/%v/%v", scored.PromptSHA256, scored.InputIDsSHA256, scored.RequestSHA256)
	}
	if scored.InputTokens == nil || *scored.InputTokens != 1234 {
		t.Fatalf("input tokens = %v, want the backend's own count", scored.InputTokens)
	}
	provenance, ok := scored.Media[0].(map[string]any)
	if !ok || provenance["kind"] != "image" || provenance["sha256"] == nil {
		t.Fatalf("provenance = %v", scored.Media)
	}
	if !strings.Contains(scored.Readout, "over a server-tokenized image prompt") {
		t.Fatalf("readout = %q", scored.Readout)
	}
}

func TestDecideMediaTopNReadsBothSlots(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/test", mediaPinsRefused: true}
	client, metadata := mediaClient(t, fake, MediaTopN)
	if metadata.MediaSupport != backend.MediaChatTopN || metadata.MediaServingConfig != ServingVLLMMediaTopN {
		t.Fatalf("media = %s/%s", metadata.MediaSupport, metadata.MediaServingConfig)
	}

	scored, err := client.DecideMedia(context.Background(), "row-1", "s", "p", mediaImage(t), []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	if scored.OptionLogprobs[0] != -0.5 || scored.OptionLogprobs[1] != -1.0 {
		t.Fatalf("option logprobs = %v", scored.OptionLogprobs)
	}
	if !strings.Contains(scored.Readout, "top-N over a server-tokenized image prompt") {
		t.Fatalf("readout = %q", scored.Readout)
	}
}

// An image prompt has no /generative_scoring fallback, so a slot outside the returned top-N must
// refuse rather than renormalize over the slots that happened to appear.
func TestDecideMediaTopNRefusesASlotOutsideTheTopN(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/test", mediaPinsRefused: true, mediaTopNOnly: true}
	client, _ := mediaClient(t, fake, MediaTopN)
	_, err := client.DecideMedia(context.Background(), "row-1", "s", "p", mediaImage(t), []string{"yes", "no"})
	if err == nil || !strings.Contains(err.Error(), "outside the returned top-N") {
		t.Fatalf("err = %v, want the top-N refusal", err)
	}
}

// The media policy, and what the probe can actually read.
//
//   - A build whose chat answers are keyed by decoded text cannot be addressed by this service at
//     all (there is no local tokenizer), so media is off whatever the policy.
//   - A build that refuses the pinned form while answering the unpinned one by id offers only the
//     best-effort route: `--allow-media` (which asks for the exact-slot route) must not accept it.
//     `--allow-media-topn` is that opt-in.
func TestMediaPolicyAndWhatTheProbeCanRead(t *testing.T) {
	decoded := &fakeBackend{version: "0.26.0", root: "/models/test", mediaTextKeys: true}
	for _, policy := range []MediaPolicy{MediaExactSlot, MediaTopN} {
		client, metadata := mediaClient(t, decoded, policy)
		if metadata.MediaSupport != backend.MediaNone || metadata.MediaServingConfig != "" {
			t.Fatalf("decoded-text build: media = %s/%q, want none", metadata.MediaSupport, metadata.MediaServingConfig)
		}
		// And a media decision is refused at the door rather than scored against a route that
		// was never proven.
		if _, err := client.DecideMedia(context.Background(), "row-1", "s", "p", mediaImage(t), []string{"yes", "no"}); err == nil {
			t.Fatal("a build without a media route must refuse a media decision")
		}
	}

	bestEffort := &fakeBackend{version: "0.26.0", root: "/models/test", mediaPinsRefused: true}
	_, strict := mediaClient(t, bestEffort, MediaExactSlot)
	if strict.MediaSupport != backend.MediaNone {
		t.Fatalf("media = %s, want none: --allow-media must not accept the best-effort media route", strict.MediaSupport)
	}
	_, widened := mediaClient(t, bestEffort, MediaTopN)
	if widened.MediaSupport != backend.MediaChatTopN || widened.MediaServingConfig != ServingVLLMMediaTopN {
		t.Fatalf("media = %s/%q, want the best-effort route once the operator allows it", widened.MediaSupport, widened.MediaServingConfig)
	}
}

func TestDecideMediaRefusesWithoutSupportOrImages(t *testing.T) {
	// Media was never enabled: an image must be refused rather than sent to a deployment that was
	// not probed for one.
	plain, _ := mediaClient(t, &fakeBackend{version: "0.26.0", root: "/models/test"}, MediaOff)
	_, err := plain.DecideMedia(context.Background(), "row-1", "s", "p", mediaImage(t), []string{"yes", "no"})
	if err == nil || !strings.Contains(err.Error(), "not probed as able to score an image") {
		t.Fatalf("err = %v, want the no-support refusal", err)
	}

	withMedia, _ := mediaClient(t, &fakeBackend{version: "0.26.0", root: "/models/test"}, MediaExactSlot)
	if _, err := withMedia.DecideMedia(context.Background(), "row-1", "s", "p", nil, []string{"yes", "no"}); err == nil || !strings.Contains(err.Error(), "at least one media part") {
		t.Fatalf("err = %v, want the empty-media refusal", err)
	}
}

func TestPromptOverTheContextLimitIsRefused(t *testing.T) {
	fake := &fakeBackend{version: "0.26.0", root: "/models/test"}
	dir := modelDir(t)
	cfg := testConfig(startFake(t, fake), dir)
	cfg.MaxTokens = 8
	client, _, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Decide(context.Background(), "row-1", "a prompt far longer than the ceiling", []string{"yes", "no"})
	if err == nil || !strings.Contains(err.Error(), "no truncation allowed") {
		t.Fatalf("err = %v, want the length refusal", err)
	}
}

func TestUnknownCeilingIsRefusedAtStartup(t *testing.T) {
	// A backend that reports no max_model_len and no local ceiling leaves the prompt bound
	// unknown, which is refused rather than guessed.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/version":
			writeJSON(w, map[string]any{"version": "1"})
		case "/v1/models":
			writeJSON(w, map[string]any{"data": []any{map[string]any{"id": "m", "root": "/models/test"}}})
		case "/tokenize":
			var body struct {
				Prompt string `json:"prompt"`
			}
			decodeBody(r, &body)
			ids := runeIDs(body.Prompt)
			// No max_model_len: this endpoint is the one whose ceiling is unknown.
			writeJSON(w, map[string]any{"tokens": ids, "count": len(ids)})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	dir := modelDir(t)
	cfg := testConfig(server.URL, dir)
	cfg.Backend = backend.ChoiceVLLM
	// top-n skips the Tier A probe, so the run reaches the ceiling check this test is about.
	cfg.Readout = backend.ReadoutTopN
	_, _, err := New(context.Background(), cfg)
	if err == nil || !strings.Contains(err.Error(), "ceiling is unknown") {
		t.Fatalf("err = %v, want the unknown-ceiling refusal", err)
	}
}

// A tokenize failure is the backend's failure, not the caller's. Reporting it as a contract
// refusal would tell the caller to fix a request that was fine, and would hide the backend from
// the error metrics. The fake answers 500 on /tokenize, which the readout must preserve as a
// backend error.
func TestTokenizeBackendFailureIsNotAContractError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"tokenizer down"}`))
	}))
	t.Cleanup(server.Close)

	transport := backend.NewClient(server.URL, 5*time.Second)
	client := &Client{
		kind: backend.KindVLLM, tierA: true, contextLimit: 4096,
		endpoints: []endpoint{{
			info:      EndpointInfo{ServedModels: []string{"served-model"}},
			transport: transport,
			tokenizer: tokenize.New(transport, backend.KindVLLM, "served-model"),
		}},
	}
	_, err := client.Decide(context.Background(), "row-1", "hello", []string{"yes", "no"})
	var contract *ContractError
	if errors.As(err, &contract) {
		t.Fatalf("a backend tokenize failure was reported as a contract error: %v", err)
	}
	var status *backend.StatusError
	if !errors.As(err, &status) || status.Code != http.StatusInternalServerError {
		t.Fatalf("err = %v, want the backend 500 preserved", err)
	}
}

// The library entry point must refuse the same client-side misconfigurations the binary does
// instead of reading them as a different deployment: an omitted dialect used to be probed as
// vLLM, an omitted tier used to fall through to top-n, and a zero budget used to leave the HTTP
// client unbounded.
func TestNewRefusesAnUnusableClientConfig(t *testing.T) {
	server := startFake(t, &fakeBackend{version: "0.26.0", root: "/models/test"})
	dir := modelDir(t)
	cases := []struct {
		name string
		tune func(*Config)
		want string
	}{
		{"omitted backend", func(c *Config) { c.Backend = "" }, "unknown backend"},
		{"unknown backend", func(c *Config) { c.Backend = "tgi" }, "unknown backend"},
		{"omitted readout", func(c *Config) { c.Readout = "" }, "unknown readout"},
		{"auto readout", func(c *Config) { c.Readout = backend.ReadoutAuto }, "auto picks a different tier"},
		{"zero timeout", func(c *Config) { c.Timeout = 0 }, "Timeout must be positive"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig(server, dir)
			tc.tune(&cfg)
			if _, _, err := New(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}
