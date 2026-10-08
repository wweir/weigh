package template

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	system = "Apply the criterion."
	user   = `{"evidence": "x", "criterion": "y", "options": []}`
)

func TestGemma4RenderingIsTheHistoricalConstruction(t *testing.T) {
	want := "<bos><|turn>system\n" + system + "<turn|>\n<|turn>user\n" + user + "<turn|>\n<|turn>model\n"
	if got := Gemma4.Render(system, user); got != want {
		t.Fatalf("Render() = %q, want %q", got, want)
	}
}

func TestEveryFamilyOpensAnAssistantTurnAfterTheUserTurn(t *testing.T) {
	cases := []struct {
		family ChatTemplate
		suffix string
	}{
		{Gemma4, "<|turn>model\n"},
		{Gemma3, "<start_of_turn>model\n"},
		{Chatml, "<|im_start|>assistant\n"},
		{ChatmlThinking, " thinking\n\n</think>\n\n"},
		{Llama3, "<|start_header_id|>assistant<|end_header_id|>\n\n"},
		{Llama2, " [/INST]"},
		{Mistral, " [/INST]"},
	}
	for _, tc := range cases {
		prompt := tc.family.Render(system, user)
		if !strings.HasSuffix(prompt, tc.suffix) {
			t.Errorf("%s: prompt does not end with %q: %q", tc.family, tc.suffix, prompt)
		}
		if !strings.Contains(prompt, system) {
			t.Errorf("%s lost the system turn", tc.family)
		}
		if !strings.Contains(prompt, user) {
			t.Errorf("%s lost the user turn", tc.family)
		}
		if strings.Index(prompt, system) > strings.Index(prompt, user) {
			t.Errorf("%s put the user turn first", tc.family)
		}
	}
}

// The exact bytes the real templates produce, for the families that are easy to get subtly
// wrong: the Qwen3 thinking block and the Mistral v0.2 spacing.
func TestFixedFamiliesRenderWhatTheirRealTemplatesProduce(t *testing.T) {
	if got, want := ChatmlThinking.Render("S", "U"),
		"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n"; got != want {
		t.Errorf("chatml-thinking = %q, want %q", got, want)
	}
	if got, want := Chatml.Render("S", "U"),
		"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"; got != want {
		t.Errorf("chatml = %q, want %q", got, want)
	}
	if got, want := Mistral.Render("S", "U"), "<s> [INST] S\n\nU [/INST]"; got != want {
		t.Errorf("mistral = %q, want %q", got, want)
	}
}

func TestDetectionIsMarkerBased(t *testing.T) {
	cases := []struct {
		chatTemplate string
		want         ChatTemplate
	}{
		{"{{ '<|turn>system' }}", Gemma4},
		{"{{ '<start_of_turn>user' }}", Gemma3},
		{"{{ '<|im_start|>system' }}", Chatml},
		{"{{ '<|im_start|>assistant\\n' }}{% if enable_thinking is false %}{{ ' thinking\\n\\n</think>\\n\\n' }}{% endif %}", ChatmlThinking},
		{"{{ '<|start_header_id|>user<|end_header_id|>' }}", Llama3},
		{"{{ '<<SYS>>' }}", Llama2},
		{"{{ ' [INST] ' + system + ' [/INST]' }}", Mistral},
	}
	for _, tc := range cases {
		if got, ok := Detect(tc.chatTemplate); !ok || got != tc.want {
			t.Errorf("Detect(%q) = %q/%v, want %q", tc.chatTemplate, got, ok, tc.want)
		}
	}
	if family, ok := Detect("{{ messages[0]['content'] }}"); ok {
		t.Errorf("an unknown shape must not be detected, got %q", family)
	}
}

func TestNamesRoundTripThroughParse(t *testing.T) {
	for _, name := range Names {
		family, err := Parse(name)
		if err != nil {
			t.Fatalf("Parse(%q): %v", name, err)
		}
		if string(family) != name {
			t.Errorf("Parse(%q) = %q", name, family)
		}
	}
	if _, err := Parse("gpt"); err == nil {
		t.Error("an unknown name must be refused")
	}
	if choice, err := ParseChoice("auto"); err != nil || !choice.Auto {
		t.Errorf("ParseChoice(auto) = %+v/%v", choice, err)
	}
	if _, err := ParseChoice("nope"); err == nil {
		t.Error("ParseChoice(nope) must be refused")
	}
}

// resolveCase is one row of the resolve decision table: what the file declares, what the
// operator asked for, and which of the two wins.
type resolveCase struct {
	name       string
	config     string
	hasConfig  bool
	choice     Choice
	wantFamily ChatTemplate
	wantSource string
	wantError  string
}

func TestResolveMatchesTheDeclaredFileAgainstTheOperatorChoice(t *testing.T) {
	const (
		gemma2    = `{"chat_template": "{{ '<start_of_turn>user' }}{{ raise_exception('System role not supported') }}"}`
		mistralV3 = `{"chat_template": "{{ \"[INST] \" + content + \"[/INST]\" }}"}`
	)
	cases := []resolveCase{
		{"nothing declared", "", false, Choice{Auto: true}, Gemma4, "default:no-chat-template", ""},
		{
			"declared, array encoding",
			`{"chat_template": [{"name": "default", "template": "{{ '<|im_start|>system' }}"}]}`,
			true, Choice{Auto: true}, Chatml, "detected:", "",
		},
		{
			"explicit agreement",
			`{"chat_template": "{{ ' [INST] ' + x + ' [/INST]' }}"}`,
			true, Choice{Fixed: Mistral}, Mistral, "explicit", "",
		},
		{"nothing matches", `{"chat_template": "{{ weird }}"}`, true, Choice{Auto: true}, "", "", "matches no supported family"},
		{
			"present but unreadable",
			`{"chat_template": [{"name": "default"}]}`,
			true, Choice{Auto: true}, "", "", "neither a string nor an array",
		},
		{
			"explicit contradicts the file",
			`{"chat_template": "{{ '<|im_start|>system' }}"}`,
			true, Choice{Fixed: Gemma4}, "", "", "describes a chatml chat template",
		},
		{"gemma-2 refuses the system role", gemma2, true, Choice{Auto: true}, "", "", "system role"},
		{"gemma-2, named anyway", gemma2, true, Choice{Fixed: Gemma3}, "", "", "system role"},
		{"mistral v0.3 spacing, named as v0.1/v0.2", mistralV3, true, Choice{Fixed: Mistral}, "", "", "spacing"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.hasConfig {
				if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(tc.config), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			resolved, err := Resolve(tc.choice, dir)
			switch {
			case tc.wantError != "":
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("err = %v, want it to contain %q", err, tc.wantError)
				}
			case err != nil:
				t.Fatalf("unexpected error: %v", err)
			default:
				if resolved.Template != tc.wantFamily {
					t.Errorf("family = %q, want %q", resolved.Template, tc.wantFamily)
				}
				if !strings.HasPrefix(resolved.Source, tc.wantSource) {
					t.Errorf("source = %q, want it to start with %q", resolved.Source, tc.wantSource)
				}
			}
		})
	}
}

func TestResolveRefusesAMalformedConfigFile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Resolve(Choice{Auto: true}, dir); err == nil || !strings.Contains(err.Error(), "not valid JSON") {
		t.Fatalf("err = %v, want a JSON refusal", err)
	}
}
