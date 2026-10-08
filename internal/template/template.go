// Package template is the chat-template registry: the wrappers that turn one system message plus
// one user message into the exact string the model was trained to see.
//
// A registry instead of a Jinja engine: the obvious implementation reads
// tokenizer_config.json's chat_template and renders it with a template engine, but a real
// template is kilobytes of Jinja referencing hundreds of names, while the served shape is
// always one plain system turn plus one plain user turn, no tools, no images. Each family's
// reachable subgraph is therefore a fixed prefix, an assistant-turn opener, and two
// interpolations, written out and named here.
//
// Detection is not a guess: a file whose chat_template matches no known family is an error, not
// a silent fall back to the default wrapper. Rendering the wrong wrapper yields a plausible
// prompt the model was never trained on, and the answer-boundary proof only validates the last
// token, so it can pass anyway.
package template

import (
	"fmt"
	"strings"
)

// Names are the families this service can render, in the order they appear in --help.
var Names = []string{
	"gemma4", "gemma3", "chatml", "chatml-thinking", "llama3", "llama2", "mistral",
}

// ChatTemplate identifies one rendered family.
type ChatTemplate string

const (
	// Gemma4 is the gemma-4 `<|turn>` format and the historical default.
	Gemma4 ChatTemplate = "gemma4"
	// Gemma3 is the gemma-3 `<start_of_turn>` format.
	Gemma3 ChatTemplate = "gemma3"
	// Chatml is ChatML without a thinking-mode gate: Qwen2/2.5, Yi, and many others.
	Chatml ChatTemplate = "chatml"
	// ChatmlThinking is ChatML whose template gates the generation prompt on enable_thinking:
	// Qwen3/3.5. Rendering plain ChatML here would prompt the model into a mode nobody asked
	// for.
	ChatmlThinking ChatTemplate = "chatml-thinking"
	// Llama3 is the Llama-3.x `<|start_header_id|>` format.
	Llama3 ChatTemplate = "llama3"
	// Llama2 is the Llama-2-chat `[INST] <<SYS>>` format.
	Llama2 ChatTemplate = "llama2"
	// Mistral is the Mistral-7B-Instruct v0.1/v0.2 `[INST]` format, system folded into the
	// user turn.
	Mistral ChatTemplate = "mistral"
)

// Parse turns a --template value into a family.
func Parse(text string) (ChatTemplate, error) {
	for _, name := range Names {
		if text == name {
			return ChatTemplate(name), nil
		}
	}
	return "", fmt.Errorf("unknown template %q; expected auto or one of %s", text, strings.Join(Names, ", "))
}

// Detect returns the family a chat_template string belongs to, by distinctive marker.
//
// The second result is false both for "a shape this service does not know" and for "a shape it
// knows it must not render"; UnrenderableReason tells those apart for a caller that must say
// why.
func Detect(chatTemplate string) (ChatTemplate, bool) {
	if _, refused := UnrenderableReason(chatTemplate); refused {
		return "", false
	}
	// The markers are mutually exclusive in practice; gemma-4 is checked before gemma-3
	// because they are different strings.
	switch {
	case strings.Contains(chatTemplate, "<|turn>"):
		return Gemma4, true
	case strings.Contains(chatTemplate, "<start_of_turn>"):
		return Gemma3, true
	case strings.Contains(chatTemplate, "<|im_start|>"):
		// Qwen3/3.5 gate the generation prompt on enable_thinking and emit an empty thinking
		// block when it is false; plain ChatML (Qwen2/2.5) does neither.
		if strings.Contains(chatTemplate, "enable_thinking") && strings.Contains(chatTemplate, " thinking") {
			return ChatmlThinking, true
		}
		return Chatml, true
	case strings.Contains(chatTemplate, "<|start_header_id|>"):
		return Llama3, true
	case strings.Contains(chatTemplate, "<<SYS>>"):
		return Llama2, true
	case strings.Contains(chatTemplate, "[INST]"):
		// Only the v0.1/v0.2 bytes are rendered here; every other [INST] shape was already
		// refused above.
		return Mistral, true
	}
	return "", false
}

// Render produces the reachable subgraph for one system turn plus one user turn, ending at the
// assistant turn opener. Everything after that point is the single scored token.
func (t ChatTemplate) Render(system, user string) string {
	var b strings.Builder
	b.Grow(len(system) + len(user) + 256)
	switch t {
	case Gemma4:
		// The historical construction verbatim: its template wraps both turns in
		// `{{- ... | trim -}}`, hence the trims.
		b.WriteString("<bos><|turn>system\n")
		b.WriteString(strings.TrimSpace(system))
		b.WriteString("<turn|>\n<|turn>user\n")
		b.WriteString(strings.TrimSpace(user))
		b.WriteString("<turn|>\n<|turn>model\n")
	case Gemma3:
		b.WriteString("<bos><start_of_turn>user\n")
		b.WriteString(system)
		b.WriteString("\n\n")
		b.WriteString(user)
		b.WriteString("<end_of_turn>\n<start_of_turn>model\n")
	case Chatml:
		b.WriteString("<|im_start|>system\n")
		b.WriteString(system)
		b.WriteString("<|im_end|>\n<|im_start|>user\n")
		b.WriteString(user)
		b.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	case ChatmlThinking:
		// The Chatml bytes plus the empty thinking block: the Qwen3 template's
		// add_generation_prompt branch with enable_thinking false.
		b.WriteString("<|im_start|>system\n")
		b.WriteString(system)
		b.WriteString("<|im_end|>\n<|im_start|>user\n")
		b.WriteString(user)
		b.WriteString("<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n")
	case Llama3:
		b.WriteString("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
		b.WriteString(system)
		b.WriteString("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")
		b.WriteString(user)
		b.WriteString("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
	case Llama2:
		b.WriteString("<s>[INST] <<SYS>>\n")
		b.WriteString(system)
		b.WriteString("\n<</SYS>>\n\n")
		b.WriteString(user)
		b.WriteString(" [/INST]")
	case Mistral:
		// Mistral-7B-Instruct v0.1/v0.2: after the <s> bos token the tag is `' [INST] '`, so
		// there is a space on both sides of [INST].
		b.WriteString("<s> [INST] ")
		b.WriteString(system)
		b.WriteString("\n\n")
		b.WriteString(user)
		b.WriteString(" [/INST]")
	default:
		// Unreachable through the flag parser; a panic rather than an empty prompt, which
		// would be scored as if it meant something.
		panic("template: unknown family " + string(t))
	}
	return b.String()
}

// UnrenderableReason explains why a chat_template this service recognises is still not
// rendered. The second result is false when the shape is not refused for a reason.
func UnrenderableReason(chatTemplate string) (string, bool) {
	// A template that raises for the system role cannot be served faithfully here — the
	// service always sends one — and gemma-2 does exactly that while sharing gemma-3's
	// `<start_of_turn>` marker.
	if strings.Contains(chatTemplate, "System role not supported") {
		return "it raises for the system role (gemma-2), and this service always sends one", true
	}
	// Mistral v0.1/v0.2 write the tag as `' [INST] '`/`' [/INST]'` — a space on both sides;
	// v0.3 writes `"[INST] "`/`"[/INST]"`. Only the v0.1/v0.2 bytes are rendered here.
	if strings.Contains(chatTemplate, "[INST]") &&
		!(strings.Contains(chatTemplate, " [INST]") && strings.Contains(chatTemplate, " [/INST]")) {
		return "its [INST] spacing is Mistral v0.3's, and only v0.1/v0.2 bytes are rendered", true
	}
	return "", false
}
