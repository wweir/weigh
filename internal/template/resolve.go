package template

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Choice is what the operator asked for on the command line.
type Choice struct {
	Auto  bool
	Fixed ChatTemplate
}

// ParseChoice parses --template.
func ParseChoice(text string) (Choice, error) {
	if text == "auto" {
		return Choice{Auto: true}, nil
	}
	fixed, err := Parse(text)
	if err != nil {
		return Choice{}, err
	}
	return Choice{Fixed: fixed}, nil
}

// Resolved is the template actually used, plus how it was decided. Both go into the artifact: a
// defaulted gemma-4 and a detected gemma-4 are the same rendering but different evidence.
type Resolved struct {
	Template ChatTemplate
	// Source is "explicit", "detected:<path>", or "default:no-chat-template".
	Source string
}

// Resolve reads {modelDir}/tokenizer_config.json and decides the template.
//
// The file is optional (a checkpoint may not ship one), but a chat_template that is *present*
// and unreadable is an error: falling back to the default wrapper would score under a template
// the checkpoint never declared.
func Resolve(choice Choice, modelDir string) (Resolved, error) {
	path := filepath.Join(modelDir, "tokenizer_config.json")

	var declared []string
	raw, err := os.ReadFile(path)
	switch {
	case err == nil:
		var value map[string]any
		if err := json.Unmarshal(raw, &value); err != nil {
			return Resolved{}, fmt.Errorf("%s is not valid JSON: %w", path, err)
		}
		declared = extractChatTemplates(value)
		if len(declared) == 0 {
			if _, present := value["chat_template"]; present {
				return Resolved{}, fmt.Errorf("%s declares a `chat_template` that is neither a string nor an array of strings/objects with a `template` key; refusing to fall back to the default wrapper", path)
			}
			declared = nil
		}
	case errors.Is(err, os.ErrNotExist):
		// No file: the default below applies.
	default:
		return Resolved{}, fmt.Errorf("cannot read %s: %w", path, err)
	}

	detected, hasDetected := detectAll(declared)

	if !choice.Auto {
		fixed := choice.Fixed
		if hasDetected {
			if detected != fixed {
				return Resolved{}, fmt.Errorf(
					"--template %s was requested, but %s describes a %s chat template. Rendering the wrong wrapper produces a plausible prompt the model was not trained on, and the answer-boundary check only validates the last token, so it can pass anyway. Pass --template %s or fix --model",
					fixed, path, detected, detected)
			}
		} else if reason, refused := unrenderableIn(declared); refused {
			// The operator named a family, but this file declares a shape this build
			// deliberately does not render — and that shape shares the named family's marker,
			// so honouring the name would render exactly the bytes the refusal exists to
			// prevent.
			return Resolved{}, fmt.Errorf(
				"--template %s was requested, but %s declares a chat_template this build will not render: %s. Point --model at a checkpoint whose template can be reproduced byte for byte",
				fixed, path, reason)
		}
		return Resolved{Template: fixed, Source: "explicit"}, nil
	}

	if hasDetected {
		return Resolved{Template: detected, Source: "detected:" + path}, nil
	}
	if declared != nil {
		// Rune-aware truncation: a byte slice could split a multi-byte character, and the head
		// is quoted into an error message.
		head := declared[0]
		if runes := []rune(head); len(runes) > 200 {
			head = string(runes[:200])
		}
		reason := ""
		if why, refused := unrenderableIn(declared); refused {
			reason = ": " + why
		}
		return Resolved{}, fmt.Errorf(
			"%s declares a chat_template that matches no supported family (%s)%s. Refusing to guess rather than render a wrapper the model was not trained on. Template head: %q",
			path, strings.Join(Names, ", "), reason, head)
	}
	// No tokenizer_config.json, or one without a chat_template: keep the historical default and
	// say so in the source rather than implying it was detected.
	return Resolved{Template: Gemma4, Source: "default:no-chat-template"}, nil
}

func detectAll(templates []string) (ChatTemplate, bool) {
	for _, text := range templates {
		if family, ok := Detect(text); ok {
			return family, true
		}
	}
	return "", false
}

func unrenderableIn(templates []string) (string, bool) {
	for _, text := range templates {
		if reason, ok := UnrenderableReason(text); ok {
			return reason, true
		}
	}
	return "", false
}

// extractChatTemplates reads the two encodings checkpoints ship: a plain string, or an array
// whose entries are strings or objects with a `template` key.
func extractChatTemplates(value map[string]any) []string {
	var out []string
	switch entry := value["chat_template"].(type) {
	case string:
		out = append(out, entry)
	case []any:
		for _, item := range entry {
			switch typed := item.(type) {
			case string:
				out = append(out, typed)
			case map[string]any:
				if text, ok := typed["template"].(string); ok {
					out = append(out, text)
				}
			}
		}
	}
	return out
}
