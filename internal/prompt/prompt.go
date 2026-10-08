// Package prompt owns the direct-options-v1 content: the validated row, the
// {evidence, criterion, options} payload, and the option-to-letter binding.
//
// The rendered wrapper is not this package's business — internal/template owns the family
// registry — while this package owns the content. PromptVersion here plus the template family
// recorded in the response are what identify the string the server actually received.
package prompt

import (
	"fmt"

	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/schema"
	"github.com/wweir/weigh/internal/slots"
	"github.com/wweir/weigh/internal/template"
)

// PromptVersion versions the content: the system text and the user JSON.
const PromptVersion = "direct-options-v1"

// The two renderers a request's `messages` array can be folded into. They produce different
// prompt strings for the same request, so every response records which one ran.
const (
	// EvidenceCanonical is the canonical [system, user] pair: the evidence is the single user
	// message verbatim.
	EvidenceCanonical = "system+user"
	// EvidenceTranscript is the general fold used for every other array shape (multi-turn
	// history, tool results, several system messages).
	EvidenceTranscript = "transcript"
)

// DirectSystem is the direct-options-v1 system prompt, verbatim.
const DirectSystem = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."

// Option is one choice in a row.
type Option struct {
	ID          string
	Description string
}

// Row is one validated decision request.
type Row struct {
	ID       string
	State    string
	Question string
	Options  []Option
}

// ValidateRow validates one direct-options-v1 row.
//
// Deliberately strict: the accepted row shapes are part of the published contract, so a caller
// cannot get a scored answer out of a row the contract does not describe.
func ValidateRow(value any) (*Row, error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("Row must be a JSON object")
	}
	for _, key := range []string{"id", "state", "question", "options"} {
		if _, present := object[key]; !present {
			return nil, fmt.Errorf("Row is missing fields: [%q]", key)
		}
	}

	id, ok := nonEmptyString(object["id"])
	if !ok {
		return nil, fmt.Errorf("id and question must be nonempty strings")
	}
	question, ok := nonEmptyString(object["question"])
	if !ok {
		return nil, fmt.Errorf("id and question must be nonempty strings")
	}

	// The state is the evidence. It is a string and not raw JSON because the serving path folds
	// every accepted message shape into one text; a decoded object would have no key order, and
	// the payload's byte layout is part of the prompt.
	state, ok := object["state"].(string)
	if !ok || state == "" {
		return nil, fmt.Errorf("state must be a nonempty string")
	}

	optionValues, ok := object["options"].([]any)
	if !ok || len(optionValues) < schema.MinValues || len(optionValues) > schema.MaxValues {
		return nil, fmt.Errorf("options must contain %d-%d entries", schema.MinValues, schema.MaxValues)
	}

	options := make([]Option, 0, len(optionValues))
	seen := make(map[string]bool, len(optionValues))
	for _, entry := range optionValues {
		item, ok := entry.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("Each option needs string id and description fields")
		}
		optionID, ok := item["id"].(string)
		if !ok {
			return nil, fmt.Errorf("Each option needs string id and description fields")
		}
		description, ok := item["description"].(string)
		if !ok {
			return nil, fmt.Errorf("Each option needs string id and description fields")
		}
		if seen[optionID] {
			return nil, fmt.Errorf("Option IDs must be unique")
		}
		seen[optionID] = true
		options = append(options, Option{ID: optionID, Description: description})
	}

	return &Row{ID: id, State: state, Question: question, Options: options}, nil
}

// UserMessage builds the user message for one validated row.
func UserMessage(row *Row) (string, error) {
	descriptions := make([]string, len(row.Options))
	for i, option := range row.Options {
		descriptions[i] = option.Description
	}
	return userPayload(row.State, row.Question, descriptions)
}

// MediaUserPayload is the text part of a media request: the same payload the text path sends,
// carrying the *text* evidence. The images become sibling content parts after it.
func MediaUserPayload(evidenceText, criterion string, optionIDs []string) (string, error) {
	return userPayload(evidenceText, criterion, optionIDs)
}

// RenderPromptWith renders the chat template for one system + user turn.
func RenderPromptWith(family template.ChatTemplate, row *Row) (string, error) {
	user, err := UserMessage(row)
	if err != nil {
		return "", err
	}
	return family.Render(DirectSystem, user), nil
}

// userPayload is the direct-options-v1 user payload: {evidence, criterion, options}, in that
// order, with every option bound to its slot letter.
//
// One builder for both paths, because the media payload is the same JSON with a different
// *evidence* source — not a second contract.
func userPayload(evidence string, criterion string, descriptions []string) (string, error) {
	options := make([]any, len(descriptions))
	for i, description := range descriptions {
		options[i] = jsonx.Object{
			{Key: "letter", Value: slots.Letter(i)},
			{Key: "description", Value: description},
		}
	}
	return jsonx.Dump(jsonx.Object{
		{Key: "evidence", Value: evidence},
		{Key: "criterion", Value: criterion},
		{Key: "options", Value: options},
	})
}

func nonEmptyString(value any) (string, bool) {
	text, ok := value.(string)
	return text, ok && text != ""
}
