package weigh

import (
	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/prompt"
	"github.com/wweir/weigh/internal/template"
)

// DirectSystem is the direct-options-v1 system prompt, verbatim.
//
// The criterion belongs in the user message on this contract, not here; the system turn states the
// task. [RenderDirectOptions] pairs it with [UserMessage].
const DirectSystem = prompt.DirectSystem

// PromptVersion versions the content of this package's default assembly: DirectSystem plus the
// [UserMessage] payload. It is the string weighd reports as `prompt_version`. A caller that
// renders anything else is free to do so — [Decide] does not check it — but must not report this
// version for it.
const PromptVersion = prompt.PromptVersion

// ValidateRow validates one direct-options-v1 row from decoded JSON.
//
// Deliberately strict: the accepted row shapes are part of the published contract, so a caller
// cannot get a scored answer out of a row the contract does not describe.
func ValidateRow(value any) (*Row, error) { return prompt.ValidateRow(value) }

// UserMessage builds the direct-options-v1 user message for one validated row: the evidence, the
// criterion, and every option bound to its slot letter, as ordered JSON.
func UserMessage(row *Row) (string, error) { return prompt.UserMessage(row) }

// MediaUserPayload is the text half of a media request's user message: the same payload
// [UserMessage] builds, with the image references left out. The images travel beside it as content
// parts, so the payload carries the *text* evidence.
func MediaUserPayload(evidenceText, criterion string, optionIDs []string) (string, error) {
	return prompt.MediaUserPayload(evidenceText, criterion, optionIDs)
}

// RenderPrompt wraps one system turn plus one user turn in a family's bytes, ending at the
// assistant-turn opener. Everything after that point is the single scored token.
//
// Nothing here interpolates or escapes: a family is a fixed prefix, the two turns, and the opener.
// The caller owns both strings.
//
// A family this build cannot render is an error rather than a panic: [Metadata.PromptTemplate] is
// empty when no template was configured, and feeding that back here is a mistake worth naming.
func RenderPrompt(family ChatTemplate, system, user string) (string, error) {
	if err := checkFamily(family); err != nil {
		return "", err
	}
	return family.Render(system, user), nil
}

// checkFamily refuses a family this build does not render. The internal renderer would otherwise
// reach its "unknown family" panic, which is right for a value the flag parser produced and wrong
// for one a caller typed.
func checkFamily(family ChatTemplate) error {
	_, err := template.Parse(string(family))
	return err
}

// RenderDirectOptions renders the default contract: DirectSystem as the system turn and
// [UserMessage] as the user turn. It is the prompt weighd sends for a text decision.
//
// The option order is what binds the slots: row.Options[i] is answer slot slots.Letter(i), so the
// optionIDs passed to [Decide] must be row.Options[i].ID in the same order.
func RenderDirectOptions(family ChatTemplate, row *Row) (string, error) {
	if err := checkFamily(family); err != nil {
		return "", err
	}
	return prompt.RenderPromptWith(family, row)
}

// DetectTemplate returns the family a chat_template string belongs to, by distinctive marker. The
// second result is false both for a shape this package does not know and for one it knows it must
// not render; use [ResolveTemplate] when the refusal reason matters.
func DetectTemplate(chatTemplate string) (ChatTemplate, bool) { return template.Detect(chatTemplate) }

// ResolveTemplate decides the family for a local checkpoint directory, by reading
// {modelDir}/tokenizer_config.json.
//
// It is what Config.Source drives, exposed for a caller that renders its own prompt and wants its
// wrapper to match what the checkpoint declares. A file whose template matches no known family is
// an error, not a default.
func ResolveTemplate(choice TemplateChoice, modelDir string) (Resolved, error) {
	return template.Resolve(choice, modelDir)
}

// ParseTemplateChoice parses a family name, or TemplateAuto, into a TemplateChoice.
func ParseTemplateChoice(text string) (TemplateChoice, error) { return template.ParseChoice(text) }

// ParseBackendChoice parses a BackendChoice name.
func ParseBackendChoice(text string) (BackendChoice, error) { return backend.ParseChoice(text) }

// ParseReadout parses a Readout name. ReadoutAuto is accepted here and refused by [New] (and by
// the serving path), which is where "it picks a different tier per backend" has to be said.
func ParseReadout(text string) (Readout, error) { return backend.ParseReadout(text) }
