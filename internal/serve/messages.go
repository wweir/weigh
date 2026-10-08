package serve

import (
	"fmt"
	"strings"

	"github.com/wweir/weigh/internal/media"
	"github.com/wweir/weigh/internal/prompt"
)

// maxMessages bounds `messages`, so a hostile request cannot make the fold quadratic in body
// size. The body limit is already a megabyte; this bounds the per-message work too.
const maxMessages = 512

// maxBatch bounds /v1/semif/batch items. Every item is a full readout against the backend, so
// this bounds the work one request can queue behind a single worker.
const maxBatch = 32

// contractError is a named refusal from the request contract. The caller decides the HTTP status
// and the `param` it is reported under.
type contractError struct {
	code    string
	message string
}

func (e *contractError) Error() string { return e.message }

func refuse(code, format string, args ...any) *contractError {
	return &contractError{code: code, message: fmt.Sprintf(format, args...)}
}

// evidencePart is one ordered piece of the folded evidence.
//
// Text and media are kept apart because a media request sends them as different things: the text
// becomes SemIf's `{evidence, criterion, options}` payload, and each image becomes a sibling
// content part the backend's processor expands.
type evidencePart struct {
	text  string
	image *media.Media
}

// messages is the scored pair plus the renderer that produced it.
type messages struct {
	criterion string
	// evidence is in caller order, so an interleaved evidence message cannot silently become
	// "text then image".
	evidence []evidencePart
	// renderer is which fold produced the evidence; recorded in every response.
	renderer string
}

func (m *messages) hasMedia() bool {
	for index := range m.evidence {
		if m.evidence[index].image != nil {
			return true
		}
	}
	return false
}

// textEvidence is the folded evidence as one string: the text parts concatenated, with any image
// ignored, because a media request sends the text as SemIf's payload and each image as a sibling
// content part.
//
// A media-free request folds to exactly one text part, which is what keeps the text prompt
// byte-identical to the pre-media contract.
func (m *messages) textEvidence() string {
	var out strings.Builder
	for index := range m.evidence {
		out.WriteString(m.evidence[index].text)
	}
	return out.String()
}

// evidenceIdentity is a stable identity for the completion id: two requests whose evidence differs
// only by which image is attached must not share one.
func (m *messages) evidenceIdentity() string {
	parts := make([]string, 0, len(m.evidence))
	for index := range m.evidence {
		part := &m.evidence[index]
		if part.image == nil {
			parts = append(parts, part.text)
			continue
		}
		if part.image.SHA256 != nil {
			parts = append(parts, *part.image.SHA256)
		} else {
			parts = append(parts, part.image.Reference)
		}
	}
	return strings.Join(parts, "\u001f")
}

// media returns the images in caller order.
func (m *messages) media() []media.Media {
	out := make([]media.Media, 0, len(m.evidence))
	for index := range m.evidence {
		if m.evidence[index].image != nil {
			out = append(out, *m.evidence[index].image)
		}
	}
	return out
}

// readMessages folds an OpenAI-shaped `messages` array into the (criterion, evidence) pair this
// engine scores.
//
// The canonical two-message [system, user] form is preserved exactly (renderer `system+user`), so
// widening the accepted shapes cannot move a single byte of the prompt for a request that already
// worked. Every other shape becomes a role-labelled transcript: the criterion is every
// system/developer message joined in order, and the evidence is every remaining message rendered
// as `role: text`.
func readMessages(object map[string]any, limits media.Limits) (*messages, *contractError) {
	raw, ok := object["messages"].([]any)
	if !ok {
		return nil, refuse("messages_required", "`messages` is required")
	}
	if len(raw) == 0 {
		return nil, refuse("messages_required", "`messages` must not be empty")
	}
	if len(raw) > maxMessages {
		return nil, refuse("messages_not_semif_contract", "`messages` has %d entries; the limit is %d", len(raw), maxMessages)
	}

	// Fold every message up front, so an unsupported part is reported by its index.
	shapes := make([]map[string]any, len(raw))
	contents := make([]*media.Content, len(raw))
	for index, entry := range raw {
		// A non-object element has no role, so it is reported as the missing role it is.
		shapes[index], _ = entry.(map[string]any)
		if roleOf(shapes[index]) == "" {
			return nil, refuse("messages_not_semif_contract", "messages[%d] needs a nonempty `role`", index)
		}
		content, err := media.ReadContent(shapes[index], index, limits)
		if err != nil {
			return nil, &contractError{code: err.Code, message: err.Message}
		}
		contents[index] = content
	}

	isCriterion := func(shape map[string]any) bool {
		switch roleOf(shape) {
		case "system", "developer":
			return true
		default:
			return false
		}
	}

	// Per *request*, not per message: ReadContent caps one message's images, but a long transcript
	// could otherwise attach one image per turn and blow past the ceiling the operator set for the
	// whole request.
	images := 0
	for _, content := range contents {
		images += len(content.Media)
	}
	if images > limits.MaxImages {
		return nil, refuse("too_many_images", "the request carries %d images; the limit is %d per request", images, limits.MaxImages)
	}

	// The criterion is the question. An image there would change what is being asked, so it is
	// refused rather than silently folded into (or dropped from) the criterion text.
	for index := range raw {
		if isCriterion(shapes[index]) && len(contents[index].Media) > 0 {
			return nil, refuse("media_in_criterion", "the criterion (a `system`/`developer` message) must be text-only: it is the question, so media there would change what is being asked")
		}
	}

	criterionParts := make([]string, 0, len(raw))
	for index := range raw {
		if isCriterion(shapes[index]) && strings.TrimSpace(contents[index].Text) != "" {
			criterionParts = append(criterionParts, contents[index].Text)
		}
	}
	criterion := strings.Join(criterionParts, "\n\n")
	if criterion == "" {
		return nil, refuse("messages_not_semif_contract", "the criterion must be a nonempty `system` (or `developer`) message")
	}

	type roleContent struct {
		role    string
		content *media.Content
	}
	rest := make([]roleContent, 0, len(raw))
	for index := range raw {
		if !isCriterion(shapes[index]) {
			rest = append(rest, roleContent{role: roleOf(shapes[index]), content: contents[index]})
		}
	}
	if len(rest) == 0 {
		return nil, refuse("messages_not_semif_contract", "there is no evidence: at least one non-system message is required")
	}
	for _, item := range rest {
		// A message with an image needs no text; a message with neither is empty evidence.
		if strings.TrimSpace(item.content.Text) == "" && len(item.content.Media) == 0 {
			return nil, refuse("messages_not_semif_contract", "a %q message carries no text or image; every evidence message needs content", item.role)
		}
	}

	canonical := len(raw) == 2 && roleOf(shapes[0]) == "system" && roleOf(shapes[1]) == "user"
	anyMedia := false
	for _, content := range contents {
		if len(content.Media) > 0 {
			anyMedia = true
			break
		}
	}

	if !anyMedia {
		// The text contract, byte for byte: one text part, folded exactly as before.
		evidence := rest[0].content.Text
		renderer := prompt.EvidenceCanonical
		if !canonical {
			lines := make([]string, 0, len(rest))
			for _, item := range rest {
				lines = append(lines, item.role+": "+item.content.Text)
			}
			evidence = strings.Join(lines, "\n\n")
			renderer = prompt.EvidenceTranscript
		}
		return &messages{criterion: criterion, evidence: []evidencePart{{text: evidence}}, renderer: renderer}, nil
	}

	// Media present: keep the caller's order. A message without media keeps its `role: text`
	// rendering; one with media emits `role: ` then its parts, so an image lands where the caller
	// put it rather than after every message.
	var parts []evidencePart
	renderer := media.EvidenceTranscriptMedia
	if canonical {
		parts = contentParts(rest[0].content)
		renderer = media.EvidenceCanonicalMedia
	} else {
		for offset := range rest {
			if offset > 0 {
				parts = append(parts, evidencePart{text: "\n\n"})
			}
			item := rest[offset]
			if len(item.content.Media) == 0 {
				parts = append(parts, evidencePart{text: item.role + ": " + item.content.Text})
				continue
			}
			parts = append(parts, evidencePart{text: item.role + ": "})
			parts = append(parts, contentParts(item.content)...)
		}
	}
	return &messages{criterion: criterion, evidence: parts, renderer: renderer}, nil
}

// contentParts is one message's content as ordered evidence parts: its text leading, then its
// media in the order the caller listed them.
func contentParts(content *media.Content) []evidencePart {
	parts := make([]evidencePart, 0, len(content.Media)+1)
	if content.Text != "" {
		parts = append(parts, evidencePart{text: content.Text})
	}
	for index := range content.Media {
		image := content.Media[index]
		parts = append(parts, evidencePart{image: &image})
	}
	return parts
}

func roleOf(message map[string]any) string {
	role, _ := message["role"].(string)
	return role
}
