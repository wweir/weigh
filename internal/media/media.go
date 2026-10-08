// Package media parses multimodal request parts: OpenAI `image_url` content, `data:` URIs, and
// the exact bytes this service hands to the backend.
//
// # Why a media part is not "one more content type"
//
// The text contract is narrow: the client renders the chat template, tokenizes it, and proves
// the answer boundary. An image breaks the last of those, because the backend's processor
// expands it into placeholder tokens this service never sees. A request carrying media is
// therefore a *degraded* readout, and every response records that (`modality`,
// `server_tokenized`, and one provenance object per image).
//
// # Pass-through is the rule for what reaches the backend
//
// Media.Reference is the caller's `image_url.url` byte for byte, and that exact string is what
// the readout forwards. Nothing is re-encoded or rewritten to a canonical form: the point of
// recording a sha256 is that the recorded value describes what was actually sent. Decoding is a
// side channel, used only for a size ceiling and a byte fingerprint, and it never feeds the
// forwarded reference.
//
// The MIME is recorded as the caller declared it, not judged here: whether a type is a usable
// image is the backend processor's decision, and a second allow-list would reject images a newer
// checkpoint handles while adding nothing the backend does not already enforce.
//
// # Refusal, never silent degradation
//
// Every way an image can be unacceptable is reported by name — an unparseable `data:` URI, an
// oversized payload, a remote URL, media in the criterion — because the alternative (dropping
// the image and scoring the text) answers a question the caller did not ask.
package media

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/wweir/weigh/internal/hashx"
	"github.com/wweir/weigh/internal/jsonx"
)

// The `evidence_renderer` values for a request that carries media. They are distinct from the
// text renderers on purpose: the media fold changes the prompt, so a recorded decision must say
// which fold produced it.
const (
	EvidenceCanonicalMedia  = "system+user+media"
	EvidenceTranscriptMedia = "transcript+media"
)

// ProbeImageDataURI is a real 1x1 PNG as a `data:` URI, used by the startup media probe.
//
// It is a real image on purpose: a probe that sent something the processor rejects would report
// "no media support" for a server that merely disliked the probe, while a build with no vision
// path answers 400/422 — which is the negative signal being read.
const ProbeImageDataURI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

// Limits is what the operator allowed a request to carry.
type Limits struct {
	// AllowMedia is the master switch; when it is false an `image_url` part is refused exactly
	// as it was before this package existed.
	AllowMedia bool
	// AllowRemote accepts `http(s)://` references. Off by default: the backend would fetch an
	// arbitrary URL on the caller's behalf, and this service can never hash the bytes it gets.
	AllowRemote bool
	MaxImages   int
	// MaxMediaBytes is the per-image decoded ceiling. The request body limit is a separate,
	// coarser guard.
	MaxMediaBytes int
}

// DefaultLimits is the disabled default, used by callers that only exercise the text path.
func DefaultLimits() Limits {
	return Limits{MaxImages: 1, MaxMediaBytes: 8 * 1024 * 1024}
}

// EncodedCeiling is the largest base64 payload that can decode to `decoded` bytes, rounded up to
// a 4-byte group. It bounds a `data:` URI before its decoding is allocated, and it is the same
// bound the request-body limit is derived from, so the two cannot drift.
func EncodedCeiling(decoded int) int {
	return decoded/3*4 + 4
}

// Error is a named refusal. The code is part of the HTTP contract; the message names the
// offending message index.
type Error struct {
	Code    string
	Message string
}

func (e *Error) Error() string { return e.Message }

func newError(code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Media is one image the caller attached to the evidence.
type Media struct {
	// Mime is what the caller declared. Never judged here.
	Mime string
	// SHA256 is the digest of the decoded bytes, nil for a remote URL this service never
	// fetched: the fingerprint would be of the URL, not of what the backend actually loads.
	SHA256 *string
	// Size is the decoded size in bytes, nil for a remote URL.
	Size *int
	// Reference is the exact `image_url.url` string handed to the backend, byte for byte. It is
	// the only field the readout reads when it builds the backend request.
	Reference string
}

// ContentPart is the `image_url` part for the backend request, forwarding Reference unchanged.
func (m Media) ContentPart() jsonx.Object {
	return jsonx.Object{
		{Key: "type", Value: "image_url"},
		{Key: "image_url", Value: jsonx.Object{{Key: "url", Value: m.Reference}}},
	}
}

// Provenance is the object recorded in a response.
//
// `url` is restated from the reference exactly when there is no local fingerprint: a remote URL
// is the only case that hashes nothing, so the two cannot disagree.
func (m Media) Provenance() map[string]any {
	provenance := map[string]any{
		"kind":   "image",
		"mime":   m.Mime,
		"bytes":  nil,
		"sha256": nil,
		"url":    nil,
	}
	if m.Size != nil {
		provenance["bytes"] = *m.Size
	}
	if m.SHA256 != nil {
		provenance["sha256"] = *m.SHA256
	} else {
		provenance["url"] = m.Reference
	}
	return provenance
}

// Content is one message's content, folded from a string or a content-parts array.
//
// Text and media are kept separate and in order: a JSON `evidence` string cannot represent "an
// image between these two paragraphs", so the fold must not try.
type Content struct {
	Text  string
	Media []Media
}

// ReadContent folds one message's `content` into text plus media.
//
// A part that is neither `text` nor an accepted `image_url` is refused by name rather than
// dropped, and the index of the offending message is always in the message.
func ReadContent(message map[string]any, index int, limits Limits) (*Content, *Error) {
	switch content := message["content"].(type) {
	case string:
		return &Content{Text: content}, nil
	case []any:
		return readParts(content, index, limits)
	default:
		return nil, newError("messages_not_semif_contract",
			"messages[%d] needs `content`: a string, or an array of text/image_url parts", index)
	}
}

func readParts(parts []any, index int, limits Limits) (*Content, *Error) {
	var text strings.Builder
	var images []Media
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			return nil, newError("unsupported_content_part",
				"messages[%d] has a content part with no \"type\"", index)
		}
		kind, ok := part["type"].(string)
		if !ok {
			return nil, newError("unsupported_content_part",
				"messages[%d] has a content part with no \"type\"", index)
		}
		switch kind {
		case "text":
			// A `text` part whose `text` is missing or not a string is refused, not skipped:
			// skipping answers a question the caller did not ask, with nothing in the response
			// saying a part was lost.
			fragment, ok := part["text"].(string)
			if !ok {
				return nil, newError("unsupported_content_part",
					"messages[%d] has a `text` content part whose `text` is missing or not a string", index)
			}
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(fragment)
		case "image_url":
			if !limits.AllowMedia {
				return nil, newError("media_not_enabled",
					"messages[%d] carries an image_url part, but this server was started without --allow-media; media scoring is a degraded readout and must be opted into.", index)
			}
			if len(images) >= limits.MaxImages {
				return nil, newError("too_many_images",
					"messages[%d] carries more than %d images", index, limits.MaxImages)
			}
			image, err := readImage(part, index, limits)
			if err != nil {
				return nil, err
			}
			images = append(images, *image)
		default:
			return nil, newError("unsupported_content_part",
				"messages[%d] carries a %q content part; this endpoint scores text and image decisions", index, kind)
		}
	}
	return &Content{Text: text.String(), Media: images}, nil
}

// readImage parses one `{"type":"image_url","image_url":{"url": …}}` part.
func readImage(part map[string]any, index int, limits Limits) (*Media, *Error) {
	reference, ok := part["image_url"].(map[string]any)
	if !ok {
		return nil, newError("invalid_media", "messages[%d] image_url part needs `image_url.url`", index)
	}
	// Only `url` is forwarded, and the other keys OpenAI defines there (`detail` above all)
	// select *how* the backend renders the image — i.e. how many image tokens the model sees.
	// Dropping one would answer at the default resolution while the response looked exactly like
	// a request that asked for it.
	for key := range reference {
		if key != "url" {
			return nil, newError("unsupported_content_part",
				"messages[%d] image_url uses `%s`; this endpoint forwards only `image_url.url`, so any other key would be silently ignored", index, key)
		}
	}
	url, ok := reference["url"].(string)
	if !ok {
		return nil, newError("invalid_media", "messages[%d] image_url part needs `image_url.url`", index)
	}

	switch {
	case strings.HasPrefix(url, "data:"):
		return decodeDataURI(strings.TrimPrefix(url, "data:"), url, index, limits)
	case strings.HasPrefix(url, "http://"), strings.HasPrefix(url, "https://"):
		if !limits.AllowRemote {
			return nil, newError("media_remote_not_allowed",
				"messages[%d] references a remote image URL; this server will not make the backend fetch an arbitrary URL, and it cannot fingerprint the bytes that come back. Enable --allow-remote-media to accept remote URLs, or send the image as a `data:` URI so its hash can be recorded.", index)
		}
		return &Media{Mime: "unknown", Reference: url}, nil
	default:
		return nil, newError("invalid_media",
			"messages[%d] image URL must be a `data:` URI or an http(s) URL; a filesystem path would be a file-read primitive on the backend host.", index)
	}
}

// decodeDataURI decodes `data:<mime>;base64,<payload>` and fingerprints the bytes.
func decodeDataURI(rest, url string, index int, limits Limits) (*Media, *Error) {
	header, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return nil, newError("invalid_media", "messages[%d] data URI has no comma before the payload", index)
	}
	fields := strings.Split(header, ";")
	mime := strings.ToLower(strings.TrimSpace(fields[0]))
	if mime == "" {
		mime = "unknown"
	}
	base64Encoded := false
	for _, field := range fields[1:] {
		if strings.EqualFold(strings.TrimSpace(field), "base64") {
			base64Encoded = true
			break
		}
	}
	// A non-base64 data URI is refused rather than forwarded: without base64 the decoded size
	// cannot be bounded, so the ceiling would be unenforceable. That is a deliberate narrowing
	// of pass-through, not a transformation of the reference.
	if !base64Encoded {
		return nil, newError("invalid_media",
			"messages[%d] data URI must be base64-encoded (a percent-encoded payload has no enforceable size ceiling)", index)
	}
	// Reject on the encoded length before allocating, so a huge payload cannot make this process
	// allocate three quarters of it just to then fail the decoded-size check.
	ceiling := EncodedCeiling(limits.MaxMediaBytes)
	if len(payload) > ceiling {
		return nil, newError("media_too_large",
			"messages[%d] image payload is %d base64 bytes; the limit is %d decoded bytes", index, len(payload), limits.MaxMediaBytes)
	}
	bytes, err := base64.StdEncoding.DecodeString(strings.TrimSpace(payload))
	if err != nil {
		return nil, newError("invalid_media", "messages[%d] image base64 is invalid: %v", index, err)
	}
	if len(bytes) > limits.MaxMediaBytes {
		return nil, newError("media_too_large",
			"messages[%d] image decodes to %d bytes; the limit is %d", index, len(bytes), limits.MaxMediaBytes)
	}
	if len(bytes) == 0 {
		return nil, newError("invalid_media", "messages[%d] image decodes to zero bytes", index)
	}
	// Reference is the caller's string verbatim; the decoded bytes are dropped after being
	// fingerprinted, because pass-through is what reaches the backend.
	digest := hashx.Sha256Hex(bytes)
	size := len(bytes)
	return &Media{Mime: mime, SHA256: &digest, Size: &size, Reference: url}, nil
}
