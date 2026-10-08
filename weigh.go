// Package weigh reads a decision out of a generative model by asking for the exact probability
// of *named answers*, and returns the subset softmax over them plus the provenance to attribute
// the number.
//
// It is the same readout the weighd service serves, as a library. The caller renders the prompt:
// [Client.Decide] takes a finished prompt string and the labels it bound its answer slots to, and
// this package renders no chat template of its own. The assembly primitives in this package —
// [ValidateRow], [UserMessage], [RenderPrompt], [RenderDirectOptions] — exist so a caller that
// wants the service's `direct-options-v1` contract can reproduce it byte for byte, and so a caller
// that wants something else can configure the pieces instead of reimplementing them.
//
// Provenance travels with the number: [Scored] carries the prompt digest, the readout recipe that
// ran, the endpoint that ran it, and the timings. It does not carry a decision confidence, and
// this package never calls it one.
package weigh

import (
	"context"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/readout"
)

// Client scores decisions against a probed fleet.
//
// Use [New]; the zero value is not usable. A Client is safe for concurrent use: endpoint
// selection is round-robin, and one request's ids and score always come from the same engine.
type Client struct {
	inner    *readout.Client
	metadata *Metadata
}

// New probes every endpoint and resolves the serving configuration.
//
// Everything that can refuse happens here, before the first request: an endpoint that cannot be
// identified, a fleet whose members disagree, a tokenizer mismatch, a backend that does not prove
// the requested readout, and an unknown prompt-length ceiling. It refuses rather than defaulting,
// because a readout that quietly ran under a different recipe produces plausible numbers that mean
// something else.
//
// Config.Source is optional. Set it to a local checkpoint directory to have Config.Template
// resolved from that checkpoint's tokenizer_config.json and to have the served checkpoint guarded
// against it. With no Source there is nothing to detect from, so an explicitly named family is
// recorded as given, TemplateAuto is refused, and an unspecified Template leaves
// [Metadata.PromptTemplate] empty. See [Metadata.PromptTemplate].
func New(ctx context.Context, cfg Config) (*Client, *Metadata, error) {
	inner, metadata, err := readout.New(ctx, cfg)
	if err != nil {
		return nil, nil, err
	}
	return &Client{inner: inner, metadata: metadata}, metadata, nil
}

// Metadata returns the serving configuration this client was opened with.
func (c *Client) Metadata() *Metadata { return c.metadata }

// Decide reads one decision from a finished prompt.
//
// prompt is rendered by the caller and is never inspected, rewritten, or re-tokenized locally:
// it goes to the backend's own tokenizer as bytes, and [Scored.PromptSHA256] is the digest of
// exactly those bytes. optionIDs[i] is the caller's label for answer slot i, where slot i is the
// letter slots.Letter(i) — `A` for index 0, up to `P` for index 15. The caller is responsible for
// having bound its prompt to those letters; [RenderDirectOptions] does that binding when the
// caller uses this package's default contract. [Scored.Probabilities] is the subset softmax over
// the declared slots, in the same order.
//
// A prompt whose answer boundary cannot be proven (the letter does not append exactly one token,
// or two letters collide) is refused as a [ContractError] rather than scored at the wrong
// position.
func (c *Client) Decide(ctx context.Context, rowID, prompt string, optionIDs []string) (*Scored, error) {
	return c.inner.Decide(ctx, rowID, prompt, optionIDs)
}

// DecideMedia reads one decision whose evidence carries images.
//
// textPayload is the text half of the prompt, built by the caller — [MediaUserPayload] produces
// the same payload [UserMessage] produces, with the image references left out. The images are sent
// as sibling content parts, and the backend's own chat template assembles the final prompt, using
// system as the system turn. That is also why this path carries a weaker proof than [Decide]:
// the image expands into placeholder tokens this process never sees, so [Scored.RequestSHA256]
// replaces [Scored.PromptSHA256], and the two modalities must not be pooled.
func (c *Client) DecideMedia(ctx context.Context, rowID, system, textPayload string, images []Media, optionIDs []string) (*Scored, error) {
	return c.inner.DecideMedia(ctx, rowID, system, textPayload, images, optionIDs)
}

// IsTimeout reports whether an error spent its backend budget. It is the one backend failure a
// caller may reasonably retry; a missing route or a refused contract will fail again.
func IsTimeout(err error) bool { return backend.IsTimeout(err) }
