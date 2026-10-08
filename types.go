package weigh

import (
	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/media"
	"github.com/wweir/weigh/internal/prompt"
	"github.com/wweir/weigh/internal/readout"
	"github.com/wweir/weigh/internal/slots"
	"github.com/wweir/weigh/internal/template"
)

// The types of the readout, re-exported so that a caller of this library names one package. They
// are aliases, not copies: a value returned here is the same type the serving path sees, so the
// two cannot drift.
type (
	// Kind is the serving dialect a probed endpoint turned out to be.
	Kind = backend.Kind
	// BackendChoice is the requested dialect: auto, vllm, or sglang.
	BackendChoice = backend.Choice
	// Readout is the exact-slot versus top-N trade-off: exact-slot or top-n.
	Readout = backend.Readout
	// MediaSupport is what the probe proved about scoring an image. It is a separate axis from
	// the text readout tier.
	MediaSupport = backend.MediaSupport
	// MediaPolicy is how far the caller lets this client go with images.
	MediaPolicy = readout.MediaPolicy
	// ChatTemplate identifies one rendered chat-template family.
	ChatTemplate = template.ChatTemplate
	// TemplateChoice is the requested wrapper: auto, or one named family.
	TemplateChoice = template.Choice
	// Resolved is a template plus how it was decided.
	Resolved = template.Resolved
	// Media is one image reference, as the caller declared it.
	Media = media.Media
	// Config is everything [New] needs to start.
	Config = readout.Config
	// Metadata is the serving configuration every Scored value is read against.
	Metadata = readout.Metadata
	// EndpointInfo is what the startup probe learned about one endpoint.
	EndpointInfo = readout.EndpointInfo
	// Scored is one decision, with the provenance needed to attribute it.
	Scored = readout.Scored
	// ContractError reports a request this client will not ask the backend to score: the prompt
	// does not tokenize, the answer boundary is not clean, the option count is outside the slot
	// alphabet, or the prompt exceeds the context ceiling. It is the caller's fault rather than
	// the backend's, so it is worth separating with errors.As.
	ContractError = readout.ContractError
	// Row is one validated decision row: an id, the evidence (State), the criterion (Question),
	// and the options.
	Row = prompt.Row
	// Option is one choice in a row.
	Option = prompt.Option
)

// Dialects. Config.Backend takes BackendAuto, BackendVLLM, or BackendSGLang.
const (
	BackendAuto   = backend.ChoiceAuto
	BackendVLLM   = backend.ChoiceVLLM
	BackendSGLang = backend.ChoiceSGLang
)

// The probed dialect, read off [Metadata.Backend].
const (
	KindVLLM   = backend.KindVLLM
	KindSGLang = backend.KindSGLang
)

// Readout tiers. Config.Readout takes ReadoutExactSlot or ReadoutTopN; ReadoutAuto has no meaning
// for a client, because it picks a different tier per backend.
const (
	ReadoutAuto      = backend.ReadoutAuto
	ReadoutExactSlot = backend.ReadoutExactSlot
	ReadoutTopN      = backend.ReadoutTopN
)

// Media policies. Config.Media takes MediaOff (the zero value), MediaExactSlot, or MediaTopN.
const (
	MediaOff       = readout.MediaOff
	MediaExactSlot = readout.MediaExactSlot
	MediaTopN      = readout.MediaTopN
)

// What the probe proved about media, read off [Metadata.MediaSupport].
const (
	MediaNone          = backend.MediaNone
	MediaChatTopN      = backend.MediaChatTopN
	MediaChatExactSlot = backend.MediaChatExactSlot
)

// Chat-template families. Config.Template takes TemplateAuto or {Fixed: one of these}.
//
// A family is a fixed prefix plus an assistant-turn opener, not a Jinja engine: the served shape
// is always one plain system turn plus one plain user turn, so each family's reachable subgraph is
// written out and named. A checkpoint whose template matches no known family is refused rather
// than defaulted, because rendering the wrong wrapper produces a prompt the model was never
// trained on and the answer-boundary proof only validates the last token.
const (
	Gemma4         = template.Gemma4
	Gemma3         = template.Gemma3
	Chatml         = template.Chatml
	ChatmlThinking = template.ChatmlThinking
	Llama3         = template.Llama3
	Llama2         = template.Llama2
	Mistral        = template.Mistral

	// TemplateAuto resolves the family from Config.Source's tokenizer_config.json, refusing an
	// unrecognised shape.
	TemplateAuto = "auto"
)

// The answer-slot alphabet: N options are bound to the first N distinct single-token letters,
// A..P. Sixteen is a deliberate ceiling rather than a protocol limit — the contract is only
// constructive while every letter is a single round-trip token, and that is checked by the
// backend's own tokenizer before anything is scored.
const (
	MinOptions    = slots.MinValues
	MaxOptions    = slots.MaxValues
	OptionLetters = slots.Letters
)

// The readout recipes, as reported in [Metadata.ServingConfig] and
// [Metadata.MediaServingConfig].
//
// They name the request the number was read from, so that two readouts are never pooled by
// accident: a subset-softmax number carries a different recipe from a pinned-token-id one, and
// the media routes differ from both.
const (
	ServingVLLMTokenIDs      = readout.ServingVLLMTokenIDs
	ServingVLLMSubsetSoftmax = readout.ServingVLLMSubsetSoftmax
	ServingSGLangGenerate    = readout.ServingSGLangGenerate
	ServingVLLMMediaTokenIDs = readout.ServingVLLMMediaTokenIDs
	ServingVLLMMediaTopN     = readout.ServingVLLMMediaTopN
)
