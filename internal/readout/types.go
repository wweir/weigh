package readout

import (
	"fmt"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/template"
)

// ServingVLLMTokenIDs names the exact-slot vLLM recipe.
const (
	ServingVLLMTokenIDs      = "vllm-openai-logprob-token-ids-v1"
	ServingVLLMSubsetSoftmax = "vllm-openai-completions-subset-softmax-v1"
	ServingSGLangGenerate    = "sglang-native-generate-token-ids-v1"
)

// ServingConfig names the recipe for a text readout.
func ServingConfig(kind backend.Kind, tierA bool) string {
	if kind == backend.KindSGLang {
		return ServingSGLangGenerate
	}
	if tierA {
		return ServingVLLMTokenIDs
	}
	return ServingVLLMSubsetSoftmax
}

// The media recipes. A media readout can only come from an endpoint that has one of these, so a
// missing recipe is a divergence to report rather than a fallback to take.
const (
	ServingVLLMMediaTokenIDs = "vllm-openai-chat-media-token-ids-v1"
	ServingVLLMMediaTopN     = "vllm-openai-chat-media-topn-v1"
)

// MediaServingConfig names the recipe for a media readout.
func MediaServingConfig(kind backend.Kind, support backend.MediaSupport) (string, bool) {
	if kind != backend.KindVLLM {
		return "", false
	}
	switch support {
	case backend.MediaChatExactSlot:
		return ServingVLLMMediaTokenIDs, true
	case backend.MediaChatTopN:
		return ServingVLLMMediaTopN, true
	default:
		return "", false
	}
}

// ContractError reports a request this service cannot ask the backend to score: the prompt does
// not tokenize, the answer boundary is not clean, the option count is outside the slot alphabet,
// or the prompt exceeds the context ceiling. It is the caller's fault, not the backend's, so the
// HTTP layer reports it as a 400 rather than as a backend failure.
type ContractError struct{ Message string }

func (e *ContractError) Error() string { return e.Message }

func contract(format string, args ...any) *ContractError {
	return &ContractError{Message: fmt.Sprintf(format, args...)}
}

// EndpointInfo is what the startup probe learned about one backend URL.
type EndpointInfo struct {
	URL          string
	ServedModels []string
	ModelRoot    *string
	Version      string
}

// Metadata is the serving configuration every response's provenance is read against.
type Metadata struct {
	Backend      backend.Kind
	TierA        bool
	MediaSupport backend.MediaSupport
	// ServingConfig names the text readout recipe. It is a property of the probed deployment, so
	// it is resolved once here rather than recomputed per response.
	ServingConfig string
	// MediaServingConfig is the media recipe, empty when this deployment does not serve media.
	MediaServingConfig   string
	Endpoints            []EndpointInfo
	MaxModelLen          int
	TokenizerMatches     *bool
	PromptTemplate       template.ChatTemplate
	PromptTemplateSource string
}

// Scored is one decision, with the provenance needed to attribute it.
type Scored struct {
	// Probabilities is the subset softmax over the declared slots, in slot order.
	Probabilities []float64
	// OptionLogprobs is what the backend returned for those slots, in slot order.
	OptionLogprobs []float64
	// AnswerTokenIDs[i] is the token id of slot i.
	AnswerTokenIDs []uint32
	// OptionIDs[i] is the caller's label for slot i.
	OptionIDs []string

	InputTokens    *int
	InputIDsSHA256 *string
	PromptSHA256   *string
	RequestSHA256  *string

	// Modality is "text" or "text+image"; the two can never be pooled silently.
	Modality string
	// ServerTokenized is true: the token ids came from the backend on every path, which is what
	// makes a client/server tokenizer disagreement impossible. `modality` distinguishes the
	// media path.
	ServerTokenized bool
	// Media is one provenance object per attached image.
	Media []any

	Readout         string
	FallbackUsed    bool
	FallbackSeconds float64
	EngineSeconds   float64
	TotalSeconds    float64
	// Endpoint is the index of the backend that served this row.
	Endpoint int
}
