package serve

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/wweir/weigh/internal/prompt"
	"github.com/wweir/weigh/internal/readout"
	"github.com/wweir/weigh/internal/schema"
	"github.com/wweir/weigh/internal/slots"
	"github.com/wweir/weigh/internal/tokenize"
)

// ignoredKeys are sampling parameters that are accepted, ignored, and listed back in
// semif.ignored_parameters. The order is this one, not the request's.
var ignoredKeys = []string{
	"temperature", "top_p", "top_k", "seed", "stop",
	"presence_penalty", "frequency_penalty", "logit_bias",
	"max_tokens", "max_completion_tokens",
}

// completionValue is the decision as a canonical (non-streaming) chat.completion object.
//
// Streaming reuses it and re-slices it into chunks, so exactly one place decides what a response
// says.
func (s *Server) completionValue(
	object map[string]any,
	decision *schema.Decision,
	scored *readout.Scored,
	renderer string,
	wantedLogprobs bool,
	topN int,
) (map[string]any, *reply) {
	if len(scored.Probabilities) != len(decision.Values) || len(scored.OptionLogprobs) != len(decision.Values) {
		rejection := errorResponse(500, internalError, "the readout did not return one value per declared slot")
		return nil, &rejection
	}
	choiceIndex := 0
	best := math.Inf(-1)
	for index, value := range scored.Probabilities {
		if value > best {
			best, choiceIndex = value, index
		}
	}
	choice := decision.Values[choiceIndex]

	ignored := make([]string, 0, len(ignoredKeys))
	for _, key := range ignoredKeys {
		if _, present := object[key]; present {
			ignored = append(ignored, key)
		}
	}

	var logprobs any
	if wantedLogprobs {
		logprobs = choiceLogprobs(scored.OptionLogprobs, choiceIndex, topN)
	}

	// The readout path, not a boolean flag, selects the recipe: `server_tokenized` is true on
	// every path now, so only the modality can tell a text readout from a media one.
	servingConfig := s.metadata.ServingConfig
	if scored.Modality == "text+image" {
		if s.metadata.MediaServingConfig == "" {
			rejection := errorResponse(500, internalError,
				"a media readout was produced by an endpoint with no media serving recipe")
			return nil, &rejection
		}
		servingConfig = s.metadata.MediaServingConfig
	}

	// The text path hashes the prompt; the media path hashes the request body it sent, because it
	// never saw the prompt. Either way the completion id is input-bound.
	idMaterial := ""
	switch {
	case scored.PromptSHA256 != nil:
		idMaterial = *scored.PromptSHA256
	case scored.RequestSHA256 != nil:
		idMaterial = *scored.RequestSHA256
	}
	if len(idMaterial) > 24 {
		idMaterial = idMaterial[:24]
	}

	images := scored.Media
	if images == nil {
		images = []any{}
	}

	return map[string]any{
		"id":      "chatcmpl-" + idMaterial,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   s.servedModel,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": decision.Content(choice)},
			"finish_reason": "stop",
			"logprobs":      logprobs,
			"semif": map[string]any{
				"field":              decision.Field,
				"values":             decision.Values,
				"labels":             decision.Labels(),
				"choice":             choice,
				"choice_index":       choiceIndex,
				"probabilities":      scored.Probabilities,
				"option_logprobs":    scored.OptionLogprobs,
				"answer_token_ids":   scored.AnswerTokenIDs,
				"option_ids":         scored.OptionIDs,
				"readout":            scored.Readout,
				"fallback_used":      scored.FallbackUsed,
				"endpoint":           scored.Endpoint,
				"input_tokens":       scored.InputTokens,
				"input_ids_sha256":   scored.InputIDsSHA256,
				"prompt_sha256":      scored.PromptSHA256,
				"request_sha256":     scored.RequestSHA256,
				"modality":           scored.Modality,
				"server_tokenized":   scored.ServerTokenized,
				"media":              images,
				"probability_status": "conditional option score; uncalibrated as decision confidence",
				"engine_seconds":     scored.EngineSeconds,
				"fallback_seconds":   scored.FallbackSeconds,
				"total_seconds":      scored.TotalSeconds,
			},
		}},
		"usage": usageBlock(scored.InputTokens),
		"semif": map[string]any{
			"backend":                string(s.metadata.Backend),
			"serving_config":         servingConfig,
			"media_support":          string(s.metadata.MediaSupport),
			"prompt_version":         prompt.PromptVersion,
			"prompt_template":        string(s.metadata.PromptTemplate),
			"prompt_template_source": s.metadata.PromptTemplateSource,
			"revision":               s.revision,
			"prompt_contract":        "system = criterion, user = evidence; schema enum values are the option descriptions",
			"evidence_renderer":      renderer,
			"ignored_parameters":     ignored,
			"client":                 clientFlavour,
			"tokenize_endpoint":      "POST " + tokenize.Path(s.metadata.Backend),
		},
	}, nil
}

func usageBlock(inputTokens *int) map[string]any {
	usage := map[string]any{
		"prompt_tokens":     inputTokens,
		"completion_tokens": 1,
		"total_tokens":      nil,
	}
	if inputTokens != nil {
		usage["total_tokens"] = *inputTokens + 1
	}
	return usage
}

// choiceLogprobs is the standard choices[].logprobs.content shape, filled from the same per-slot
// logprobs the vendor block carries. The token text is the slot's letter, which is literally what
// the model scored.
func choiceLogprobs(optionLogprobs []float64, choiceIndex, topN int) map[string]any {
	ranked := make([]int, len(optionLogprobs))
	for index := range ranked {
		ranked[index] = index
	}
	sort.SliceStable(ranked, func(a, b int) bool {
		return optionLogprobs[ranked[a]] > optionLogprobs[ranked[b]]
	})

	entry := func(index int) map[string]any {
		letter := slots.Letter(index)
		return map[string]any{
			"token":   letter,
			"logprob": optionLogprobs[index],
			// A JSON array of numbers, not a base64 string: Go marshals []byte as base64.
			"bytes": []int{int(letter[0])},
		}
	}
	top := make([]any, 0, min(topN, len(ranked)))
	for _, index := range ranked[:min(topN, len(ranked))] {
		top = append(top, entry(index))
	}

	choice := entry(choiceIndex)
	choice["top_logprobs"] = top
	return map[string]any{"content": []any{choice}}
}

// sseBody is the streaming form: exactly one content chunk, then a stop chunk, then [DONE].
//
// Honest rather than approximate: this endpoint produces one position, so the stream has one
// delta. `stream_options.include_usage` adds the usage-only chunk OpenAI clients expect.
func sseBody(value map[string]any, includeUsage bool) string {
	choices, _ := value["choices"].([]any)
	choice, _ := choices[0].(map[string]any)

	envelope := func(chunks any) map[string]any {
		return map[string]any{
			"id":      value["id"],
			"object":  "chat.completion.chunk",
			"created": value["created"],
			"model":   value["model"],
			"choices": chunks,
		}
	}
	content := envelope([]any{map[string]any{
		"index":         0,
		"delta":         map[string]any{"role": "assistant", "content": choice["message"].(map[string]any)["content"]},
		"finish_reason": nil,
		"logprobs":      choice["logprobs"],
		"semif":         choice["semif"],
	}})
	stop := envelope([]any{map[string]any{
		"index":         0,
		"delta":         map[string]any{},
		"finish_reason": "stop",
	}})

	var out strings.Builder
	for _, chunk := range []map[string]any{content, stop} {
		out.WriteString("data: " + marshal(chunk) + "\n\n")
	}
	if includeUsage {
		usage := envelope([]any{})
		usage["usage"] = value["usage"]
		out.WriteString("data: " + marshal(usage) + "\n\n")
	}
	out.WriteString("data: [DONE]\n\n")
	return out.String()
}
