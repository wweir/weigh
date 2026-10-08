package readout

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/hashx"
	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/media"
)

// DecideMedia scores a decision whose evidence carries images.
//
// The media prompt is tokenized by the backend's own processor, which expands an image into
// placeholder tokens this service never sees. There is therefore no answer-boundary proof: the
// slot contract (one token per letter, distinct tokens) is all that can be checked, and every
// response records the weaker state — `modality`, `server_tokenized`, and the digest of the
// request body instead of the digest of the prompt.
func (c *Client) DecideMedia(
	ctx context.Context,
	rowID, system, textPayload string,
	images []media.Media,
	optionIDs []string,
) (*Scored, error) {
	if c.mediaSupport == backend.MediaNone {
		return nil, contract("this endpoint was not probed as able to score an image; start the server with --allow-media against a multimodal model")
	}
	if len(images) == 0 {
		return nil, contract("a media readout needs at least one media part")
	}

	// The clock starts before the slot lookup: that is network work now, and total_seconds is
	// what the readout histogram measures.
	started := time.Now()
	index := int((c.next.Add(1) - 1) % uint64(len(c.endpoints)))
	endpoint := &c.endpoints[index]

	slots, err := endpoint.tokenizer.SlotIDs(ctx, len(optionIDs))
	if err != nil {
		return nil, contractErrorUnlessBackend(err)
	}

	exact := c.mediaSupport == backend.MediaChatExactSlot
	request := jsonx.Object{
		{Key: "model", Value: first(endpoint.info.ServedModels)},
		{Key: "messages", Value: chatMessages(system, textPayload, images)},
		{Key: "max_tokens", Value: 1},
		{Key: "temperature", Value: 0},
		{Key: "logprobs", Value: true},
		{Key: "top_logprobs", Value: maxLogprobs},
		// Both media routes ask for token-id keys. Without a local tokenizer this service cannot
		// match decoded text, so a build that answers under decoded text is refused rather than
		// guessed at — and a slot outside the returned top-N has no recovery here.
		{Key: "return_tokens_as_token_ids", Value: true},
	}
	if exact {
		request = append(request, jsonx.Field{
			Key:   "logprob_token_ids",
			Value: jsonx.Raw(jsonx.IntList(slots)),
		})
	}
	body, err := jsonx.Dump(request)
	if err != nil {
		return nil, err
	}

	send := time.Now()
	var response map[string]any
	if err := endpoint.transport.PostJSON(ctx, "/v1/chat/completions", []byte(body), &response); err != nil {
		return nil, fmt.Errorf("Row %s: %w", rowID, err)
	}
	engineSeconds := time.Since(send).Seconds()

	entry, ok := chatLogprobEntry(response)
	if !ok {
		return nil, fmt.Errorf(
			"Row %s: the chat route returned no `choices[].logprobs.content` entry for the image request; `logprob_token_ids` on /v1/chat/completions is not supported by this build",
			rowID)
	}
	if !chatTokenIDs(entry) {
		return nil, fmt.Errorf(
			"Row %s: this build answers the media route with decoded token text and this service has no local tokenizer to match it; refusing rather than reading a distribution it cannot address by id",
			rowID)
	}

	optionLogprobs, missingErr := parseChatSlots(entry, slots, rowID)
	if missingErr != nil {
		var missing *missingSlot
		if !errors.As(missingErr, &missing) {
			return nil, missingErr
		}
		if exact {
			return nil, fmt.Errorf(
				"Row %s: the chat route omitted requested slot %d; this endpoint was probed as returning exact answer slots keyed by `token_id:`, but this response carried none of them",
				rowID, missing.slot)
		}
		return nil, fmt.Errorf(
			"Row %s: answer slot %d is outside the returned top-N, and a media prompt has no /generative_scoring fallback (that route takes token ids and cannot carry an image). Refusing rather than renormalizing over the slots that happened to appear",
			rowID, missing.slot)
	}
	probabilities, err := Softmax(optionLogprobs)
	if err != nil {
		return nil, fmt.Errorf("Row %s: %w", rowID, err)
	}

	readout := "vLLM /v1/chat/completions logprob_token_ids: exact answer slots keyed by token_id over a server-tokenized image prompt"
	if !exact {
		readout = "vLLM /v1/chat/completions top-N over a server-tokenized image prompt: answer slots matched by token id, no fallback"
	}
	requestDigest := hashx.Sha256Hex([]byte(body))
	return &Scored{
		Probabilities:   probabilities,
		OptionLogprobs:  optionLogprobs,
		AnswerTokenIDs:  slots,
		OptionIDs:       append([]string(nil), optionIDs...),
		InputTokens:     usagePromptTokens(response),
		RequestSHA256:   &requestDigest,
		Modality:        "text+image",
		ServerTokenized: true,
		Media:           provenance(images),
		Readout:         readout,
		EngineSeconds:   engineSeconds,
		TotalSeconds:    time.Since(started).Seconds(),
		Endpoint:        index,
	}, nil
}

// chatLogprobEntry returns the first scored position of a chat-completion response.
//
// The chat route nests that position under `logprobs.content[]`, and its `top_logprobs` is a *list*
// of entries carrying a `token` field — unlike `/v1/completions`, whose `top_logprobs` is an object
// keyed by `token_id:<id>`. Reading one shape with the other's parser reports a perfectly good
// answer as unreadable, which is what a live vLLM did.
func chatLogprobEntry(response map[string]any) (map[string]any, bool) {
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return nil, false
	}
	choice, _ := choices[0].(map[string]any)
	logprobs, _ := choice["logprobs"].(map[string]any)
	content, _ := logprobs["content"].([]any)
	if len(content) == 0 {
		return nil, false
	}
	entry, ok := content[0].(map[string]any)
	return entry, ok
}

// chatTokenIDs reports whether a chat position is keyed by token id, which is what makes its slots
// addressable without a local tokenizer.
//
// The signal is the candidate list, because that is the only part this service reads: the sampled
// token beside it is not consulted, so a build that pinned one but answered the list in decoded
// text is still unaddressable here.
func chatTokenIDs(entry map[string]any) bool {
	entries, _ := entry["top_logprobs"].([]any)
	for _, raw := range entries {
		item, _ := raw.(map[string]any)
		if token, ok := item["token"].(string); ok && strings.HasPrefix(token, "token_id:") {
			return true
		}
	}
	return false
}

// parseChatSlots reads each requested slot out of the chat route's top-logprob list.
//
// `logprob_token_ids` makes the requested ids appear in that list whatever the top-N was — measured
// on vLLM 0.27.1, they are present even at `top_logprobs: 1` — so a missing one is a contract
// violation rather than a ranking accident.
func parseChatSlots(entry map[string]any, slots []uint32, rowID string) ([]float64, error) {
	byToken := make(map[string]float64)
	entries, _ := entry["top_logprobs"].([]any)
	for _, raw := range entries {
		item, _ := raw.(map[string]any)
		token, _ := item["token"].(string)
		number, ok := toFloat(item["logprob"])
		if !ok || token == "" {
			continue
		}
		byToken[token] = number
	}

	out := make([]float64, 0, len(slots))
	for _, slot := range slots {
		number, present := byToken[fmt.Sprintf("token_id:%d", slot)]
		if !present {
			return nil, &missingSlot{slot: slot}
		}
		if math.IsNaN(number) || math.IsInf(number, 0) {
			return nil, fmt.Errorf("Row %s: non-finite logprob for slot %d", rowID, slot)
		}
		out = append(out, number)
	}
	return out, nil
}

// chatMessages builds the request the chat route reads: a system turn, then one user turn whose
// content is the caller's text payload followed by the images in the order given.
func chatMessages(system, textPayload string, images []media.Media) []any {
	parts := make([]any, 0, len(images)+1)
	parts = append(parts, jsonx.Object{
		{Key: "type", Value: "text"},
		{Key: "text", Value: textPayload},
	})
	for _, image := range images {
		parts = append(parts, image.ContentPart())
	}
	return []any{
		jsonx.Object{{Key: "role", Value: "system"}, {Key: "content", Value: system}},
		jsonx.Object{{Key: "role", Value: "user"}, {Key: "content", Value: parts}},
	}
}

func provenance(images []media.Media) []any {
	out := make([]any, 0, len(images))
	for _, image := range images {
		out = append(out, image.Provenance())
	}
	return out
}

// usagePromptTokens reads the backend's own prompt-token count, or nil when it reports none: a
// client-side guess would be wrong for a prompt the backend tokenized.
func usagePromptTokens(response map[string]any) *int {
	usage, _ := response["usage"].(map[string]any)
	number, ok := toFloat(usage["prompt_tokens"])
	if !ok || number < 0 {
		return nil
	}
	tokens := int(number)
	return &tokens
}
