package readout

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strings"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/media"
	"github.com/wweir/weigh/internal/tokenize"
)

// MediaPolicy is how far the operator let this deployment go with images.
type MediaPolicy int

const (
	// MediaOff refuses image parts.
	MediaOff MediaPolicy = iota
	// MediaExactSlot accepts media only where the exact-slot route was proven.
	MediaExactSlot
	// MediaTopN additionally accepts the best-effort top-N media route.
	MediaTopN
)

// Enabled reports whether any media route is permitted.
func (p MediaPolicy) Enabled() bool { return p != MediaOff }

// maxLogprobs is the measured cap on `logprobs`: above it the backend answers 400.
const maxLogprobs = 20

// probeResult is everything the startup probe learned about one endpoint.
type probeResult struct {
	kind         backend.Kind
	tierA        bool
	mediaSupport backend.MediaSupport
	servedModels []string
	modelRoot    *string
	version      string
	maxModelLen  int
}

// probeEndpoint decides the dialect and proves what it can about the endpoint.
//
// An explicit --backend skips detection but still verifies identity: the operator's claim is
// checked, not trusted.
func probeEndpoint(
	ctx context.Context,
	transport *backend.Client,
	choice backend.Choice,
	wantReadout backend.Readout,
	policy MediaPolicy,
) (*probeResult, error) {
	kind := backend.Kind(choice)
	if choice == backend.ChoiceAuto {
		detected, err := detectBackend(ctx, transport)
		if err != nil {
			return nil, err
		}
		kind = detected
	}
	// The probe tokenizer carries no model id: vLLM resolves a single served model on its own,
	// and the serving client is rebuilt with the served id once the probe has found it.
	probeTokenizer := tokenize.New(transport, kind, "")
	// Every deployment needs tokenization, whatever the readout tier and whether or not media is
	// allowed: without this endpoint there is no way to build a scoring request at all, so it is
	// proven here rather than discovered on the first request.
	if err := probeTokenizer.Probe(ctx); err != nil {
		return nil, err
	}
	if kind == backend.KindSGLang {
		return probeSGLang(ctx, transport, wantReadout)
	}
	return probeVLLM(ctx, transport, wantReadout, policy, probeTokenizer)
}

// detectBackend asks the two dialects' identity routes. A transport failure is not a negative
// answer, so one timeout must not label an SGLang server as vLLM.
func detectBackend(ctx context.Context, transport *backend.Client) (backend.Kind, error) {
	var modelInfo struct {
		ModelPath *string `json:"model_path"`
	}
	infoErr := transport.GetJSON(ctx, "/get_model_info", &modelInfo)
	if infoErr == nil {
		if modelInfo.ModelPath != nil {
			return backend.KindSGLang, nil
		}
		infoErr = errors.New("an HTTP 200 without a `model_path`")
	}

	var version struct {
		Version string `json:"version"`
	}
	if err := transport.GetJSON(ctx, "/version", &version); err == nil {
		return backend.KindVLLM, nil
	}
	var models struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := transport.GetJSON(ctx, "/v1/models", &models); err == nil {
		return backend.KindVLLM, nil
	}
	return "", fmt.Errorf(
		"cannot determine the backend at %s: neither /get_model_info nor the vLLM routes (/version, /v1/models) answered (last error: %w)",
		transport.Base(), infoErr)
}

func probeVLLM(
	ctx context.Context,
	transport *backend.Client,
	wantReadout backend.Readout,
	policy MediaPolicy,
	tokenizer *tokenize.Client,
) (*probeResult, error) {
	result := &probeResult{kind: backend.KindVLLM, mediaSupport: backend.MediaNone}

	// /version is optional: a single endpoint may legitimately not expose it.
	var version struct {
		Version string `json:"version"`
	}
	if err := transport.GetJSON(ctx, "/version", &version); err == nil {
		result.version = version.Version
	}

	var models struct {
		Data []struct {
			ID          string  `json:"id"`
			Root        *string `json:"root"`
			MaxModelLen *int    `json:"max_model_len"`
		} `json:"data"`
	}
	if err := transport.GetJSON(ctx, "/v1/models", &models); err != nil {
		return nil, err
	}
	for _, entry := range models.Data {
		if entry.ID != "" {
			result.servedModels = append(result.servedModels, entry.ID)
		}
	}
	if len(result.servedModels) == 0 {
		return nil, fmt.Errorf("No served models reported by %s; is the server up?", transport.Base())
	}
	if len(models.Data) > 0 {
		result.modelRoot = models.Data[0].Root
		if models.Data[0].MaxModelLen != nil {
			result.maxModelLen = *models.Data[0].MaxModelLen
			if result.maxModelLen <= 0 {
				return nil, fmt.Errorf("/v1/models reported an unusable max_model_len (%d)", result.maxModelLen)
			}
		}
	}

	if wantReadout == backend.ReadoutExactSlot {
		supported, err := probeVLLMTierA(ctx, transport, result.servedModels[0], tokenizer)
		if err != nil {
			return nil, err
		}
		if !supported {
			return nil, fmt.Errorf(
				"--readout exact-slot was requested, but this build did not honor `logprob_token_ids` on /v1/completions (no `token_id:` keys came back). Use --readout top-n or a build that supports it")
		}
		result.tierA = true
	}

	if policy.Enabled() {
		support, err := probeVLLMMedia(ctx, transport, result.servedModels[0], policy, tokenizer)
		if err != nil {
			return nil, err
		}
		result.mediaSupport = support
	}
	return result, nil
}

// probeVLLMTierA proves `logprob_token_ids` support by asking for a slot by id and checking that
// the response is keyed by id. The signal is the response key space, not a version number: a
// build can accept the field and ignore it.
func probeVLLMTierA(ctx context.Context, transport *backend.Client, model string, tokenizer *tokenize.Client) (bool, error) {
	slots, err := tokenizer.SlotIDs(ctx, 2)
	if err != nil {
		return false, err
	}
	prompt, _, err := tokenizer.Encode(ctx, ".")
	if err != nil {
		return false, err
	}

	payload, err := jsonx.Dump(jsonx.Object{
		{Key: "model", Value: model},
		{Key: "prompt", Value: jsonx.Raw(jsonx.IntList(prompt))},
		{Key: "max_tokens", Value: 1},
		{Key: "temperature", Value: 0},
		{Key: "logprobs", Value: true},
		{Key: "logprob_token_ids", Value: jsonx.Raw(jsonx.IntList(slots[:1]))},
		{Key: "return_tokens_as_token_ids", Value: true},
	})
	if err != nil {
		return false, err
	}

	var response map[string]any
	err = transport.PostJSON(ctx, "/v1/completions", []byte(payload), &response)
	if err != nil {
		// A 400 or 422 is a real negative answer: the server understood the field and refused
		// it. Anything else (5xx, 429, 401, transport) is a failure, not a negative.
		var status *backend.StatusError
		if errors.As(err, &status) && (status.Code == 400 || status.Code == 422) {
			return false, nil
		}
		return false, fmt.Errorf("Tier A probe on /v1/completions failed: %w", err)
	}
	return hasPinnedKeys(response), nil
}

// hasPinnedKeys reports whether a completion response named at least one slot by token id.
func hasPinnedKeys(response map[string]any) bool {
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return false
	}
	choice, _ := choices[0].(map[string]any)
	logprobs, _ := choice["logprobs"].(map[string]any)
	top, _ := logprobs["top_logprobs"].([]any)
	if len(top) == 0 {
		return false
	}
	first, _ := top[0].(map[string]any)
	return pinnedSlotsIn(first)
}

// probeVLLMMedia proves what the chat route can do with an image.
func probeVLLMMedia(ctx context.Context, transport *backend.Client, model string, policy MediaPolicy, tokenizer *tokenize.Client) (backend.MediaSupport, error) {
	slots, err := tokenizer.SlotIDs(ctx, 2)
	if err != nil {
		return backend.MediaNone, err
	}

	payload := func(pinned bool) (string, error) {
		object := jsonx.Object{
			{Key: "model", Value: model},
			{Key: "messages", Value: []any{jsonx.Object{
				{Key: "role", Value: "user"},
				{Key: "content", Value: []any{
					jsonx.Object{{Key: "type", Value: "text"}, {Key: "text", Value: "."}},
					jsonx.Object{{Key: "type", Value: "image_url"}, {Key: "image_url", Value: jsonx.Object{{Key: "url", Value: media.ProbeImageDataURI}}}},
				}},
			}}},
			{Key: "max_tokens", Value: 1},
			{Key: "temperature", Value: 0},
			{Key: "logprobs", Value: true},
			{Key: "top_logprobs", Value: maxLogprobs},
		}
		if pinned {
			object = append(object,
				jsonx.Field{Key: "logprob_token_ids", Value: jsonx.Raw(jsonx.IntList(slots[:1]))},
				jsonx.Field{Key: "return_tokens_as_token_ids", Value: true},
			)
		}
		return jsonx.Dump(object)
	}

	body, err := payload(true)
	if err != nil {
		return backend.MediaNone, err
	}
	var response map[string]any
	err = transport.PostJSON(ctx, "/v1/chat/completions", []byte(body), &response)
	if err == nil {
		support, err := mediaSupportOf(response, transport.Base())
		if err != nil {
			return backend.MediaNone, err
		}
		return acceptMedia(support, policy), nil
	}

	var status *backend.StatusError
	if !errors.As(err, &status) || (status.Code != 400 && status.Code != 422) {
		return backend.MediaNone, fmt.Errorf("media probe on /v1/chat/completions failed: %w", err)
	}

	// The pinned form was refused: a real negative answer. Only the best-effort policy may retry
	// without the pins.
	if policy != MediaTopN {
		return backend.MediaNone, nil
	}
	body, err = payload(false)
	if err != nil {
		return backend.MediaNone, err
	}
	response = nil
	if err := transport.PostJSON(ctx, "/v1/chat/completions", []byte(body), &response); err != nil {
		if errors.As(err, &status) && (status.Code == 400 || status.Code == 422) {
			return backend.MediaNone, nil
		}
		return backend.MediaNone, fmt.Errorf("media probe on /v1/chat/completions failed: %w", err)
	}
	// The exact-slot route is proven absent (the pinned form was refused above), so whatever the
	// unpinned form answers is the best-effort route. The response shape cannot be used to tell the
	// two apart here, because both forms ask for token-id keys — this service has no local
	// tokenizer to read decoded text with — so a token-id-keyed answer is expected of either.
	if _, err := mediaSupportOf(response, transport.Base()); err != nil {
		return backend.MediaNone, err
	}
	return acceptMedia(backend.MediaChatTopN, policy), nil
}

// acceptMedia applies the operator's policy to what the probe proved.
//
// `--allow-media` asks for the exact-slot media route specifically: a build that answers under
// decoded text must disable media rather than be silently upgraded to a best-effort readout the
// operator did not opt into. `--allow-media-topn` is that opt-in, and accepts both routes.
func acceptMedia(support backend.MediaSupport, policy MediaPolicy) backend.MediaSupport {
	if policy == MediaExactSlot && support == backend.MediaChatTopN {
		return backend.MediaNone
	}
	return support
}

// mediaSupportOf classifies a chat-route answer. The exact-slot route is not decided here: that
// comes from the pinned form having been accepted (see probeVLLMMedia). What this reads is whether
// the answer can be addressed by token id at all.
func mediaSupportOf(response map[string]any, base string) (backend.MediaSupport, error) {
	entry, ok := chatLogprobEntry(response)
	if !ok {
		return backend.MediaNone, fmt.Errorf("the media probe on %s answered without a `choices[].logprobs.content` entry, so this build's response cannot be read as a decision", base)
	}
	if !chatTokenIDs(entry) {
		// Keyed by decoded text, and this service has no local tokenizer to match it with.
		// Disabling media is the honest answer; the text path is unaffected.
		return backend.MediaNone, nil
	}
	return backend.MediaChatExactSlot, nil
}

func probeSGLang(ctx context.Context, transport *backend.Client, wantReadout backend.Readout) (*probeResult, error) {
	if wantReadout == backend.ReadoutTopN {
		return nil, fmt.Errorf("--readout top-n is not implemented for SGLang; its adapter is exact-slot only")
	}

	var info struct {
		ModelPath     *string         `json:"model_path"`
		ContextLength *int            `json:"context_length"`
		Version       json.RawMessage `json:"version"`
	}
	if err := transport.GetJSON(ctx, "/get_model_info", &info); err != nil {
		return nil, fmt.Errorf("/get_model_info: %w", err)
	}
	if info.ModelPath == nil {
		return nil, fmt.Errorf("/get_model_info reported no `model_path`; not an SGLang server")
	}
	if info.ContextLength != nil {
		if *info.ContextLength <= 0 {
			return nil, fmt.Errorf("/get_model_info reported an unusable context_length (%d)", *info.ContextLength)
		}
	}

	result := &probeResult{
		kind:         backend.KindSGLang,
		tierA:        true, // native token_ids_logprob is exact-slot by construction
		mediaSupport: backend.MediaNone,
		modelRoot:    info.ModelPath,
		maxModelLen:  0,
		version:      strings.Trim(string(info.Version), `"`),
	}
	if info.ContextLength != nil {
		result.maxModelLen = *info.ContextLength
	}

	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := transport.GetJSON(ctx, "/v1/models", &models); err == nil {
		for _, entry := range models.Data {
			if entry.ID != "" {
				result.servedModels = append(result.servedModels, entry.ID)
			}
		}
	}
	if len(result.servedModels) == 0 {
		result.servedModels = []string{*info.ModelPath}
	}
	return result, nil
}

// checkFleet refuses a fleet whose endpoints do not demonstrably agree on everything that
// affects a scored number. `None == None` is never a pass: a field missing on any endpoint is a
// refusal, because "the endpoints are identical" must be verified rather than assumed.
func checkFleet(base *probeResult, current *probeResult, baseURL, currentURL string) error {
	if base.kind != current.kind {
		return fmt.Errorf("Fleet mismatch at %s: this endpoint speaks %s but %s speaks %s. A mixed fleet would silently mix two readouts under one artifact",
			currentURL, current.kind, baseURL, base.kind)
	}
	if first(base.servedModels) != first(current.servedModels) {
		return fmt.Errorf("Fleet mismatch at %s: first served model id is %q but %s reported %q. Requests are sent with that id, so the endpoints must agree on it",
			currentURL, first(current.servedModels), baseURL, first(base.servedModels))
	}
	if !sameCheckpoint(base.modelRoot, current.modelRoot) {
		return fmt.Errorf("Fleet mismatch at %s: checkpoint directory is %s but %s reported %s. Scoring across different checkpoints would mix two models' probabilities under one artifact",
			currentURL, describe(base.modelRoot), baseURL, describe(current.modelRoot))
	}
	if base.tierA != current.tierA {
		return fmt.Errorf("Fleet mismatch at %s: readout tier differs (Tier A=%t). Every endpoint must reach the same readout, or a row's value would also depend on which replica served it",
			currentURL, current.tierA)
	}
	if base.mediaSupport != current.mediaSupport {
		return fmt.Errorf("Fleet mismatch at %s: media support differs (%s). Every endpoint must reach the same media readout, or a media row's provenance would depend on which replica served it",
			currentURL, current.mediaSupport)
	}
	if base.version != current.version {
		return fmt.Errorf("Fleet mismatch at %s: server version is %q but %s reported %q", currentURL, current.version, baseURL, base.version)
	}
	if base.maxModelLen != current.maxModelLen {
		return fmt.Errorf("Fleet mismatch at %s: max_model_len is %d but %s reported %d. All endpoints must serve the same configuration",
			currentURL, current.maxModelLen, baseURL, base.maxModelLen)
	}
	return nil
}

func first(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func sameCheckpoint(left, right *string) bool {
	if left == nil || right == nil {
		return false
	}
	return path.Base(strings.TrimRight(*left, "/")) == path.Base(strings.TrimRight(*right, "/"))
}

func describe(root *string) string {
	if root == nil {
		return "unreported"
	}
	return fmt.Sprintf("%q", *root)
}
