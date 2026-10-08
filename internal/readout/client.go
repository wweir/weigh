package readout

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path"
	"strings"
	"sync/atomic"
	"time"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/hashx"
	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/template"
	"github.com/wweir/weigh/internal/tokenize"
)

// Config is everything the readout needs to start. [New] validates it before probing: the dialect
// and the readout tier must both be ones this build knows (a client cannot leave the tier to
// `auto`, which picks a different tier per backend), and the timeout must be positive so no
// backend request is unbounded. Both the binary and the library pass through here.
type Config struct {
	URLs                   []string
	Backend                backend.Choice
	Readout                backend.Readout
	Template               template.Choice
	Timeout                time.Duration
	MaxTokens              int
	AllowTokenizerMismatch bool
	Media                  MediaPolicy
	// Source is the local checkpoint directory; only tokenizer_config.json is read from it.
	Source string
}

// Client scores decisions against a probed fleet.
type Client struct {
	endpoints    []endpoint
	next         atomic.Uint64
	kind         backend.Kind
	tierA        bool
	mediaSupport backend.MediaSupport
	localCeiling int
	contextLimit int
}

type endpoint struct {
	info      EndpointInfo
	transport *backend.Client
	tokenizer *tokenize.Client
}

// New probes every endpoint and returns a client plus the metadata its provenance is read
// against.
//
// Everything that can refuse happens here, before a listener binds: an endpoint that cannot be
// identified, a fleet whose members disagree, a tokenizer mismatch, a backend that does not
// prove the requested readout, and an unknown prompt-length ceiling.
func New(ctx context.Context, cfg Config) (*Client, *Metadata, error) {
	if len(cfg.URLs) == 0 {
		return nil, nil, fmt.Errorf("at least one --url is required")
	}
	// The library entry point refuses the same client-side misconfigurations the binary does,
	// before any probe: a typo'd dialect or tier must not be silently read as vLLM or top-n, and a
	// zero budget would leave every backend request unbounded.
	if _, err := backend.ParseChoice(string(cfg.Backend)); err != nil {
		return nil, nil, err
	}
	readoutChoice, err := backend.ParseReadout(string(cfg.Readout))
	if err != nil {
		return nil, nil, err
	}
	if readoutChoice == backend.ReadoutAuto {
		return nil, nil, fmt.Errorf("Config.Readout must name a tier; auto picks a different tier per backend, which a client cannot leave undecided")
	}
	if cfg.Timeout <= 0 {
		return nil, nil, fmt.Errorf("Config.Timeout must be positive; a zero budget would leave every backend request unbounded")
	}
	// A named family needs no checkpoint; only detection does. The choice is interpreted rather
	// than skipped as a whole, so an explicitly configured family is never dropped and one that
	// cannot be honoured is refused instead of ignored.
	if cfg.Template.Fixed != "" {
		// Checked here rather than only inside Resolve: an unknown family name must be refused
		// whether or not there is a file to compare it against.
		if _, err := template.Parse(string(cfg.Template.Fixed)); err != nil {
			return nil, nil, err
		}
	}

	resolved := template.Resolved{}
	checkpointName := ""
	switch {
	case cfg.Source != "":
		// Resolving against an empty dir would silently pick up a tokenizer_config.json in the
		// process's working directory, so Resolve is only ever called with a directory to read.
		var err error
		resolved, err = template.Resolve(cfg.Template, cfg.Source)
		if err != nil {
			return nil, nil, err
		}
		checkpointName = path.Base(strings.TrimRight(cfg.Source, "/"))
	case cfg.Template.Fixed != "":
		resolved = template.Resolved{Template: cfg.Template.Fixed, Source: "explicit"}
	case cfg.Template.Auto:
		return nil, nil, fmt.Errorf("Config.Template asks to detect the chat template from a checkpoint, but Config.Source names none; point it at a checkpoint directory, or name the family explicitly")
	}
	fleet := len(cfg.URLs) > 1

	client := &Client{localCeiling: cfg.MaxTokens}
	metadata := &Metadata{
		PromptTemplate:       resolved.Template,
		PromptTemplateSource: resolved.Source,
	}
	var base *probeResult
	var baseURL string
	tokenizerMatches := true
	tokenizerChecked := false

	for index, raw := range cfg.URLs {
		url := strings.TrimRight(strings.TrimSpace(raw), "/")
		transport := backend.NewClient(url, cfg.Timeout)

		probed, err := probeEndpoint(ctx, transport, cfg.Backend, cfg.Readout, cfg.Media)
		if err != nil {
			return nil, nil, fmt.Errorf("endpoint %d (%s): %w", index, url, err)
		}

		if fleet {
			if probed.kind == backend.KindVLLM && probed.version == "" {
				return nil, nil, fmt.Errorf("endpoint %d (%s): /version did not answer, so this endpoint's build cannot be compared with the rest of the fleet. Refusing to score rather than assume the endpoints agree", index, url)
			}
			if base == nil {
				base, baseURL = probed, url
			} else if err := checkFleet(base, probed, baseURL, url); err != nil {
				return nil, nil, err
			}
		} else {
			base = probed
			baseURL = url
		}

		// The chat template is read from the local checkpoint, so the checkpoint the backend
		// serves must be the one --model names. The guard compares directory names, because the
		// same checkpoint is routinely mounted at different paths on different hosts.
		if cfg.Source != "" && probed.modelRoot != nil {
			tokenizerChecked = true
			same := path.Base(strings.TrimRight(*probed.modelRoot, "/")) == checkpointName
			tokenizerMatches = tokenizerMatches && same
			if !same && !cfg.AllowTokenizerMismatch {
				return nil, nil, fmt.Errorf(
					"checkpoint mismatch at endpoint %d (%s): --model is %q but the server reports it serves %q. The chat template is read from --model, so a different checkpoint means the prompt is rendered for a model that is not answering. Point --model at the served checkpoint, or pass --allow-tokenizer-mismatch",
					index, url, cfg.Source, *probed.modelRoot)
			}
		}

		if client.contextLimit == 0 {
			client.contextLimit = probed.maxModelLen
		}
		metadata.Endpoints = append(metadata.Endpoints, EndpointInfo{
			URL:          url,
			ServedModels: probed.servedModels,
			ModelRoot:    probed.modelRoot,
			Version:      probed.version,
		})
		// The serving tokenizer names the first served model: the same id every scoring request
		// is sent with, so the ids and the scored position come from one engine.
		client.endpoints = append(client.endpoints, endpoint{
			info:      metadata.Endpoints[len(metadata.Endpoints)-1],
			transport: transport,
			tokenizer: tokenize.New(transport, probed.kind, first(probed.servedModels)),
		})
	}

	client.kind = base.kind
	client.tierA = base.tierA
	client.mediaSupport = base.mediaSupport
	metadata.Backend = base.kind
	metadata.TierA = base.tierA
	metadata.MediaSupport = base.mediaSupport
	metadata.ServingConfig = ServingConfig(base.kind, base.tierA)
	if recipe, ok := MediaServingConfig(base.kind, base.mediaSupport); ok {
		metadata.MediaServingConfig = recipe
	}
	metadata.MaxModelLen = client.contextLimit
	if tokenizerChecked {
		metadata.TokenizerMatches = &tokenizerMatches
	}

	if cfg.Readout == backend.ReadoutExactSlot && !base.tierA {
		return nil, nil, fmt.Errorf("startup did not prove the exact-slot readout on this backend. Point --url at a vLLM or an SGLang server, or pass --readout top-n to accept the measured top-N path")
	}
	if cfg.MaxTokens <= 0 && client.contextLimit <= 0 {
		return nil, nil, fmt.Errorf("the backend reported no max_model_len, so the prompt-length ceiling is unknown; pass --max-prompt-tokens N to state it rather than let this process guess")
	}
	return client, metadata, nil
}

// Decide tokenizes the prompt on one endpoint, proves the answer boundary there, and reads the
// decision.
//
// Endpoint selection is round-robin and happens before the request, so one endpoint serves the
// row's whole lifecycle: the ids and the score come from the same engine.
func (c *Client) Decide(ctx context.Context, rowID, prompt string, optionIDs []string) (*Scored, error) {
	index := int((c.next.Add(1) - 1) % uint64(len(c.endpoints)))
	ep := &c.endpoints[index]

	// The clock starts before tokenization: that is network work now, and total_seconds is what
	// the readout histogram measures.
	started := time.Now()
	prepared, err := ep.tokenizer.Prepare(ctx, rowID, prompt, len(optionIDs))
	if err != nil {
		return nil, contractErrorUnlessBackend(err)
	}
	// The ceiling comes from the engine that tokenized this prompt, since that is the engine that
	// has to accept it; the startup probe's value is only a fallback for a backend that reports
	// none per request.
	contextLimit := prepared.MaxModelLen
	if contextLimit <= 0 {
		contextLimit = c.contextLimit
	}
	limit := tokenize.Limit(c.localCeiling, contextLimit)
	if len(prepared.IDs) > limit {
		return nil, contract("%s: %d input tokens exceed limit %d; no truncation allowed", rowID, len(prepared.IDs), limit)
	}

	var result readoutResult
	switch {
	case c.kind == backend.KindSGLang:
		result, err = c.fetchSGLang(ctx, ep, rowID, prepared)
	case c.tierA:
		result, err = c.fetchVLLMTierA(ctx, ep, rowID, prepared)
	default:
		result, err = c.fetchVLLMTopN(ctx, ep, rowID, prepared)
	}
	if err != nil {
		return nil, err
	}

	promptDigest := hashx.Sha256Hex([]byte(prompt))
	idsDigest := hashx.Sha256Hex([]byte(jsonx.IntList(prepared.IDs)))
	inputTokens := len(prepared.IDs)
	return &Scored{
		Probabilities:   result.probabilities,
		OptionLogprobs:  result.optionLogprobs,
		AnswerTokenIDs:  prepared.Slots,
		OptionIDs:       append([]string(nil), optionIDs...),
		InputTokens:     &inputTokens,
		InputIDsSHA256:  &idsDigest,
		PromptSHA256:    &promptDigest,
		Modality:        "text",
		ServerTokenized: true,
		Readout:         result.readout,
		FallbackUsed:    result.fallbackUsed,
		FallbackSeconds: result.fallbackSeconds,
		EngineSeconds:   result.engineSeconds,
		TotalSeconds:    time.Since(started).Seconds(),
		Endpoint:        index,
	}, nil
}

// contractErrorUnlessBackend wraps a tokenize failure as a ContractError only when it is the
// caller's prompt at fault. A backend that could not answer is not the caller's failure, so its
// error is passed through and the HTTP layer classifies it as a backend error (502/504) rather
// than telling the caller to fix a request that was fine.
func contractErrorUnlessBackend(err error) error {
	var backendErr *tokenize.BackendError
	if errors.As(err, &backendErr) {
		return err
	}
	return &ContractError{Message: err.Error()}
}

// readoutResult is the per-path outcome, before provenance is attached.
type readoutResult struct {
	probabilities   []float64
	optionLogprobs  []float64
	readout         string
	fallbackUsed    bool
	fallbackSeconds float64
	engineSeconds   float64
}

// fetchVLLMTierA names the answer slots by token id and asks for them back keyed by id, so
// there is no top-N dependence and no fallback. A missing slot is a contract violation, not a
// condition to paper over.
func (c *Client) fetchVLLMTierA(ctx context.Context, ep *endpoint, rowID string, prepared *tokenize.Prepared) (readoutResult, error) {
	payload, err := jsonx.Dump(jsonx.Object{
		{Key: "model", Value: first(ep.info.ServedModels)},
		{Key: "prompt", Value: jsonx.Raw(jsonx.IntList(prepared.IDs))},
		{Key: "max_tokens", Value: 1},
		{Key: "temperature", Value: 0},
		{Key: "logprobs", Value: true},
		{Key: "logprob_token_ids", Value: jsonx.Raw(jsonx.IntList(prepared.Slots))},
		{Key: "return_tokens_as_token_ids", Value: true},
	})
	if err != nil {
		return readoutResult{}, err
	}

	send := time.Now()
	var response map[string]any
	if err := ep.transport.PostJSON(ctx, "/v1/completions", []byte(payload), &response); err != nil {
		return readoutResult{}, fmt.Errorf("Row %s: %w", rowID, err)
	}
	engineSeconds := time.Since(send).Seconds()

	top, err := topLogprobs(response, rowID)
	if err != nil {
		return readoutResult{}, err
	}
	optionLogprobs, err := parsePinnedSlots(top, prepared.Slots, rowID)
	if err != nil {
		var missing *missingSlot
		if errors.As(err, &missing) {
			return readoutResult{}, fmt.Errorf("Row %s: response omitted requested slot %d; the server accepted `logprob_token_ids` but did not echo it", rowID, missing.slot)
		}
		return readoutResult{}, err
	}
	probabilities, err := Softmax(optionLogprobs)
	if err != nil {
		return readoutResult{}, fmt.Errorf("Row %s: %w", rowID, err)
	}
	return readoutResult{
		probabilities:  probabilities,
		optionLogprobs: optionLogprobs,
		readout:        "vLLM /v1/completions logprob_token_ids: exact answer slots keyed by token_id, no top-N dependence and no fallback",
		engineSeconds:  engineSeconds,
	}, nil
}

// fetchVLLMTopN is the path for a build that ignores `logprob_token_ids`.
//
// It still asks for token-id keys (`return_tokens_as_token_ids`), because this service has no
// local tokenizer to match decoded text with. A build that answers in decoded text is refused
// rather than guessed at, and a slot outside the returned top-N is recovered through
// /generative_scoring, which is exact.
func (c *Client) fetchVLLMTopN(ctx context.Context, ep *endpoint, rowID string, prepared *tokenize.Prepared) (readoutResult, error) {
	payload, err := jsonx.Dump(jsonx.Object{
		{Key: "model", Value: first(ep.info.ServedModels)},
		{Key: "prompt", Value: jsonx.Raw(jsonx.IntList(prepared.IDs))},
		{Key: "max_tokens", Value: 1},
		{Key: "temperature", Value: 0},
		{Key: "logprobs", Value: maxLogprobs},
		{Key: "return_tokens_as_token_ids", Value: true},
	})
	if err != nil {
		return readoutResult{}, err
	}

	send := time.Now()
	var response map[string]any
	if err := ep.transport.PostJSON(ctx, "/v1/completions", []byte(payload), &response); err != nil {
		return readoutResult{}, fmt.Errorf("Row %s: %w", rowID, err)
	}
	received := time.Now()

	top, err := topLogprobs(response, rowID)
	if err != nil {
		return readoutResult{}, err
	}
	if !pinnedSlotsIn(top) {
		return readoutResult{}, fmt.Errorf("Row %s: this build answers the top-N route with decoded token text and this service has no local tokenizer to match it; use --readout exact-slot against a build that honors `logprob_token_ids`", rowID)
	}

	optionLogprobs, missingErr := parsePinnedSlots(top, prepared.Slots, rowID)
	if missingErr == nil {
		probabilities, err := Softmax(optionLogprobs)
		if err != nil {
			return readoutResult{}, fmt.Errorf("Row %s: %w", rowID, err)
		}
		return readoutResult{
			probabilities:  probabilities,
			optionLogprobs: optionLogprobs,
			readout:        "vLLM /v1/completions max_tokens=1 top_logprobs; subset softmax over declared answer slots matched by token id",
			engineSeconds:  received.Sub(send).Seconds(),
		}, nil
	}
	var missing *missingSlot
	if !errors.As(missingErr, &missing) {
		return readoutResult{}, missingErr
	}

	// A slot outside this server's top-N. /generative_scoring honors token ids and returns
	// probabilities already normalized over the option subset, so they must not be passed
	// through Softmax again.
	fallbackStarted := time.Now()
	probabilities, err := c.fetchViaGenerativeScoring(ctx, ep, rowID, prepared)
	if err != nil {
		return readoutResult{}, err
	}
	fallbackSeconds := time.Since(fallbackStarted).Seconds()
	optionLogprobs = make([]float64, len(probabilities))
	for i, value := range probabilities {
		optionLogprobs[i] = math.Log(value)
	}
	return readoutResult{
		probabilities:   probabilities,
		optionLogprobs:  optionLogprobs,
		readout:         "vLLM /generative_scoring subset softmax over declared answer slots (fallback: option slots outside the /v1/completions top-N)",
		fallbackUsed:    true,
		fallbackSeconds: fallbackSeconds,
		engineSeconds:   received.Sub(send).Seconds() + fallbackSeconds,
	}, nil
}

// fetchViaGenerativeScoring asks for one slot at a time, because the route normalizes over the
// label list it is handed: putting the wanted slot first makes it the answer that comes back.
func (c *Client) fetchViaGenerativeScoring(ctx context.Context, ep *endpoint, rowID string, prepared *tokenize.Prepared) ([]float64, error) {
	scores := make([]float64, 0, len(prepared.Slots))
	for i, slot := range prepared.Slots {
		labels := make([]uint32, 0, len(prepared.Slots))
		labels = append(labels, slot)
		for j, other := range prepared.Slots {
			if j != i {
				labels = append(labels, other)
			}
		}
		payload, err := jsonx.Dump(jsonx.Object{
			{Key: "model", Value: first(ep.info.ServedModels)},
			{Key: "query", Value: jsonx.Raw(jsonx.IntList(prepared.IDs))},
			{Key: "items", Value: []any{""}},
			{Key: "label_token_ids", Value: jsonx.Raw(jsonx.IntList(labels))},
			{Key: "apply_softmax", Value: true},
		})
		if err != nil {
			return nil, err
		}
		var response map[string]any
		if err := ep.transport.PostJSON(ctx, "/generative_scoring", []byte(payload), &response); err != nil {
			return nil, fmt.Errorf("Row %s: %w", rowID, err)
		}
		data, _ := response["data"].([]any)
		if len(data) == 0 {
			return nil, fmt.Errorf("Row %s: /generative_scoring returned no scores", rowID)
		}
		entry, _ := data[0].(map[string]any)
		score, ok := toFloat(entry["score"])
		if !ok {
			return nil, fmt.Errorf("Row %s: /generative_scoring returned no scores", rowID)
		}
		scores = append(scores, score)
	}

	total := 0.0
	for _, score := range scores {
		if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 {
			return nil, fmt.Errorf("Row %s: /generative_scoring returned unusable scores %v", rowID, scores)
		}
		total += score
	}
	if total <= 0 {
		return nil, fmt.Errorf("Row %s: /generative_scoring returned unusable scores %v", rowID, scores)
	}
	for i := range scores {
		scores[i] /= total
	}
	return scores, nil
}

// fetchSGLang uses the native route, which names the answer slots by token id and returns their
// logprobs per output position.
func (c *Client) fetchSGLang(ctx context.Context, ep *endpoint, rowID string, prepared *tokenize.Prepared) (readoutResult, error) {
	// Sampling parameters are sent explicitly and neutrally: SGLang defaults come from the
	// model's generation_config.json, and any penalty or truncation applied before the logprobs
	// would silently move the numbers this readout normalizes.
	payload, err := jsonx.Dump(jsonx.Object{
		{Key: "input_ids", Value: jsonx.Raw(jsonx.IntList(prepared.IDs))},
		{Key: "sampling_params", Value: jsonx.Object{
			{Key: "max_new_tokens", Value: 1},
			{Key: "temperature", Value: 0},
			{Key: "top_p", Value: 1},
			{Key: "top_k", Value: -1},
			{Key: "min_p", Value: 0},
			{Key: "frequency_penalty", Value: 0},
			{Key: "presence_penalty", Value: 0},
			{Key: "repetition_penalty", Value: 1},
		}},
		{Key: "return_logprob", Value: true},
		{Key: "logprob_start_len", Value: -1},
		{Key: "token_ids_logprob", Value: jsonx.Raw(jsonx.IntList(prepared.Slots))},
	})
	if err != nil {
		return readoutResult{}, err
	}

	send := time.Now()
	var response map[string]any
	if err := ep.transport.PostJSON(ctx, "/generate", []byte(payload), &response); err != nil {
		return readoutResult{}, fmt.Errorf("Row %s: %w", rowID, err)
	}
	engineSeconds := time.Since(send).Seconds()

	optionLogprobs, err := parseSGLangTokenIDs(response, prepared.Slots, rowID)
	if err != nil {
		return readoutResult{}, err
	}
	probabilities, err := Softmax(optionLogprobs)
	if err != nil {
		return readoutResult{}, fmt.Errorf("Row %s: %w", rowID, err)
	}
	return readoutResult{
		probabilities:  probabilities,
		optionLogprobs: optionLogprobs,
		readout:        "SGLang native /generate token_ids_logprob: exact answer slots keyed by token_id, no top-N dependence and no fallback",
		engineSeconds:  engineSeconds,
	}, nil
}

// topLogprobs pulls the first generated position's logprob object out of a completion response.
func topLogprobs(response map[string]any, rowID string) (map[string]any, error) {
	choices, _ := response["choices"].([]any)
	if len(choices) == 0 {
		return nil, fmt.Errorf("Row %s: response carried no choices", rowID)
	}
	choice, _ := choices[0].(map[string]any)
	logprobs, ok := choice["logprobs"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("Row %s: response carried no top_logprobs", rowID)
	}
	top, ok := logprobs["top_logprobs"].([]any)
	if !ok || len(top) == 0 {
		return nil, fmt.Errorf("Row %s: response carried no top_logprobs", rowID)
	}
	first, ok := top[0].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("Row %s: top_logprobs was not an object", rowID)
	}
	return first, nil
}

// missingSlot reports which requested slot a response did not carry, so each readout names the
// cause in its own terms: a Tier A contract violation, or a slot outside the returned top-N.
type missingSlot struct{ slot uint32 }

func (e *missingSlot) Error() string {
	return fmt.Sprintf("slot %d was not returned", e.slot)
}

// pinnedSlotsIn reports whether a top-logprobs object is keyed by token id, which is what makes a
// slot addressable without a local tokenizer.
func pinnedSlotsIn(top map[string]any) bool {
	for key := range top {
		if strings.HasPrefix(key, "token_id:") {
			return true
		}
	}
	return false
}

// parsePinnedSlots reads the requested slots out of a `token_id:<id>` keyed object.
func parsePinnedSlots(top map[string]any, slots []uint32, rowID string) ([]float64, error) {
	out := make([]float64, 0, len(slots))
	for _, slot := range slots {
		number, ok := toFloat(top[fmt.Sprintf("token_id:%d", slot)])
		if !ok {
			return nil, &missingSlot{slot: slot}
		}
		if math.IsNaN(number) || math.IsInf(number, 0) {
			return nil, fmt.Errorf("Row %s: non-finite logprob for slot %d", rowID, slot)
		}
		out = append(out, number)
	}
	return out, nil
}

// parseSGLangTokenIDs reads the pinned slots out of meta_info.output_token_ids_logprobs.
func parseSGLangTokenIDs(response map[string]any, slots []uint32, rowID string) ([]float64, error) {
	meta, _ := response["meta_info"].(map[string]any)
	entries, _ := meta["output_token_ids_logprobs"].([]any)
	if len(entries) == 0 {
		return nil, fmt.Errorf("Row %s: /generate returned no meta_info.output_token_ids_logprobs", rowID)
	}
	// Only the first output position is scored.
	position, ok := entries[0].([]any)
	if !ok {
		return nil, fmt.Errorf("Row %s: token-logprob entry was not an array", rowID)
	}
	byToken := make(map[int64]float64, len(position))
	for _, entry := range position {
		pair, ok := entry.([]any)
		if !ok || len(pair) < 2 {
			return nil, fmt.Errorf("Row %s: unusable token-logprob entry %v", rowID, entry)
		}
		logprob, okLogprob := toFloat(pair[0])
		tokenID, okToken := toInt(pair[1])
		if !okLogprob || !okToken || math.IsNaN(logprob) || math.IsInf(logprob, 0) {
			return nil, fmt.Errorf("Row %s: unusable token-logprob entry %v", rowID, entry)
		}
		byToken[tokenID] = logprob
	}

	out := make([]float64, 0, len(slots))
	for _, slot := range slots {
		logprob, present := byToken[int64(slot)]
		if !present {
			return nil, fmt.Errorf("Row %s: SGLang omitted requested slot %d from output_token_ids_logprobs", rowID, slot)
		}
		out = append(out, logprob)
	}
	return out, nil
}

// toFloat reads a number out of a decoded backend response. encoding/json decodes every JSON
// number into a float64 on this path — backend.Client unmarshals into map[string]any — so that is
// the one shape that can occur; anything else is a response this service does not understand and
// is refused rather than coerced.
func toFloat(value any) (float64, bool) {
	number, ok := value.(float64)
	return number, ok
}

// toInt reads a token id out of a decoded backend response, on the same float64-only path as
// [toFloat].
func toInt(value any) (int64, bool) {
	number, ok := value.(float64)
	return int64(number), ok
}
