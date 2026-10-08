// Package tokenize asks the backend to tokenize, and proves the answer boundary with the
// backend's own tokenizer.
//
// This replaces a local tokenizer on purpose. The readout submits the prompt as token ids and
// names the answer slots by token id, so those ids must be the ids the scoring engine uses; a
// second, local implementation can disagree with the engine, and the disagreement is silent —
// the request is accepted and returns plausible but wrong probabilities. Asking the backend
// removes that class of error instead of testing for it.
//
// What is proven here, for one rendered prompt P and N options:
//
//  1. tokenize(P) is non-empty,
//  2. for every letter L, tokenize(P + L) == tokenize(P) + [one token]: the letter appends
//     exactly one token, so the scored position is the one the caller means,
//  3. the resulting tokens are distinct, so a slot id identifies one option.
//
// A tokenizer that merges " A" into the preceding token fails step 2, and the request is
// refused rather than scored at the wrong position.
package tokenize

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/slots"
)

const (
	// BoundaryWindow is how much of the prompt tail identifies a boundary proof. Two prompts
	// that agree on their last ids tokenize an appended letter the same way in every tokenizer
	// this service targets; the window is an accepted approximation, stated here rather than
	// hidden.
	BoundaryWindow = 8
	// BoundaryCacheCap bounds the memo. It is cleared wholesale when full: the entries are
	// cheap to recompute and a long-lived server must not grow without limit.
	BoundaryCacheCap = 4096
)

// BackendError reports that the tokenize endpoint did not answer, or answered something this
// client cannot use. It is the backend's failure, not the caller's prompt, so a readout must not
// report it as a contract refusal: the request may be perfectly valid and the tokenizer merely
// unavailable.
type BackendError struct{ Err error }

func (e *BackendError) Error() string { return e.Err.Error() }
func (e *BackendError) Unwrap() error { return e.Err }

// Prepared is one prompt, tokenized and boundary-proven.
type Prepared struct {
	// IDs are the prompt's token ids, as the backend tokenized them.
	IDs []uint32
	// Slots[i] is the token id of answer slot slots.Letter(i).
	Slots []uint32
	// MaxModelLen is the context the backend reported for this tokenization, 0 when it
	// reported none.
	MaxModelLen int
}

// Client is the tokenize transport for one backend.
type Client struct {
	transport *backend.Client
	kind      backend.Kind
	// model is sent to vLLM, which requires it to resolve the served checkpoint. SGLang's
	// tokenize route does not take one.
	model string

	mu     sync.Mutex
	proven map[string][]uint32
	// letters memoizes the per-count slot ids for [Client.SlotIDs].
	letters map[int][]uint32
}

// New builds a tokenize client. model may be empty when the backend does not need it.
func New(transport *backend.Client, kind backend.Kind, model string) *Client {
	return &Client{
		transport: transport,
		kind:      kind,
		model:     model,
		proven:    make(map[string][]uint32),
		letters:   make(map[int][]uint32),
	}
}

// Path is the tokenize endpoint for a backend dialect.
func Path(kind backend.Kind) string {
	if kind == backend.KindSGLang {
		return "/v1/tokenize"
	}
	return "/tokenize"
}

// Probe asks the backend to tokenize once and prove the minimum answer-slot contract. It is the
// startup precondition: without a working tokenize route there is no way to build a scoring
// request, so the process must refuse to start rather than serve unlabelled numbers.
//
// The message names the route and the operation but does not assert a cause: a missing route and
// a tokenizer that cannot support the slot contract fail here for different reasons, and the
// wrapped error carries which one it was.
func (c *Client) Probe(ctx context.Context) error {
	if _, err := c.Prepare(ctx, "probe", ".", slots.MinValues); err != nil {
		return fmt.Errorf("POST %s must tokenize a probe prompt before this service can score anything: %w", Path(c.kind), err)
	}
	return nil
}

// Prepare tokenizes the prompt and proves the answer boundary for optionCount slots.
func (c *Client) Prepare(ctx context.Context, rowID, prompt string, optionCount int) (*Prepared, error) {
	if optionCount < slots.MinValues {
		return nil, fmt.Errorf("%d options are not a decision; at least %d are required", optionCount, slots.MinValues)
	}
	if optionCount > slots.MaxValues {
		return nil, fmt.Errorf("%d options exceed the %d answer slots", optionCount, slots.MaxValues)
	}

	ids, maxModelLen, err := c.Encode(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("cannot tokenize the prompt for %s: %w", rowID, err)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("%s: prompt tokenizes to zero tokens", rowID)
	}

	if cached, ok := c.cached(ids, optionCount); ok {
		return &Prepared{IDs: ids, Slots: cached, MaxModelLen: maxModelLen}, nil
	}

	slotIDs := make([]uint32, 0, optionCount)
	for index := 0; index < optionCount; index++ {
		letter := slots.Letter(index)
		appended, _, err := c.Encode(ctx, prompt+letter)
		if err != nil {
			return nil, fmt.Errorf("cannot tokenize the answer boundary for %s: %w", rowID, err)
		}
		if len(appended) != len(ids)+1 || !equalIDs(appended[:len(ids)], ids) {
			return nil, fmt.Errorf(
				"answer boundary changes tokenization for slot %s: %s does not append the letter as exactly one token, so the scored position would not be the one the caller means",
				letter, Path(c.kind))
		}
		slotIDs = append(slotIDs, appended[len(ids)])
	}
	if !distinct(slotIDs) {
		return nil, fmt.Errorf("answer-slot tokens collide: two letters map to the same token, so a slot id would not identify one option")
	}

	c.remember(ids, optionCount, slotIDs)
	return &Prepared{IDs: ids, Slots: slotIDs, MaxModelLen: maxModelLen}, nil
}

// SlotIDs returns the token id of each slot letter, requiring every letter to be exactly one
// token and the tokens to be distinct.
//
// Unlike Prepare this proves nothing about a prompt: it is the answer-slot contract alone, which
// is all a server-tokenized prompt (the media path) can support.
func (c *Client) SlotIDs(ctx context.Context, count int) ([]uint32, error) {
	if count < slots.MinValues || count > slots.MaxValues {
		return nil, fmt.Errorf("%d options are outside the %d..=%d answer slots", count, slots.MinValues, slots.MaxValues)
	}
	c.mu.Lock()
	if cached, ok := c.letters[count]; ok {
		c.mu.Unlock()
		return cached, nil
	}
	c.mu.Unlock()

	ids := make([]uint32, 0, count)
	for index := 0; index < count; index++ {
		letter := slots.Letter(index)
		encoded, _, err := c.Encode(ctx, letter)
		if err != nil {
			return nil, fmt.Errorf("cannot tokenize answer slot %s: %w", letter, err)
		}
		if len(encoded) != 1 {
			return nil, fmt.Errorf("answer slot %s is not exactly one token (%d tokens)", letter, len(encoded))
		}
		ids = append(ids, encoded[0])
	}
	if !distinct(ids) {
		return nil, fmt.Errorf("answer-slot tokens collide: two letters map to the same token, so a slot id would not identify one option")
	}

	c.mu.Lock()
	c.letters[count] = ids
	c.mu.Unlock()
	return ids, nil
}

// Encode tokenizes one string and returns its ids plus the backend's reported context length.
//
// add_special_tokens is always false: the prompt already carries whatever the chat template
// adds, and a second BOS would be a token the scoring request then pins twice.
func (c *Client) Encode(ctx context.Context, text string) ([]uint32, int, error) {
	payload := jsonx.Object{
		{Key: "prompt", Value: text},
		{Key: "add_special_tokens", Value: false},
	}
	if c.kind == backend.KindVLLM && c.model != "" {
		// vLLM resolves the served checkpoint from this. SGLang's route takes no model field,
		// and sending one it does not declare would be an unvalidated extra field.
		payload = append(jsonx.Object{{Key: "model", Value: c.model}}, payload...)
	}
	body, err := jsonx.Dump(payload)
	if err != nil {
		return nil, 0, err
	}

	var response struct {
		Tokens      []uint32 `json:"tokens"`
		Count       int      `json:"count"`
		MaxModelLen int      `json:"max_model_len"`
	}
	if err := c.transport.PostJSON(ctx, Path(c.kind), []byte(body), &response); err != nil {
		return nil, 0, &BackendError{Err: err}
	}
	// A list-shaped prompt would decode `tokens` as a nested array and fail; so would any other
	// shape this service does not send. Saying so beats a wrong id list, and a response that
	// contradicts itself is the backend's failure rather than the caller's.
	if response.Count != len(response.Tokens) {
		return nil, 0, &BackendError{Err: fmt.Errorf("POST %s reported count %d for %d tokens; refusing an inconsistent tokenization",
			Path(c.kind), response.Count, len(response.Tokens))}
	}
	if response.MaxModelLen < 0 {
		response.MaxModelLen = 0
	}
	return response.Tokens, response.MaxModelLen, nil
}

// Limit is the prompt-token ceiling: the local flag, further capped by the backend's own
// context minus the one token reserved for the scored position.
//
// local <= 0 means the operator stated no ceiling; context <= 0 means the backend reported none.
func Limit(local, context int) int {
	if context > 0 {
		capped := context - 1
		if capped < 0 {
			capped = 0
		}
		if local <= 0 || local > capped {
			return capped
		}
	}
	if local <= 0 {
		return math.MaxInt
	}
	return local
}

func (c *Client) cached(ids []uint32, optionCount int) ([]uint32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	slotIDs, ok := c.proven[boundaryKey(ids, optionCount)]
	return slotIDs, ok
}

func (c *Client) remember(ids []uint32, optionCount int, slotIDs []uint32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.proven) >= BoundaryCacheCap {
		c.proven = make(map[string][]uint32)
	}
	c.proven[boundaryKey(ids, optionCount)] = slotIDs
}

func boundaryKey(ids []uint32, optionCount int) string {
	start := len(ids) - BoundaryWindow
	if start < 0 {
		start = 0
	}
	key := make([]byte, 0, (len(ids)-start)*4+2)
	for _, id := range ids[start:] {
		key = append(key, byte(id), byte(id>>8), byte(id>>16), byte(id>>24))
	}
	key = append(key, byte(optionCount), byte(optionCount>>8))
	return string(key)
}

func equalIDs(left, right []uint32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func distinct(values []uint32) bool {
	seen := make(map[uint32]bool, len(values))
	for _, value := range values {
		if seen[value] {
			return false
		}
		seen[value] = true
	}
	return true
}
