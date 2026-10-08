package tokenize

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/slots"
)

// fakeTokenizer is a hand-written stand-in for the backend's tokenizer: one token per rune, so
// the answer boundary holds by construction. The switches make it do the wrong thing on
// purpose, which is how the proof is tested adversarially.
type fakeTokenizer struct {
	// merge makes the final " X" collapse into one token, so appending a letter does not grow
	// the id list — the exact silent failure the boundary proof exists to catch.
	merge bool
	// collide maps every uppercase letter to one token, so two slots would name one option.
	collide bool
	// countMismatch makes the response contradict itself.
	countMismatch bool
	// status, when non-zero, is returned instead of a tokenization.
	status int

	mu     sync.Mutex
	calls  int
	bodies []map[string]any
	paths  []string
}

func (f *fakeTokenizer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	f.calls++
	f.bodies = append(f.bodies, body)
	f.paths = append(f.paths, r.URL.Path)
	f.mu.Unlock()

	if f.status != 0 {
		w.WriteHeader(f.status)
		_, _ = w.Write([]byte(`{"error":"refused"}`))
		return
	}

	prompt, _ := body["prompt"].(string)
	special, _ := body["add_special_tokens"].(bool)
	ids := f.encode(prompt, special)
	count := len(ids)
	if f.countMismatch {
		count = len(ids) + 1
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"tokens": ids, "count": count, "max_model_len": 4096,
	})
}

func (f *fakeTokenizer) encode(text string, special bool) []uint32 {
	ids := make([]uint32, 0, len(text)+1)
	if special {
		ids = append(ids, 1)
	}
	runes := []rune(text)
	for i, r := range runes {
		if f.collide && r >= 'A' && r <= 'P' {
			ids = append(ids, 700)
			continue
		}
		if f.merge && i > 0 && runes[i-1] == ' ' && i == len(runes)-1 && len(ids) > 0 {
			ids[len(ids)-1] = 500 + uint32(r)
			continue
		}
		ids = append(ids, 100+uint32(r))
	}
	return ids
}

func (f *fakeTokenizer) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func newClient(t *testing.T, fake *fakeTokenizer, kind backend.Kind, model string) *Client {
	t.Helper()
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)
	transport := backend.NewClient(server.URL, 5*time.Second)
	return New(transport, kind, model)
}

func TestPrepareProvesTheBoundaryAndBindsSlotsInOrder(t *testing.T) {
	fake := &fakeTokenizer{}
	client := newClient(t, fake, backend.KindVLLM, "served-model")

	prepared, err := client.Prepare(context.Background(), "row-1", "hello", 3)
	if err != nil {
		t.Fatal(err)
	}
	wantIDs := []uint32{100 + 'h', 100 + 'e', 100 + 'l', 100 + 'l', 100 + 'o'}
	if len(prepared.IDs) != len(wantIDs) {
		t.Fatalf("ids = %v, want %v", prepared.IDs, wantIDs)
	}
	for i := range wantIDs {
		if prepared.IDs[i] != wantIDs[i] {
			t.Fatalf("ids = %v, want %v", prepared.IDs, wantIDs)
		}
	}
	if len(prepared.Slots) != 3 {
		t.Fatalf("slots = %v, want 3", prepared.Slots)
	}
	for i := 0; i < 3; i++ {
		if want := uint32(100 + slots.Letter(i)[0]); prepared.Slots[i] != want {
			t.Errorf("slot %s = %d, want %d", slots.Letter(i), prepared.Slots[i], want)
		}
	}
	if prepared.MaxModelLen != 4096 {
		t.Errorf("max_model_len = %d, want 4096", prepared.MaxModelLen)
	}

	// Every request must say add_special_tokens: false, and vLLM needs the served model.
	for _, body := range fake.bodies {
		if special, ok := body["add_special_tokens"].(bool); !ok || special {
			t.Errorf("tokenize body did not ask for no special tokens: %v", body)
		}
		if body["model"] != "served-model" {
			t.Errorf("vLLM tokenize body must name the model: %v", body)
		}
	}
}

func TestPrepareIsMemoizedOnThePromptTail(t *testing.T) {
	fake := &fakeTokenizer{}
	client := newClient(t, fake, backend.KindVLLM, "m")

	if _, err := client.Prepare(context.Background(), "row-1", "hello", 2); err != nil {
		t.Fatal(err)
	}
	after := fake.callCount()
	if after != 3 { // prompt + two letters
		t.Fatalf("first prepare made %d calls, want 3", after)
	}
	// The prompt itself is always tokenized (it is what the scoring request pins), but the
	// boundary proof is reused, so a repeat costs one call instead of three.
	if _, err := client.Prepare(context.Background(), "row-2", "hello", 2); err != nil {
		t.Fatal(err)
	}
	if got := fake.callCount(); got != after+1 {
		t.Fatalf("a proven prompt cost %d extra calls, want 1 (the prompt only)", got-after)
	}
	// A different option count is a different proof.
	if _, err := client.Prepare(context.Background(), "row-3", "hello", 3); err != nil {
		t.Fatal(err)
	}
	if got := fake.callCount(); got != after+1+4 {
		t.Fatalf("a new option count must be proven again: %d calls, want %d", got, after+5)
	}
}

// A generation prompt that ends in a space is the realistic merge case: the tokenizer folds
// " A" into one token, so appending the letter does not grow the id list.
// A realistic prompt tokenizes to thousands of ids, so the response body is tens of KB. The
// transport must return it whole: a success body capped at an error-detail length parses as
// "invalid JSON" and this test is what catches that.
func TestPrepareHandlesARealisticPrompt(t *testing.T) {
	fake := &fakeTokenizer{}
	client := newClient(t, fake, backend.KindVLLM, "m")
	prompt := strings.Repeat("evidence ", 500)
	prepared, err := client.Prepare(context.Background(), "row-1", prompt, 2)
	if err != nil {
		t.Fatal(err)
	}
	if want := len([]rune(prompt)); len(prepared.IDs) != want {
		t.Fatalf("ids = %d, want %d", len(prepared.IDs), want)
	}
}

func TestMergingTokenizerIsRefused(t *testing.T) {
	fake := &fakeTokenizer{merge: true}
	client := newClient(t, fake, backend.KindVLLM, "m")
	_, err := client.Prepare(context.Background(), "row-1", "hello ", 2)
	if err == nil || !strings.Contains(err.Error(), "answer boundary changes tokenization") {
		t.Fatalf("err = %v, want the boundary refusal", err)
	}
}

func TestCollidingSlotsAreRefused(t *testing.T) {
	fake := &fakeTokenizer{collide: true}
	client := newClient(t, fake, backend.KindVLLM, "m")
	_, err := client.Prepare(context.Background(), "row-1", "hello", 2)
	if err == nil || !strings.Contains(err.Error(), "answer-slot tokens collide") {
		t.Fatalf("err = %v, want the collision refusal", err)
	}
}

func TestZeroTokenPromptIsRefused(t *testing.T) {
	fake := &fakeTokenizer{}
	client := newClient(t, fake, backend.KindVLLM, "m")
	_, err := client.Prepare(context.Background(), "row-1", "", 2)
	if err == nil || !strings.Contains(err.Error(), "tokenizes to zero tokens") {
		t.Fatalf("err = %v", err)
	}
}

func TestInconsistentCountIsRefused(t *testing.T) {
	fake := &fakeTokenizer{countMismatch: true}
	client := newClient(t, fake, backend.KindVLLM, "m")
	_, err := client.Prepare(context.Background(), "row-1", "hello", 2)
	if err == nil || !strings.Contains(err.Error(), "inconsistent tokenization") {
		t.Fatalf("err = %v", err)
	}
}

func TestMissingEndpointIsRefusedAtProbe(t *testing.T) {
	fake := &fakeTokenizer{status: http.StatusNotFound}
	client := newClient(t, fake, backend.KindVLLM, "m")
	err := client.Probe(context.Background())
	if err == nil || !strings.Contains(err.Error(), "POST /tokenize") {
		t.Fatalf("err = %v, want it to name the missing endpoint", err)
	}
}

func TestSGLangUsesItsOwnPathAndOmitsTheModel(t *testing.T) {
	fake := &fakeTokenizer{}
	client := newClient(t, fake, backend.KindSGLang, "m")
	if err := client.Probe(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, path := range fake.paths {
		if path != "/v1/tokenize" {
			t.Errorf("path = %s, want /v1/tokenize", path)
		}
	}
	for _, body := range fake.bodies {
		if _, present := body["model"]; present {
			t.Errorf("SGLang tokenize body must not carry a model: %v", body)
		}
	}
}

func TestOptionCountOutsideTheAlphabetIsRefused(t *testing.T) {
	fake := &fakeTokenizer{}
	client := newClient(t, fake, backend.KindVLLM, "m")
	if _, err := client.Prepare(context.Background(), "row-1", "hello", 1); err == nil {
		t.Error("one option is not a decision")
	}
	if _, err := client.Prepare(context.Background(), "row-1", "hello", slots.MaxValues+1); err == nil {
		t.Error("more options than letters must be refused")
	}
	if fake.callCount() != 0 {
		t.Errorf("a refused option count must not spend a tokenize call: %d", fake.callCount())
	}
}

func TestLimitReservesTheScoredPosition(t *testing.T) {
	cases := []struct {
		name          string
		local, ctxLen int
		want          int
	}{
		{"backend context wins over the local default", 0, 4096, 4095},
		{"the smaller local ceiling wins", 100, 4096, 100},
		{"no backend context leaves the local ceiling", 100, 0, 100},
		{"neither ceiling stated is unbounded", 0, 0, math.MaxInt},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Limit(tc.local, tc.ctxLen); got != tc.want {
				t.Fatalf("Limit(%d, %d) = %d, want %d", tc.local, tc.ctxLen, got, tc.want)
			}
		})
	}
}
