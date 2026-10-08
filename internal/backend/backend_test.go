package backend

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// A success body is returned whole. The transport previously capped every response at the error
// detail length, which made any real body — a tokenization of a long prompt is tens of KB —
// fail to parse as "invalid JSON".
func TestSuccessBodyIsReturnedWhole(t *testing.T) {
	// 8 KB of JSON: far past any error-detail cap, well under the response ceiling.
	models := make([]string, 400)
	for i := range models {
		models[i] = `{"id":"model-` + strings.Repeat("x", 8) + `"}`
	}
	body := `{"data":[` + strings.Join(models, ",") + `]}`
	if len(body) < 4096 {
		t.Fatalf("fixture is only %d bytes; it must exceed any detail cap", len(body))
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)

	var decoded struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := NewClient(server.URL, 5*time.Second).GetJSON(context.Background(), "/v1/models", &decoded); err != nil {
		t.Fatalf("a %d byte body must decode: %v", len(body), err)
	}
	if len(decoded.Data) != 400 {
		t.Fatalf("decoded %d entries, want 400", len(decoded.Data))
	}
}

// A large body round-trips through PostJSON too: that is the path every scoring request takes.
func TestPostJSONReturnsALargeBody(t *testing.T) {
	slots := make([]any, 16)
	for i := range slots {
		slots[i] = map[string]any{"token_id": i, "logprob": -0.5}
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"top": slots}}})
	}))
	t.Cleanup(server.Close)

	var response map[string]any
	if err := NewClient(server.URL, 5*time.Second).PostJSON(context.Background(), "/v1/completions", []byte(`{}`), &response); err != nil {
		t.Fatalf("a large completion response must decode: %v", err)
	}
	if _, ok := response["choices"]; !ok {
		t.Fatalf("response = %v", response)
	}
}

// An error body is truncated for the message, and the status is preserved so callers can tell a
// real negative answer (400/422) from a failure.
func TestStatusErrorCarriesCodeAndTruncatedDetail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(strings.Repeat("x", maxDetail+200)))
	}))
	t.Cleanup(server.Close)

	var out map[string]any
	err := NewClient(server.URL, 5*time.Second).PostJSON(context.Background(), "/v1/completions", []byte(`{}`), &out)
	var status *StatusError
	if !errors.As(err, &status) {
		t.Fatalf("err = %v, want a *StatusError", err)
	}
	if status.Code != http.StatusUnprocessableEntity {
		t.Errorf("code = %d", status.Code)
	}
	if len(status.Detail) != maxDetail {
		t.Errorf("detail length = %d, want %d", len(status.Detail), maxDetail)
	}
}

// A response past the ceiling is refused rather than buffered without limit.
func TestResponsePastTheCeilingIsRefused(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("x", maxBodyBytes+1)))
	}))
	t.Cleanup(server.Close)

	var out map[string]any
	err := NewClient(server.URL, 10*time.Second).GetJSON(context.Background(), "/big", &out)
	if err == nil || !strings.Contains(err.Error(), "response limit") {
		t.Fatalf("err = %v, want the ceiling refusal", err)
	}
}
