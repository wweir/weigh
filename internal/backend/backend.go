// Package backend is the HTTP transport to an inference server, plus the enums that name what
// that server is and which readout it was probed for.
//
// The transport is deliberately plain: no TLS, no retries. A retry would silently spend a
// second readout against a backend that may already have answered, and every number this
// service publishes is attributable to exactly one request.
package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// Kind is the serving dialect.
type Kind string

const (
	KindVLLM   Kind = "vllm"
	KindSGLang Kind = "sglang"
)

// Choice is the --backend value.
type Choice string

const (
	ChoiceAuto   Choice = "auto"
	ChoiceVLLM   Choice = "vllm"
	ChoiceSGLang Choice = "sglang"
)

// ParseChoice parses --backend.
func ParseChoice(text string) (Choice, error) {
	switch Choice(text) {
	case ChoiceAuto, ChoiceVLLM, ChoiceSGLang:
		return Choice(text), nil
	default:
		return "", fmt.Errorf("unknown backend %q; expected auto, vllm, or sglang", text)
	}
}

// Readout is the --readout value: how the exact-slot versus top-N trade-off is chosen.
type Readout string

const (
	ReadoutAuto      Readout = "auto"
	ReadoutExactSlot Readout = "exact-slot"
	ReadoutTopN      Readout = "top-n"
)

// ParseReadout parses --readout.
func ParseReadout(text string) (Readout, error) {
	switch Readout(text) {
	case ReadoutAuto, ReadoutExactSlot, ReadoutTopN:
		return Readout(text), nil
	default:
		return "", fmt.Errorf("unknown readout %q; expected auto, exact-slot, or top-n", text)
	}
}

// MediaSupport is what the startup probe proved about scoring an image. It is a separate axis
// from the text tier: a server can be exact-slot for text and have no media route at all.
type MediaSupport string

const (
	MediaNone          MediaSupport = "none"
	MediaChatTopN      MediaSupport = "chat-top-n"
	MediaChatExactSlot MediaSupport = "chat-exact-slot"
)

// maxDetail bounds the response body echoed in an error, so a backend that answers with a
// megabyte of HTML does not become a megabyte of log line.
const maxDetail = 500

// maxBodyBytes bounds a response this service will buffer. A tokenization of a long prompt is
// tens of KB and a completion with every declared slot is a few KB, so the ceiling is far above
// any real answer; it exists so a misbehaving backend cannot make this process allocate without
// limit. It must never be applied to the decoded value: a truncated body would fail to parse
// and be reported as "invalid JSON" rather than as a truncated answer.
const maxBodyBytes = 8 << 20

// StatusError is a non-2xx answer.
type StatusError struct {
	Code   int
	Detail string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("HTTP %d from server: %s", e.Code, e.Detail)
}

// IsTimeout reports whether a transport error spent its budget. Go exposes this properly, so
// unlike a version that has to infer it from elapsed time, the classification is exact.
func IsTimeout(err error) bool {
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	return errors.Is(err, context.DeadlineExceeded)
}

// Client talks to one backend base URL.
type Client struct {
	base string
	http *http.Client
}

// ConnectTimeout bounds connection establishment, separate from the per-request budget.
const ConnectTimeout = 15 * time.Second

// NewClient builds a client for one base URL. timeout is the whole-request budget (the
// --timeout flag) and covers connection, write and read.
func NewClient(base string, timeout time.Duration) *Client {
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   ConnectTimeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConnsPerHost: 8,
		IdleConnTimeout:     90 * time.Second,
	}
	return &Client{
		base: strings.TrimRight(strings.TrimSpace(base), "/"),
		http: &http.Client{Transport: transport, Timeout: timeout},
	}
}

// Base returns the base URL, for logs and provenance.
func (c *Client) Base() string { return c.base }

// GetJSON performs a GET and decodes a JSON response.
func (c *Client) GetJSON(ctx context.Context, path string, out any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+path, nil)
	if err != nil {
		return fmt.Errorf("Cannot reach %s%s: %w", c.base, path, err)
	}
	body, err := c.do(request, path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("%s%s returned invalid JSON: %w", c.base, path, err)
	}
	return nil
}

// PostJSON performs a POST with a JSON body and decodes a JSON response.
func (c *Client) PostJSON(ctx context.Context, path string, payload []byte, out any) error {
	body, err := c.PostRaw(ctx, path, payload)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("%s%s returned invalid JSON: %w", c.base, path, err)
	}
	return nil
}

// PostRaw performs a POST with a JSON body and returns the raw response body.
func (c *Client) PostRaw(ctx context.Context, path string, payload []byte) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+path, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("Cannot reach %s%s: %w", c.base, path, err)
	}
	request.Header.Set("Content-Type", "application/json")
	return c.do(request, path)
}

func (c *Client) do(request *http.Request, path string) ([]byte, error) {
	response, err := c.http.Do(request)
	if err != nil {
		return nil, fmt.Errorf("Cannot reach %s%s: %w", c.base, path, err)
	}
	// The body is read to the end below, so a close error carries nothing the caller can act on
	// and must not mask whatever the response said.
	defer func() { _ = response.Body.Close() }()
	// The whole body is read whatever the status: an error detail is truncated separately, and a
	// success body must arrive intact or it will not parse.
	body, err := io.ReadAll(io.LimitReader(response.Body, maxBodyBytes+1))
	if err != nil {
		return nil, fmt.Errorf("Cannot read %s%s: %w", c.base, path, err)
	}
	if len(body) > maxBodyBytes {
		return nil, fmt.Errorf("%s%s exceeded the %d byte response limit", c.base, path, maxBodyBytes)
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		detail := string(body)
		if len(detail) > maxDetail {
			detail = detail[:maxDetail]
		}
		return nil, &StatusError{Code: response.StatusCode, Detail: detail}
	}
	return body, nil
}
