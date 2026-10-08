package serve

import (
	"encoding/json"
	"strings"

	"github.com/wweir/weigh/internal/jsonx"
)

// Content types. Every response carries one, including a 204 and every error body.
const (
	jsonType      = "application/json"
	sseType       = "text/event-stream"
	metricsType   = "text/plain; version=0.0.4; charset=utf-8"
	internalError = "internal_error"
)

// reply is one HTTP response: status, body and the content type it is served under.
type reply struct {
	status      int
	contentType string
	body        string
}

// jsonReply wraps a JSON body.
func jsonReply(status int, body string) reply {
	return reply{status: status, contentType: jsonType, body: body}
}

// errorResponse is the envelope without a `param`: {"error":{message,type,code}}.
func errorResponse(status int, code, message string) reply {
	body, err := jsonx.Compact(jsonx.Object{{Key: "error", Value: jsonx.Object{
		{Key: "message", Value: message},
		{Key: "type", Value: errorType(status)},
		{Key: "code", Value: code},
	}}})
	if err != nil {
		// Unreachable: the envelope holds only strings this process built.
		panic("serve: error envelope: " + err.Error())
	}
	return jsonReply(status, body)
}

// paramError is the envelope naming the request field at fault.
func paramError(status int, code, param, message string) reply {
	body, err := jsonx.Compact(jsonx.Object{{Key: "error", Value: jsonx.Object{
		{Key: "message", Value: message},
		{Key: "type", Value: errorType(status)},
		{Key: "code", Value: code},
		{Key: "param", Value: param},
	}}})
	if err != nil {
		panic("serve: error envelope: " + err.Error())
	}
	return jsonReply(status, body)
}

func errorType(status int) string {
	if status >= 500 {
		return "server_error"
	}
	return "invalid_request_error"
}

// marshal renders a response body without HTML escaping, so a message containing `<` is served
// as written rather than as \u003c.
//
// The escaping is done by the encoder in one pass, not by undoing encoding/json's HTML escaping
// afterwards: a caller-controlled string may itself contain a literal `\u003c`, which the
// encoder writes as `\\u003c`, and a substring replacement would rewrite that tail back into the
// invalid JSON escape `\<`.
func marshal(value any) string {
	var out strings.Builder
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		// Unreachable: every value on this path is built from JSON-decoded data, floats this
		// process computed, and strings.
		panic("serve: marshal response: " + err.Error())
	}
	// Encode appends a newline no response body carries.
	return strings.TrimSuffix(out.String(), "\n")
}
