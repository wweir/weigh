package serve

import (
	"context"
	"strings"

	"github.com/wweir/weigh/internal/jsonx"
)

// batchCompletion is an envelope for N independent decisions.
//
// This is not a multi-field readout and does not pretend to be one: every item runs the same
// readout a standalone request would, so validation, provenance and the one-readout discipline are
// identical. Each result carries the status that item would have had on its own, so one bad item
// cannot hide the others.
func (s *Server) batchCompletion(ctx context.Context, body []byte) reply {
	object, err := jsonx.DecodeObject(body)
	if err != nil {
		if strings.Contains(err.Error(), "must be a JSON object") {
			return errorResponse(400, "invalid_json", "body must be a JSON object")
		}
		return errorResponse(400, "invalid_json", "body is not valid JSON: "+err.Error())
	}
	requests, ok := object["requests"].([]any)
	if !ok {
		return paramError(400, "invalid_parameter", "requests", "`requests` must be an array of chat-completion bodies")
	}
	if len(requests) == 0 {
		return paramError(400, "invalid_parameter", "requests", "`requests` must not be empty")
	}
	if len(requests) > maxBatch {
		return paramError(400, "invalid_parameter", "requests",
			"`requests` has "+formatUint(uint64(len(requests)))+" items; the limit is "+formatUint(maxBatch))
	}

	results := make([]any, 0, len(requests))
	for _, item := range requests {
		// A streaming item has no meaning inside a batch: the envelope carries the items, so it
		// must not be answered as a stream. Refused here rather than spending a readout on it.
		if body, ok := item.(map[string]any); ok {
			if streaming, ok := body["stream"].(bool); ok && streaming {
				rejection := paramError(400, "unsupported_parameter", "stream",
					"streaming has no meaning inside a batch: this response carries the items, so send a streaming item as its own request")
				results = append(results, map[string]any{"status": rejection.status, "body": decodeOrText(rejection.body)})
				continue
			}
		}
		response := s.decide(ctx, []byte(marshal(item)))
		results = append(results, map[string]any{"status": response.status, "body": decodeOrText(response.body)})
	}

	return jsonReply(200, marshal(map[string]any{
		"object":  "list",
		"count":   len(results),
		"results": results,
	}))
}

// decodeOrText embeds a response body as a JSON value, falling back to the raw text when it is not
// JSON (which no path here produces, but which must not become a nil result).
func decodeOrText(body string) any {
	decoded, err := jsonx.Decode([]byte(body))
	if err != nil {
		return body
	}
	return decoded
}
