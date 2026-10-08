package serve

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/wweir/weigh/internal/media"
)

// Body ceilings. The floor keeps a text decision request comfortable; the slack is room for the
// non-image parts of a request (the schema, the messages, the criterion).
const (
	minBodyBytes  = 1024 * 1024
	bodyJSONSlack = 64 * 1024
)

// bodyLimit derives the request-body ceiling from the media limits, so `--max-media-bytes` can
// actually bind instead of being pre-empted by a fixed body limit.
func bodyLimit(limits media.Limits) int {
	if !limits.AllowMedia {
		return minBodyBytes
	}
	encoded := media.EncodedCeiling(limits.MaxMediaBytes)
	images := max(limits.MaxImages, 1)
	return max(minBodyBytes, encoded*images+bodyJSONSlack)
}

// readBody reads the request body under the ceiling, distinguishing "the declared length is
// already too large" from "the body turned out to be too large".
func readBody(request *http.Request, limit int) ([]byte, error) {
	if request.ContentLength > int64(limit) {
		return nil, fmt.Errorf("request body is %d bytes; the limit is %d", request.ContentLength, limit)
	}
	body, err := io.ReadAll(io.LimitReader(request.Body, int64(limit)+1))
	if err != nil {
		return nil, fmt.Errorf("cannot read request body: %w", err)
	}
	if len(body) > limit {
		return nil, fmt.Errorf("request body exceeds the %d byte limit", limit)
	}
	return body, nil
}

// rejectUnsupported refuses the parameters this endpoint cannot honour. The order is fixed: the
// first disagreement is the one reported.
func rejectUnsupported(object map[string]any, servedModel string) *reply {
	if model, ok := object["model"].(string); ok && model != servedModel {
		rejection := paramError(404, "model_not_found", "model",
			fmt.Sprintf("this server serves %q; the request asked for %q", servedModel, model))
		return &rejection
	}
	if value, present := object["n"]; present {
		if count, ok := asInt64(value); !ok || count != 1 {
			rejection := paramError(400, "unsupported_parameter", "n",
				"`n` must be the integer 1: this endpoint scores one distribution over one answer-slot set, it does not sample")
			return &rejection
		}
	}
	for _, key := range []string{"tools", "functions", "tool_choice"} {
		if _, present := object[key]; present {
			rejection := paramError(400, "unsupported_parameter", key,
				fmt.Sprintf("`%s` is not supported: this endpoint cannot call tools", key))
			return &rejection
		}
	}
	for _, key := range []string{"max_tokens", "max_completion_tokens"} {
		value, present := object[key]
		if !present {
			continue
		}
		// Any positive integer budget is satisfiable: exactly one position is returned, so 64 is
		// as good as 1. Deliberately *not* a compatibility wall — nearly every OpenAI client sends
		// `max_tokens`, and the parameter is echoed in `ignored_parameters` regardless.
		if budget, ok := asInt64(value); !ok || budget < 1 {
			rejection := paramError(400, "unsupported_parameter", key,
				fmt.Sprintf("`%s` must be a positive integer (or omitted): this endpoint always returns exactly one position", key))
			return &rejection
		}
	}
	return nil
}

func streamOptions(object map[string]any) (bool, bool, *reply) {
	streaming := false
	if value, present := object["stream"]; present {
		flag, ok := value.(bool)
		if !ok {
			rejection := paramError(400, "invalid_parameter", "stream", "`stream` must be true or false")
			return false, false, &rejection
		}
		streaming = flag
	}
	value, present := object["stream_options"]
	if !present {
		return streaming, false, nil
	}
	options, ok := value.(map[string]any)
	if !ok {
		rejection := paramError(400, "invalid_parameter", "stream_options", "`stream_options` must be an object")
		return false, false, &rejection
	}
	if !streaming {
		rejection := paramError(400, "invalid_parameter", "stream_options", "`stream_options` requires `stream: true`")
		return false, false, &rejection
	}
	if include, present := options["include_usage"]; present {
		flag, ok := include.(bool)
		if !ok {
			rejection := paramError(400, "invalid_parameter", "stream_options.include_usage", "`include_usage` must be true or false")
			return false, false, &rejection
		}
		return streaming, flag, nil
	}
	return streaming, false, nil
}

func logprobsOptions(object map[string]any) (bool, int, *reply) {
	wanted := false
	if value, present := object["logprobs"]; present {
		flag, ok := value.(bool)
		if !ok {
			rejection := paramError(400, "invalid_parameter", "logprobs", "`logprobs` must be true or false")
			return false, 0, &rejection
		}
		wanted = flag
	}
	if _, present := object["top_logprobs"]; present && !wanted {
		rejection := paramError(400, "invalid_parameter", "top_logprobs", "`top_logprobs` requires `logprobs: true`")
		return false, 0, &rejection
	}
	value, present := object["top_logprobs"]
	if !present {
		return wanted, 0, nil
	}
	count, ok := asInt64(value)
	if !ok || count < 0 || count > 20 {
		rejection := paramError(400, "invalid_parameter", "top_logprobs", "`top_logprobs` must be an integer between 0 and 20")
		return false, 0, &rejection
	}
	return wanted, int(count), nil
}

func asInt64(value any) (int64, bool) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, false
	}
	parsed, err := number.Int64()
	if err != nil {
		return 0, false
	}
	return parsed, true
}
