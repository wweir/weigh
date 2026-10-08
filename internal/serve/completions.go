package serve

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/hashx"
	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/prompt"
	"github.com/wweir/weigh/internal/readout"
	"github.com/wweir/weigh/internal/schema"
)

func (s *Server) chatCompletion(ctx context.Context, body []byte) reply {
	object, err := jsonx.DecodeObject(body)
	if err != nil {
		if strings.Contains(err.Error(), "must be a JSON object") {
			return errorResponse(400, "invalid_json", "body must be a JSON object")
		}
		return errorResponse(400, "invalid_json", "body is not valid JSON: "+err.Error())
	}

	if rejection := rejectUnsupported(object, s.servedModel); rejection != nil {
		return *rejection
	}
	streaming, includeUsage, bad := streamOptions(object)
	if bad != nil {
		return *bad
	}
	// Parsed before the readout: a bad `logprobs` is a caller error, and answering it after
	// spending a backend readout would waste that readout and count it.
	wantedLogprobs, topN, bad := logprobsOptions(object)
	if bad != nil {
		return *bad
	}
	decision, schemaErr := schema.FromRequest(object)
	if schemaErr != nil {
		return paramError(400, schemaErr.Code, schema.Param(object), schemaErr.Message)
	}
	folded, refusal := readMessages(object, s.media)
	if refusal != nil {
		return paramError(400, refusal.code, "messages", refusal.message)
	}

	var scored *readout.Scored
	if folded.hasMedia() {
		if s.metadata.MediaSupport == backend.MediaNone {
			return paramError(400, "media_unsupported", "messages",
				"this endpoint was not probed as able to score an image; see `media_support` in GET /health, and start the server with --allow-media against a multimodal model")
		}
		optionIDs := decision.Labels()
		payload, err := prompt.MediaUserPayload(folded.textEvidence(), folded.criterion, optionIDs)
		if err != nil {
			return errorResponse(400, "media_contract", err.Error())
		}
		rowID := "serve-" + requestID(folded.criterion, folded.evidenceIdentity(), decision)
		scored, err = s.client.DecideMedia(ctx, rowID, prompt.DirectSystem, payload, folded.media(), optionIDs)
		if err != nil {
			return readoutFailure(s, err, "media_contract")
		}
	} else {
		evidence := folded.textEvidence()
		optionIDs := decision.Labels()
		options := make([]any, 0, len(optionIDs))
		for _, label := range optionIDs {
			options = append(options, map[string]any{"id": label, "description": label})
		}
		rowValue := map[string]any{
			"id":       "serve-" + requestID(folded.criterion, evidence, decision),
			"state":    evidence,
			"question": folded.criterion,
			"options":  options,
		}
		row, err := prompt.ValidateRow(rowValue)
		if err != nil {
			return errorResponse(400, "row_contract", err.Error())
		}
		// The prompt is rendered here rather than inside the readout: `direct-options-v1` is this
		// service's contract, and the readout deliberately takes a finished prompt string.
		promptText, err := prompt.RenderPromptWith(s.metadata.PromptTemplate, row)
		if err != nil {
			return errorResponse(400, "prompt_contract", err.Error())
		}
		scored, err = s.client.Decide(ctx, row.ID, promptText, optionIDs)
		if err != nil {
			return readoutFailure(s, err, "prompt_contract")
		}
	}

	s.metrics.observeReadout(scored.TotalSeconds, scored.FallbackUsed)
	value, bad := s.completionValue(object, decision, scored, folded.renderer, wantedLogprobs, topN)
	if bad != nil {
		return *bad
	}
	// A streaming request gets the same decision, emitted as the one-delta stream an OpenAI client
	// waits for. Nothing about the readout is incremental, so this is a formatting choice rather
	// than a second code path.
	if streaming {
		return reply{status: 200, contentType: sseType, body: sseBody(value, includeUsage)}
	}
	return jsonReply(200, marshal(value))
}

// readoutFailure maps a readout error: a spent budget is a 504, a contract failure is a 400 under
// the code the caller passed, and anything else is a 502.
func readoutFailure(s *Server, err error, contractCode string) reply {
	// Go exposes timeouts properly, so the classification is exact rather than inferred from
	// elapsed time.
	var contract *readout.ContractError
	if errors.As(err, &contract) {
		return errorResponse(400, contractCode, err.Error())
	}
	if backend.IsTimeout(err) {
		s.metrics.backendTimeouts.Add(1)
		return errorResponse(504, "backend_timeout", err.Error())
	}
	s.metrics.backendErrors.Add(1)
	return errorResponse(502, "backend_error", err.Error())
}

// requestID identifies one decision. The schema is part of the identity, not just the messages:
// the same evidence with a different option set is a different decision.
func requestID(criterion, evidence string, decision *schema.Decision) string {
	var preimage strings.Builder
	preimage.WriteString(criterion)
	preimage.WriteByte(0)
	preimage.WriteString(evidence)
	preimage.WriteByte(0)
	preimage.WriteString(decision.Field)
	for _, value := range decision.Values {
		preimage.WriteByte(0)
		preimage.WriteString(jsonRendering(value))
	}
	// The first 12 bytes of the digest, hex-encoded: 24 characters.
	return hashx.Sha256Hex([]byte(preimage.String()))[:24]
}

// jsonRendering renders a schema value the way its JSON form reads, so a string value and a
// number that render alike cannot collide in an id.
func jsonRendering(value any) string {
	rendered, err := jsonx.Compact(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return rendered
}
