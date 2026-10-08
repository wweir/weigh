package serve

import (
	"context"
	"crypto/subtle"
	"fmt"
	"net/http"
	"strings"

	"github.com/wweir/weigh/api"
	"github.com/wweir/weigh/internal/media"
	"github.com/wweir/weigh/internal/readout"
)

// Server is the HTTP surface of one probed deployment.
type Server struct {
	client      *readout.Client
	metadata    *readout.Metadata
	servedModel string
	revision    string
	apiKey      string
	corsOrigin  string
	media       media.Limits
	bodyLimit   int
	metrics     *metrics
	// concurrency is the operator's --workers as a ceiling on simultaneous decisions. The readout
	// is almost entirely blocked on the backend, so this bounds backend fan-out rather than CPU.
	concurrency chan struct{}
}

// ServeHTTP is the whole route table: preflight, then CORS, then auth, then the route.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.metrics.requests.Add(1)
	s.metrics.inflight.Add(1)
	defer s.metrics.inflight.Add(-1)

	path := r.URL.Path
	origin := r.Header.Get("Origin")
	var response reply
	switch {
	case r.Method == http.MethodOptions:
		// Preflight is answered before the bearer check on purpose: a browser does not attach the
		// token to it.
		response = reply{status: 204, contentType: jsonType}
	case !s.originAllowed(origin):
		response = errorResponse(403, "origin_not_allowed",
			fmt.Sprintf("the browser origin %q is not allowed by this server", origin))
	case path != "/health" && !s.authorized(r.Header.Get("Authorization")):
		// /health stays open so a supervisor can probe liveness without the key.
		response = errorResponse(401, "invalid_api_key", "this server requires `Authorization: Bearer <key>`")
	default:
		response = s.route(r)
	}

	s.metrics.finished.Add(1)
	switch response.status / 100 {
	case 5:
		s.metrics.serverErrors.Add(1)
	case 4:
		s.metrics.clientErrors.Add(1)
	}

	for name, value := range s.corsHeaders(origin, r.Method) {
		w.Header().Set(name, value)
	}
	// Every response carries a content type, including a 204 and every error body.
	w.Header().Set("Content-Type", response.contentType)
	w.WriteHeader(response.status)
	if response.body != "" {
		_, _ = w.Write([]byte(response.body))
	}
}

func (s *Server) route(r *http.Request) reply {
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/health":
		return jsonReply(200, s.healthBody())
	case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
		return jsonReply(200, s.modelsBody())
	case r.Method == http.MethodGet && r.URL.Path == "/openapi.json":
		return jsonReply(200, api.OpenAPI)
	case r.Method == http.MethodGet && r.URL.Path == "/metrics":
		return reply{status: 200, contentType: metricsType, body: s.metrics.Render()}
	case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
		body, err := readBody(r, s.bodyLimit)
		if err != nil {
			return errorResponse(413, "payload_too_large", err.Error())
		}
		return s.decide(r.Context(), body)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/semif/batch":
		body, err := readBody(r, s.bodyLimit)
		if err != nil {
			return errorResponse(413, "payload_too_large", err.Error())
		}
		return s.batchCompletion(r.Context(), body)
	case r.Method == http.MethodGet || r.Method == http.MethodPost:
		return errorResponse(404, "not_found", "no route for "+r.URL.Path)
	default:
		return errorResponse(405, "method_not_allowed", r.Method+" is not allowed on "+r.URL.Path)
	}
}

// decide holds one concurrency slot for the whole decision, so the backend never sees more
// simultaneous requests than the operator configured. A batch item goes through here too, which
// is what bounds a batch by the same ceiling; the channel is always set by newServer.
func (s *Server) decide(ctx context.Context, body []byte) reply {
	select {
	case s.concurrency <- struct{}{}:
		defer func() { <-s.concurrency }()
		return s.chatCompletion(ctx, body)
	case <-ctx.Done():
		// The caller is gone; spending a backend readout for them would be waste.
		return errorResponse(503, "client_gone", "the request was cancelled before this decision started")
	}
}

// authorized is the bearer check. No key configured means the server is open, which is the default
// and is only appropriate on a trusted network.
func (s *Server) authorized(header string) bool {
	if s.apiKey == "" {
		return true
	}
	presented, found := strings.CutPrefix(header, "Bearer ")
	if !found {
		return false
	}
	// Constant time: a wrong key must not be recoverable byte by byte from response timing.
	return subtle.ConstantTimeCompare([]byte(s.apiKey), []byte(strings.TrimSpace(presented))) == 1
}

// originAllowed refuses a browser origin the operator did not allow, before any work is done. A
// request with no Origin header is not a browser request and is not subject to CORS.
func (s *Server) originAllowed(origin string) bool {
	if s.corsOrigin == "" || origin == "" {
		return true
	}
	return s.corsOrigin == "*" || s.corsOrigin == origin
}

func (s *Server) corsHeaders(origin, method string) map[string]string {
	if s.corsOrigin == "" {
		return nil
	}
	if s.corsOrigin != "*" && origin != s.corsOrigin {
		return nil
	}
	headers := map[string]string{"Access-Control-Allow-Origin": s.corsOrigin}
	if method == http.MethodOptions {
		headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
		headers["Access-Control-Allow-Headers"] = "authorization, content-type"
		headers["Access-Control-Max-Age"] = "600"
	}
	return headers
}
