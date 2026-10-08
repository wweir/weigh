package serve

import (
	"github.com/wweir/weigh/internal/prompt"
	"github.com/wweir/weigh/internal/tokenize"
)

// clientFlavour names this implementation in every response, so a number produced by one build
// can be attributed to it.
const clientFlavour = "go-weighd serve"

func (s *Server) healthBody() string {
	urls := make([]string, 0, len(s.metadata.Endpoints))
	for _, endpoint := range s.metadata.Endpoints {
		urls = append(urls, endpoint.URL)
	}
	// An unreported ceiling is null rather than 0: a consumer must be able to tell "the backend
	// did not say" from "the backend said zero".
	var maxModelLen any
	if s.metadata.MaxModelLen > 0 {
		maxModelLen = s.metadata.MaxModelLen
	}
	return marshal(map[string]any{
		"status":                        "ok",
		"backend":                       string(s.metadata.Backend),
		"tier_a":                        s.metadata.TierA,
		"media_support":                 string(s.metadata.MediaSupport),
		"serving_config":                s.metadata.ServingConfig,
		"media_serving_config":          emptyToNil(s.metadata.MediaServingConfig),
		"prompt_version":                prompt.PromptVersion,
		"prompt_template":               string(s.metadata.PromptTemplate),
		"prompt_template_source":        s.metadata.PromptTemplateSource,
		"revision":                      s.revision,
		"tokenizer_matches_served_root": s.metadata.TokenizerMatches,
		"client":                        clientFlavour,
		// The route this deployment depends on for tokenization, in place of the crate version
		// a local tokenizer would have reported.
		"tokenize_endpoint": "POST " + tokenize.Path(s.metadata.Backend),
		"endpoints":         len(s.metadata.Endpoints),
		"endpoint_urls":     urls,
		"media": map[string]any{
			"allow_media":     s.media.AllowMedia,
			"allow_remote":    s.media.AllowRemote,
			"max_images":      s.media.MaxImages,
			"max_media_bytes": s.media.MaxMediaBytes,
		},
		"max_body_bytes": s.bodyLimit,
		"served_model":   s.servedModel,
		"max_model_len":  maxModelLen,
	})
}

func (s *Server) modelsBody() string {
	return marshal(map[string]any{
		"object": "list",
		"data": []any{map[string]any{
			"id":       s.servedModel,
			"object":   "model",
			"created":  0,
			"owned_by": "semif",
		}},
	})
}

func emptyToNil(text string) any {
	if text == "" {
		return nil
	}
	return text
}
