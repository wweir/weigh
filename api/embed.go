// Package api embeds the OpenAPI description of this service.
//
// A file so the document is diffable, embedded at compile time so it cannot go missing at run
// time.
package api

import _ "embed"

// OpenAPI is the machine-readable contract served at GET /openapi.json.
//
//go:embed openapi.json
var OpenAPI string
