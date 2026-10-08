// Package serve owns weighd's HTTP contract: routing, request validation, the error envelope,
// SSE, /health, the drain and the metrics.
//
// The readout itself lives in internal/readout, the decision contract in internal/schema,
// internal/prompt and internal/template, and the media parts in internal/media; this package is
// the transport and the validation order, and nothing else.
package serve

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/wweir/weigh/config"
	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/media"
	"github.com/wweir/weigh/internal/prompt"
	"github.com/wweir/weigh/internal/readout"
	"github.com/wweir/weigh/internal/template"
)

// Run binds the listener and serves until ctx is cancelled, then drains in-flight decisions
// before returning.
//
// Everything that can refuse happens before the listener binds: an endpoint that cannot be
// identified, a fleet whose members disagree, a checkpoint mismatch, a backend that does not prove
// the requested readout, an unknown prompt ceiling, and a missing tokenize route. newServer owns
// all of those, so this function does not repeat them.
func Run(ctx context.Context, cfg *config.Config, logger *slog.Logger) error {
	server, err := newServer(ctx, cfg, logger)
	if err != nil {
		return err
	}
	metadata, workers := server.metadata, cap(server.concurrency)

	address := net.JoinHostPort(cfg.Host, strconv.Itoa(cfg.Port))
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("cannot listen on %s: %w", address, err)
	}
	logger.Info("listening",
		"url", "http://"+listener.Addr().String(),
		"backend", string(metadata.Backend),
		"model", server.servedModel,
		"revision", cfg.Revision,
		"prompt_version", prompt.PromptVersion,
		"prompt_template", string(metadata.PromptTemplate),
		"prompt_template_source", metadata.PromptTemplateSource,
		"serving_config", metadata.ServingConfig,
		"media_support", string(metadata.MediaSupport),
		"workers", workers)

	httpServer := &http.Server{
		Handler:           server,
		ReadHeaderTimeout: 30 * time.Second,
	}
	served := make(chan error, 1)
	go func() { served <- httpServer.Serve(listener) }()

	select {
	case err := <-served:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("listener stopped: %w", err)
		}
		return nil
	case <-ctx.Done():
	}

	// A supervisor's SIGTERM must not drop an in-flight decision: stop accepting, let every
	// handler that is already running finish, then close idle connections. The budget is the
	// per-request timeout plus room to finish, matching the systemd unit's TimeoutStopSec.
	drainContext, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.Timeout+10)*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(drainContext); err != nil {
		return fmt.Errorf("drain did not complete: %w", err)
	}
	logger.Info("drained in-flight requests and exiting on a shutdown signal")
	if err := <-served; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("listener stopped: %w", err)
	}
	return nil
}

// newServer resolves the configuration into a probed, ready server. Every refusal a deployment can
// hit happens here, so Run and the tests exercise the same startup path.
func newServer(ctx context.Context, cfg *config.Config, logger *slog.Logger) (*Server, error) {
	backendChoice, err := backend.ParseChoice(cfg.Backend)
	if err != nil {
		return nil, err
	}
	readoutChoice, err := backend.ParseReadout(cfg.Readout)
	if err != nil {
		return nil, err
	}
	templateChoice, err := template.ParseChoice(cfg.Template)
	if err != nil {
		return nil, err
	}
	// `--allow-media-topn` widens the media permission; `--allow-media` alone asks for the
	// exact-slot media route specifically.
	policy := readout.MediaOff
	switch {
	case cfg.AllowMediaTopN:
		policy = readout.MediaTopN
	case cfg.AllowMedia:
		policy = readout.MediaExactSlot
	}
	limits := media.Limits{
		AllowMedia:    cfg.AllowMedia,
		AllowRemote:   cfg.AllowRemoteMedia,
		MaxImages:     cfg.MaxImages,
		MaxMediaBytes: cfg.MaxMediaBytes,
	}

	client, metadata, err := readout.New(ctx, readout.Config{
		URLs:                   cfg.URLs,
		Backend:                backendChoice,
		Readout:                readoutChoice,
		Template:               templateChoice,
		Timeout:                time.Duration(cfg.Timeout) * time.Second,
		MaxTokens:              cfg.MaxPromptTokens,
		AllowTokenizerMismatch: cfg.AllowTokenizerMismatch,
		Media:                  policy,
		Source:                 cfg.Model,
	})
	if err != nil {
		return nil, err
	}
	if readoutChoice == backend.ReadoutTopN {
		logger.Warn("--readout top-n selected; a row whose answer slot misses the top-20 is read through /generative_scoring, and that fallback rate rises with load. Each response records which path it used in choices[].semif.fallback_used")
	}
	if len(metadata.Endpoints) == 0 || len(metadata.Endpoints[0].ServedModels) == 0 {
		return nil, errors.New("no served model reported by the backend")
	}

	workers := cfg.WorkersCount()
	return &Server{
		client:      client,
		metadata:    metadata,
		servedModel: metadata.Endpoints[0].ServedModels[0],
		revision:    cfg.Revision,
		apiKey:      cfg.APIKey,
		corsOrigin:  cfg.CorsOrigin,
		media:       limits,
		bodyLimit:   bodyLimit(limits),
		metrics:     &metrics{},
		concurrency: make(chan struct{}, workers),
	}, nil
}
