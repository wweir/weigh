// Command weighd serves the decision API.
//
// It owns argv, configuration, logging and the process lifecycle; the HTTP surface lives in
// internal/serve. All startup decisions that can refuse (backend dialect, readout tier,
// checkpoint identity, chat template) happen inside Run, before the listener binds.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/wweir/weigh/config"
	"github.com/wweir/weigh/internal/logging"
	"github.com/wweir/weigh/internal/serve"
	"github.com/wweir/weigh/internal/version"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "weighd: error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// Load installs the usage text and parses flags; --help is reported as ErrHelp so the help
	// text can go to stdout while every refusal goes to stderr.
	cfg, err := config.Load()
	if errors.Is(err, config.ErrHelp) {
		config.WriteUsage(os.Stdout)
		return nil
	}
	if err != nil {
		return err
	}

	logger := logging.Setup(os.Stderr)
	logger.Info("weighd starting", "version", version.String())

	// Cancellation is the shutdown signal: Run returns once every in-flight decision has
	// finished, and the deferred stop restores default signal handling.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	return serve.Run(ctx, cfg, logger)
}
