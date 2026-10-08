// Package logging installs the process logger.
//
// Logs go to stderr, not stdout: the service's observable contract is that only --help writes
// to stdout, and a gateway or a test harness may rely on that split. (The snippet in AGENTS.md
// passes os.Stdout; the stream is the one place this implementation deliberately differs, and
// the caller supplies the writer so a test can capture it.)
package logging

import (
	"io"
	"log/slog"
	"os"

	"github.com/lmittmann/tint"
	"github.com/sower-proxy/deferlog/v2"
)

// Setup installs the process logger and returns it. It is installed as deferlog's default so
// that deferlog.DebugError and friends (used at critical function boundaries) write to the same
// handler with the same format. The level is Info: nothing in this process has a debug-level
// story to tell yet, and a knob no caller varies is not a configuration surface.
func Setup(w io.Writer) *slog.Logger {
	logger := slog.New(tint.NewHandler(w, &tint.Options{
		Level:     slog.LevelInfo,
		AddSource: true,
		NoColor:   !isTerminal(w),
	}))
	slog.SetDefault(logger)
	deferlog.SetDefault(logger)
	return logger
}

// isTerminal reports whether w is a character device. Colour is disabled everywhere else, so
// piped output and log files stay greppable.
func isTerminal(w io.Writer) bool {
	file, ok := w.(*os.File)
	if !ok {
		return false
	}
	info, err := file.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}
