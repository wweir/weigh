// Package config defines weighd's configuration, its command-line flags and its validation.
//
// The precedence is the conventional one: an explicitly set flag beats a value from the
// configuration file, which beats the built-in default. feconf owns the file half (it also
// registers the file's fields as flags) and this package owns the defaults and the refusals.
//
// Every default here is the value the Rust implementation documents in --help. A flag whose
// zero value is meaningful (--timeout 0, --workers 0) is distinguished from an omitted flag by
// asking the flag package which flags were actually set, so an explicit bad value is refused
// instead of being silently read as "use the default".
package config

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/sower-proxy/feconf"

	"github.com/wweir/weigh/internal/backend"
	"github.com/wweir/weigh/internal/template"

	// The file reader and the TOML decoder are registered by import side effects.
	_ "github.com/sower-proxy/feconf/decoder/toml"
	_ "github.com/sower-proxy/feconf/reader/file"
)

// Defaults and limits. These mirror the Rust implementation's constants.
const (
	DefaultHost          = "127.0.0.1"
	DefaultPort          = 8080
	DefaultTimeout       = 120
	DefaultMaxImages     = 1
	DefaultMaxMediaBytes = 4 << 20
	MaxMediaBytesCeiling = 64 << 20
	DefaultBackend       = "auto"
	DefaultReadout       = "exact-slot"
	DefaultTemplate      = "auto"
	DefaultRevision      = "unknown"
	DefaultURL           = "http://localhost:8000"
)

// Config is the whole runtime configuration.
//
// A field is exposed as a flag when it carries a `usage` tag, and the flag name is the `json`
// tag. The configuration file uses the same spelling, deliberately: feconf decodes the file and
// the flags into one key space and matches a key against that tag, so a flag spelled differently
// from the file key is silently dropped — `--api-key` included, whose absence is an open server.
// URLs is the one field with no `usage` tag here, because [Load] registers it as a repeatable
// `--url`.
type Config struct {
	// setFlags records which flags the operator set explicitly. Unexported so no decoder
	// carries it; used only to tell "omitted" from "set to the zero value".
	setFlags map[string]bool

	Model    string   `json:"model" usage:"local checkpoint dir; tokenizer_config.json drives --template auto"`
	URLs     []string `json:"urls"`
	Backend  string   `json:"backend" usage:"auto, vllm or sglang; auto probes each endpoint"`
	Readout  string   `json:"readout" usage:"exact-slot or top-n; top-n exists for vLLM < v0.26.0"`
	Template string   `json:"template" usage:"auto, gemma4, gemma3, chatml, chatml-thinking, llama3, llama2 or mistral"`

	Host string `json:"host" usage:"bind address"`
	Port int    `json:"port" usage:"bind port"`

	Workers         int `json:"workers" usage:"simultaneous decisions; default 4x CPUs, clamped to [8, 64]"`
	Timeout         int `json:"timeout" usage:"per backend request, in seconds"`
	MaxPromptTokens int `json:"max-prompt-tokens" usage:"prompt token ceiling; default: the server's max_model_len - 1"`

	Revision               string `json:"revision" usage:"recorded in every response"`
	AllowTokenizerMismatch bool   `json:"allow-tokenizer-mismatch" usage:"relax the checkpoint-name guard"`
	APIKey                 string `json:"api-key" usage:"require Authorization: Bearer KEY on every route except GET /health"`
	CorsOrigin             string `json:"cors-origin" usage:"add CORS headers: * for any origin, or one exact origin"`

	AllowMedia       bool `json:"allow-media" usage:"accept image_url parts; needs a multimodal model"`
	AllowMediaTopN   bool `json:"allow-media-topn" usage:"also accept the best-effort media route"`
	AllowRemoteMedia bool `json:"allow-remote-media" usage:"accept http(s) image URLs"`
	MaxImages        int  `json:"max-images" usage:"images per request"`
	MaxMediaBytes    int  `json:"max-media-bytes" usage:"decoded bytes per image"`
}

// ErrHelp is returned by [Load] when the operator asked for --help. main prints the usage text
// to stdout and exits 0, which is the contract the published CLI documents: only --help writes
// to stdout, everything else (including every refusal) writes to stderr.
var ErrHelp = errors.New("help requested")

// Load parses the command line and the configuration file and returns a validated Config.
//
// It installs the usage text and calls flag.Parse (through feconf), so it must be called once,
// from main, before anything else reads flags.
func Load() (*Config, error) {
	for _, arg := range os.Args[1:] {
		switch arg {
		case "-h", "-help", "--help":
			return nil, ErrHelp
		}
	}

	flag.CommandLine.Usage = Usage

	var urls urlsFlag
	flag.Var(&urls, "url", "backend base URL; repeat or comma-separate for a fleet")

	// feconf picks the first candidate a reader can actually open, so this list is a real
	// search order rather than a precedence rule. With no file present it decodes the flags
	// alone, which is why every default also lives in ApplyDefaults.
	loader := feconf.New[Config]("config",
		"file://weighd.toml",
		"file://config/weighd.toml",
		"file:///etc/weighd/weighd.toml",
	)
	cfg, err := loader.Parse()
	if err != nil {
		return nil, fmt.Errorf("load configuration: %w", err)
	}

	cfg.setFlags = explicitFlags()
	// --url is registered here rather than by feconf (it is the only repeatable flag), so its
	// value overrides the file only when it was actually given.
	if len(urls) > 0 {
		cfg.URLs = urls
	}

	cfg.ApplyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// ApplyDefaults fills in every value the operator left unset. It runs before Validate so that
// Validate only ever sees a fully specified configuration.
func (c *Config) ApplyDefaults() {
	if len(c.URLs) == 0 {
		c.URLs = []string{DefaultURL}
	}
	if c.Backend == "" {
		c.Backend = DefaultBackend
	}
	if c.Readout == "" {
		c.Readout = DefaultReadout
	}
	if c.Template == "" {
		c.Template = DefaultTemplate
	}
	if c.Host == "" {
		c.Host = DefaultHost
	}
	if c.Port == 0 {
		c.Port = DefaultPort
	}
	if c.Revision == "" {
		c.Revision = DefaultRevision
	}
	// 0 is a legal explicit value for these two (--workers 0 means one worker; --timeout 0 is
	// refused by Validate), so only an omitted flag takes the default.
	if c.Workers == 0 && !c.wasSet("workers") {
		c.Workers = defaultWorkers()
	}
	if c.Timeout == 0 && !c.wasSet("timeout") {
		c.Timeout = DefaultTimeout
	}
	if c.MaxImages == 0 && !c.wasSet("max-images") {
		c.MaxImages = DefaultMaxImages
	}
	if c.MaxMediaBytes == 0 && !c.wasSet("max-media-bytes") {
		c.MaxMediaBytes = DefaultMaxMediaBytes
	}
}

// Validate refuses an unusable configuration. The messages are the ones the Rust
// implementation prints, because operators and scripts match on them.
func (c *Config) Validate() error {
	if c.Model == "" {
		return fmt.Errorf("--model DIR is required")
	}
	if len(c.URLs) == 0 {
		return fmt.Errorf("at least one --url is required")
	}
	// The allowed values live with the types they name, so there is one list per concept rather
	// than a copy here and another in the package that consumes it.
	if _, err := backend.ParseChoice(c.Backend); err != nil {
		return err
	}
	readout, err := backend.ParseReadout(c.Readout)
	if err != nil {
		return err
	}
	if readout == backend.ReadoutAuto {
		return fmt.Errorf("--readout auto has no meaning for a service: it picks a different tier per backend. Pass exact-slot (the default) or top-n")
	}
	if _, err := template.ParseChoice(c.Template); err != nil {
		return err
	}
	if c.Timeout < 1 {
		return fmt.Errorf("--timeout must be at least 1 second: 0 would time out every request")
	}
	if c.MaxPromptTokens < 1 && c.wasSet("max-prompt-tokens") {
		return fmt.Errorf("--max-prompt-tokens must be at least 1")
	}
	if c.MaxImages < 1 {
		return fmt.Errorf("--max-images must be at least 1 (omit the flag for the default)")
	}
	if c.MaxMediaBytes < 1 || c.MaxMediaBytes > MaxMediaBytesCeiling {
		return fmt.Errorf("--max-media-bytes must be between 1 and %d bytes", MaxMediaBytesCeiling)
	}
	if c.AllowMediaTopN && !c.AllowMedia {
		return fmt.Errorf("--allow-media-topn requires --allow-media: it widens a permission rather than granting one")
	}
	if c.APIKey == "" && c.wasSet("api-key") {
		return fmt.Errorf("--api-key must not be empty (omit the flag for an open server)")
	}
	if c.CorsOrigin == "" && c.wasSet("cors-origin") {
		return fmt.Errorf("--cors-origin must not be empty (use `*` to allow any origin)")
	}
	return nil
}

// WorkersCount is the resolved concurrency ceiling, never zero.
func (c *Config) WorkersCount() int {
	if c.Workers < 1 {
		return 1
	}
	return c.Workers
}

func (c *Config) wasSet(name string) bool {
	return c.setFlags[name]
}

// defaultWorkers mirrors the Rust default: over-provisioned because each handler spends almost
// all of its time blocked on the backend, and clamped so a misconfigured deployment cannot open
// hundreds of connections.
func defaultWorkers() int {
	cpus := runtime.NumCPU()
	if cpus < 1 {
		cpus = 4
	}
	return min(max(cpus*4, 8), 64)
}

// explicitFlags reads back which flags the flag package saw on the command line. It must run
// after flag.Parse, which feconf performs.
func explicitFlags() map[string]bool {
	set := make(map[string]bool)
	flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
	return set
}

// urlsFlag is the repeatable --url flag. A single value may also carry a comma-separated list,
// which is how the fleet form is normally written.
type urlsFlag []string

func (u *urlsFlag) String() string { return strings.Join(*u, ",") }

func (u *urlsFlag) Set(value string) error {
	for _, part := range strings.Split(value, ",") {
		if part = strings.TrimSpace(part); part != "" {
			*u = append(*u, part)
		}
	}
	return nil
}
