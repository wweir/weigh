package config

import (
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// base is a configuration that passes Validate, so a test can change one field and expect one
// failure rather than a pile of them.
func base() *Config {
	cfg := &Config{Model: "/models/test", setFlags: map[string]bool{}}
	cfg.ApplyDefaults()
	return cfg
}

func TestApplyDefaultsFillsEveryOmittedValue(t *testing.T) {
	cfg := &Config{Model: "/models/test", setFlags: map[string]bool{}}
	cfg.ApplyDefaults()

	if got := cfg.URLs; len(got) != 1 || got[0] != DefaultURL {
		t.Errorf("urls = %v, want [%s]", got, DefaultURL)
	}
	if cfg.Backend != DefaultBackend || cfg.Readout != DefaultReadout || cfg.Template != DefaultTemplate {
		t.Errorf("backend/readout/template = %q/%q/%q, want %q/%q/%q",
			cfg.Backend, cfg.Readout, cfg.Template, DefaultBackend, DefaultReadout, DefaultTemplate)
	}
	if cfg.Host != DefaultHost || cfg.Port != DefaultPort {
		t.Errorf("host/port = %q/%d, want %q/%d", cfg.Host, cfg.Port, DefaultHost, DefaultPort)
	}
	if cfg.Timeout != DefaultTimeout {
		t.Errorf("timeout = %d, want %d", cfg.Timeout, DefaultTimeout)
	}
	if cfg.MaxImages != DefaultMaxImages || cfg.MaxMediaBytes != DefaultMaxMediaBytes {
		t.Errorf("media limits = %d/%d, want %d/%d",
			cfg.MaxImages, cfg.MaxMediaBytes, DefaultMaxImages, DefaultMaxMediaBytes)
	}
	// The derived worker count is over-provisioned and clamped.
	if cfg.Workers < 8 || cfg.Workers > 64 {
		t.Errorf("workers = %d, want within [8, 64]", cfg.Workers)
	}
	if cfg.Revision != DefaultRevision {
		t.Errorf("revision = %q, want %q", cfg.Revision, DefaultRevision)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("a defaulted config must validate: %v", err)
	}
}

// The zero value of these flags is meaningful, so an explicit 0 must not be read as "omitted":
// that is the difference between refusing `--timeout 0` and quietly serving with the default.
func TestExplicitZeroIsNotOmitted(t *testing.T) {
	cases := []struct {
		name   string
		flag   string
		mutate func(*Config)
		want   string
	}{
		{"timeout", "timeout", func(c *Config) {}, "at least 1 second"},
		{"max-images", "max-images", func(c *Config) {}, "--max-images must be at least 1"},
		{"max-media-bytes", "max-media-bytes", func(c *Config) {}, "--max-media-bytes must be between"},
		{"max-prompt-tokens", "max-prompt-tokens", func(c *Config) {}, "--max-prompt-tokens must be at least 1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{Model: "/models/test", setFlags: map[string]bool{tc.flag: true}}
			tc.mutate(cfg)
			cfg.ApplyDefaults()
			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

// --workers 0 is legal (it means one worker); only an omitted flag takes the derived default.
func TestWorkersZeroMeansOne(t *testing.T) {
	cfg := &Config{Model: "/models/test", setFlags: map[string]bool{"workers": true}}
	cfg.ApplyDefaults()
	if cfg.Workers != 0 {
		t.Fatalf("workers was rewritten to %d; explicit 0 must survive ApplyDefaults", cfg.Workers)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("--workers 0 is legal: %v", err)
	}
	if got := cfg.WorkersCount(); got != 1 {
		t.Fatalf("WorkersCount() = %d, want 1", got)
	}
}

func TestValidateRefusals(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
		want   string
	}{
		{"no model", func(c *Config) { c.Model = "" }, "--model DIR is required"},
		{"unknown backend", func(c *Config) { c.Backend = "tgi" }, `unknown backend "tgi"`},
		{"readout auto", func(c *Config) { c.Readout = "auto" }, "--readout auto has no meaning"},
		{"unknown readout", func(c *Config) { c.Readout = "best" }, `unknown readout "best"`},
		{"unknown template", func(c *Config) { c.Template = "gpt" }, `unknown template "gpt"`},
		{"topn without media", func(c *Config) { c.AllowMediaTopN = true }, "--allow-media-topn requires --allow-media"},
		{"media bytes ceiling", func(c *Config) { c.MaxMediaBytes = MaxMediaBytesCeiling + 1 }, "--max-media-bytes must be between"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base()
			tc.mutate(cfg)
			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

// An empty --api-key is refused, but an absent one is an open server; the two are only
// distinguishable through the set-flag record.
func TestEmptyAPIKeyIsRefusedOnlyWhenSet(t *testing.T) {
	open := base()
	if err := open.Validate(); err != nil {
		t.Fatalf("an absent --api-key is an open server: %v", err)
	}
	empty := base()
	empty.setFlags["api-key"] = true
	if err := empty.Validate(); err == nil || !strings.Contains(err.Error(), "--api-key must not be empty") {
		t.Fatalf("err = %v, want the empty --api-key refusal", err)
	}
}

func TestURLsFlagSplitsTrimsAndDropsEmpties(t *testing.T) {
	var urls urlsFlag
	for _, value := range []string{" http://a:1 , http://b:2 ", "", "http://c:3"} {
		if err := urls.Set(value); err != nil {
			t.Fatalf("Set(%q): %v", value, err)
		}
	}
	want := []string{"http://a:1", "http://b:2", "http://c:3"}
	if len(urls) != len(want) {
		t.Fatalf("urls = %v, want %v", urls, want)
	}
	for i := range want {
		if urls[i] != want[i] {
			t.Fatalf("urls = %v, want %v", urls, want)
		}
	}
}

func TestWorkersCountClamp(t *testing.T) {
	for _, workers := range []int{0, -3} {
		cfg := &Config{Workers: workers}
		if got := cfg.WorkersCount(); got != 1 {
			t.Errorf("WorkersCount() = %d for %d, want 1", got, workers)
		}
	}
}

// Load is the only place flags become configuration, and flag registration is process-global, so
// this is the one test here that may call it.
//
// It is also the test whose absence let every dashed flag be dropped: feconf decodes the file and
// the flags into one key space and matches a key against the field's `json` tag, so `allow-media`
// bound to nothing at all — silently, for `--api-key` included.
func TestLoadBindsEveryFlagAndTheFile(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "weighd.toml")
	file := "model = \"/file/model\"\nport = 9999\ntimeout = 7\nmax-images = 3\n"
	if err := os.WriteFile(configPath, []byte(file), 0o600); err != nil {
		t.Fatal(err)
	}

	original := os.Args
	originalFlags := flag.CommandLine
	t.Cleanup(func() {
		os.Args = original
		flag.CommandLine = originalFlags
	})
	// The testing package has already parsed the real command line by the time a test runs, and
	// feconf only parses when the flag package has not. A fresh set is what lets this test exercise
	// the real Load path instead of silently skipping it.
	flag.CommandLine = flag.NewFlagSet("weighd", flag.ExitOnError)
	os.Args = []string{
		"weighd",
		"--config", configPath,
		"--backend", "sglang",
		"--readout", "top-n",
		"--template", "llama3",
		"--host", "0.0.0.0",
		"--port", "1234", // overrides the file
		"--workers", "5",
		"--timeout", "11", // overrides the file
		"--max-prompt-tokens", "2048",
		"--revision", "r7",
		"--allow-tokenizer-mismatch",
		"--api-key", "k",
		"--cors-origin", "https://x.example",
		"--allow-media",
		"--allow-media-topn",
		"--allow-remote-media",
		"--max-images", "7", // overrides the file
		"--max-media-bytes", "524288",
		"--url", "http://a:1,http://b:2",
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load refused a valid command line: %v", err)
	}

	// Every flag, not a sample: a tag typo that drops one flag would otherwise pass unnoticed.
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"model (from the file)", cfg.Model, "/file/model"},
		{"backend", cfg.Backend, "sglang"},
		{"readout", cfg.Readout, "top-n"},
		{"template", cfg.Template, "llama3"},
		{"host", cfg.Host, "0.0.0.0"},
		{"port (flag beats the file)", cfg.Port, 1234},
		{"workers", cfg.Workers, 5},
		{"timeout (flag beats the file)", cfg.Timeout, 11},
		{"max-prompt-tokens", cfg.MaxPromptTokens, 2048},
		{"revision", cfg.Revision, "r7"},
		{"allow-tokenizer-mismatch", cfg.AllowTokenizerMismatch, true},
		{"api-key", cfg.APIKey, "k"},
		{"cors-origin", cfg.CorsOrigin, "https://x.example"},
		{"allow-media", cfg.AllowMedia, true},
		{"allow-media-topn", cfg.AllowMediaTopN, true},
		{"allow-remote-media", cfg.AllowRemoteMedia, true},
		{"max-images (flag beats the file)", cfg.MaxImages, 7},
		{"max-media-bytes", cfg.MaxMediaBytes, 524288},
	}
	for _, check := range checks {
		if check.got != check.want {
			t.Errorf("%s = %v, want %v", check.name, check.got, check.want)
		}
	}
	if len(cfg.URLs) != 2 || cfg.URLs[0] != "http://a:1" || cfg.URLs[1] != "http://b:2" {
		t.Errorf("urls = %v, want both repeated values", cfg.URLs)
	}
}

// --help must be answered by main writing to stdout and exiting 0, not by the flag package
// writing to stderr and exiting 2.
func TestHelpIsReportedAsASentinel(t *testing.T) {
	original := os.Args
	t.Cleanup(func() { os.Args = original })
	for _, form := range []string{"-h", "-help", "--help"} {
		os.Args = []string{"weighd", form}
		_, err := Load()
		if !errors.Is(err, ErrHelp) {
			t.Errorf("Load(%q) = %v, want ErrHelp", form, err)
		}
	}
}

// Usage must document every flag the parser accepts, or --help lies about the contract.
func TestUsageDocumentsEveryFlag(t *testing.T) {
	cfg := &Config{}
	_ = cfg
	for _, flagName := range []string{
		"--model", "--url", "--backend", "--readout", "--template", "--host", "--port",
		"--workers", "--timeout", "--max-prompt-tokens", "--revision",
		"--allow-tokenizer-mismatch", "--api-key", "--cors-origin", "--allow-media",
		"--allow-media-topn", "--allow-remote-media", "--max-images", "--max-media-bytes",
		"--config", "--help",
	} {
		if !strings.Contains(usageText, flagName) {
			t.Errorf("usage text does not document %s", flagName)
		}
	}
}
