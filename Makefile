# weighd build entry point. Everything that produces an artifact goes through here so the
# version and build date reach the binary the same way in every environment
# (AGENTS.md, Build and Deployment 1-2).
SHELL := /bin/bash

MODULE     := github.com/wweir/weigh
CMD        := ./cmd/weighd
BIN        := dist/weighd
VERSION    ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
REVISION   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GOFLAGS    ?=

LDFLAGS := -s -w \
	-X $(MODULE)/internal/version.Version=$(VERSION) \
	-X $(MODULE)/internal/version.Revision=$(REVISION) \
	-X $(MODULE)/internal/version.BuildDate=$(BUILD_DATE)

.PHONY: all build test vet fmt fmt-check lint clean package release

all: build

## build: the release binary with version metadata injected
build:
	CGO_ENABLED=0 go build $(GOFLAGS) -trimpath -ldflags '$(LDFLAGS)' -o $(BIN) $(CMD)

## test: the hermetic suite (no network, no weights)
test:
	go test $(GOFLAGS) ./...

## vet: go vet on the whole module
vet:
	go vet $(GOFLAGS) ./...

## fmt: rewrite files with gofmt
fmt:
	gofmt -l -w .

## fmt-check: fail if any file is not gofmt-clean (the CI gate)
fmt-check:
	@out=$$(gofmt -l .); \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

## lint: the quality gate run before a commit
lint: fmt-check vet

## package: a release archive with the license next to the binary
package: build
	@set -euo pipefail; \
	name="weighd-$(VERSION)-$$(go env GOOS)-$$(go env GOARCH)"; \
	rm -rf "dist/$$name" "dist/$$name.tar.gz"; \
	mkdir -p "dist/$$name"; \
	cp $(BIN) LICENSE "dist/$$name/"; \
	tar -C dist -czf "dist/$$name.tar.gz" "$$name"; \
	cd dist && { command -v sha256sum >/dev/null && sha256sum "$$name.tar.gz" \
		|| shasum -a 256 "$$name.tar.gz"; } > "$$name.tar.gz.sha256"; \
	echo "packaged dist/$$name.tar.gz"

## release: the packaging path a tag uses
release: lint test package

clean:
	rm -rf dist
