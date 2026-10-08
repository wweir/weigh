# Daily development entry point (AGENTS.md Deploy conventions 1): build with the same
# ldflags the Makefile uses and run the service. Extra arguments are passed through.
#
#   just deploy --model /models/gemma-3-27b --url http://localhost:8000
#
# `just deploy` alone is not enough to serve: --model and a reachable backend are required,
# and the process refuses to start without them rather than serving unlabelled numbers.

# build and run weighd
deploy *args:
    #!/usr/bin/env bash
    set -euo pipefail
    make build
    exec ./dist/weighd {{args}}

# the quality gate
check:
    make lint test

# the release binary and archives
release:
    make release

# list the recipes
default:
    @just --list
