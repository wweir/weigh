# weigh

[English](README.md) | [简体中文](README.zh-CN.md)

Exact per-slot logprobs from an OpenAI-compatible inference server, with the provenance to
attribute them — served as SemIf's `direct-options-v1` decision API.

`weighd` reads a decision out of a generative model by asking for the probability of *named
answers*, not by asking the model what it feels like saying. For one request it validates a
single-field decision schema, renders the `direct-options-v1` prompt, asks the backend for the
exact logprob of every declared slot at the one scored position, and returns the argmax as a chat
completion whose content is constructed locally rather than generated.

```bash
make build        # dist/weighd, with version and build date injected from git
make test         # the hermetic suite: no network, no GPU, no weights
make lint         # gofmt --check + go vet
just deploy -- --model /models/gemma-3-27b --url http://localhost:8000
```

`--url` is the backend **root**, without `/v1`: the service appends its own paths (`/version`,
`/v1/models`, `/tokenize`).

## As a library

The same readout is importable: `github.com/wweir/weigh`. The caller renders the prompt — the
library renders no chat template — and gets back the subset softmax plus the provenance to
attribute it:

```go
client, metadata, err := weigh.New(ctx, weigh.Config{
	URLs:    []string{"http://localhost:8000"},
	Backend: weigh.BackendVLLM,
	Readout: weigh.ReadoutExactSlot,
	Timeout: 120 * time.Second,
	// No Source: this process renders its own prompt and has no local checkpoint to read.
})
// The prompt is bytes you built; optionIDs[i] is the label you bound to answer slot i (A..P).
scored, err := client.Decide(ctx, "row-1", finishedPrompt, optionIDs)
// scored.Probabilities[i] is optionIDs[i]'s score at the one scored position.
// scored.PromptSHA256 is the digest of exactly the bytes you passed.
```

`Config.Source` is optional. Without it there is no checkpoint to detect from, so an explicitly
named family is still recorded and renderable, `TemplateAuto` is refused, and an unnamed template
leaves `Metadata.PromptTemplate` empty. A caller that wants the service's own `direct-options-v1`
bytes can build them with `weigh.ValidateRow` + `weigh.RenderDirectOptions` against the family it
configured, instead of reimplementing the contract. `weigh.New` probes and refuses exactly as the
binary does; nothing degrades silently.

## Why this exists

Doing this faithfully has three requirements, and the service enforces all three at startup rather
than hoping:

1. **The slots are named.** `logprob_token_ids` (vLLM) / `token_ids_logprob` (SGLang) return every
   declared slot by construction, so no slot can be silently missing.
2. **The answer is one token.** The client asks the backend to prove
   `tokenize(prompt + letter) == ids + [slot]`, so the position the engine scores is the position the
   client means — and it proves it with the engine's own tokenizer, not with a second implementation
   that could disagree (see `docs/tokenize-contract.md`).
3. **The number is attributable.** Every response carries the prompt hash, the readout path, the
   serving configuration and the fallback that was used — because two different readouts must never
   be pooled, and a number without its provenance is not evidence (see `docs/provenance.md`).

Everything the service refuses, it refuses loudly: an unrecognised chat template, a backend that
cannot prove the exact-slot route, a schema with two fields, an image in the criterion. The
alternative — guessing — produces plausible numbers that mean something else.

## Layout

| Path | What it is |
|---|---|
| `weigh.go`, `types.go`, `assemble.go` | the library surface: `weigh.New` / `Client.Decide` over a caller-rendered prompt, plus the assembly primitives and the re-exported types |
| `cmd/weighd` | entry point: argv, config, logger, exit codes |
| `internal/serve` | the HTTP contract: routing, validation, error envelope, SSE, `/health`, drain, metrics |
| `internal/{schema,prompt,template,slots}` | the decision contract: refusals, the payload, the chat-template registry, the answer slots |
| `internal/{readout,tokenize,media}` | the readout: backend probe, exact-slot requests, softmax, provenance, images |
| `api/openapi.json` | the machine-readable contract, embedded and served at `GET /openapi.json` |
| `config/`, `deploy/` | configuration and the systemd unit |
| `docs/` | the design documents; `ARCHITECTURE.md` is the index |

## Acknowledgements

The `direct-options-v1` contract this service implements, and the measurement discipline it
inherits, come from the **SemIf** project — an independent research effort on reading decisions out
of a model's option logits. Thanks to it.

## License

MPL-2.0. See `LICENSE`.
