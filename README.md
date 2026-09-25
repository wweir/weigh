# weigh

Exact per-slot logprobs from an OpenAI-compatible inference server, with the provenance to
attribute them — plus one consumer that serves a decision API on top of the readout.

| Crate | Kind | What it is |
|---|---|---|
| [`weigh`](weigh) | library | Exact per-slot logprobs from an OpenAI-compatible server, with provenance. Renders no prompt and knows nothing about decision contracts |
| [`weighd`](weighd) | binary | SemIf's `direct-options-v1` decision API over HTTP. Owns the prompt contract and the HTTP surface |

The split is an API boundary, not a rename. `weigh` never sees a "criterion", an
"evidence" string, or a `direct-options-v1` prompt, and does not depend on the HTTP layer;
`weighd` owns no transport, no tokenizer and no softmax. If you want the readout, take the
library; if you want the decision API, take the binary.

```bash
cargo test --workspace                       # 93 tests, hermetic: no network, no GPU, no weights
cargo build --workspace --release            # ~4.6 MB stripped binary
cargo run -p weighd --release -- --help
```

## Why this exists

Reading a decision out of a generative model means asking "what is the probability of *this*
answer", not "what does the model feel like saying". Doing that faithfully has three
requirements, and this workspace enforces all three at startup rather than hoping:

1. **The slots are named.** `logprob_token_ids` (vLLM) / `token_ids_logprob` (SGLang) return
   every declared slot by construction, so no slot can be silently missing.
2. **The answer is one token.** The client proves `encode(prompt + letter) == ids + [slot]`, so
   the position the server scores is the position the client means.
3. **The number is attributable.** Every response carries the prompt hash, the readout path, and
   the serving configuration — because two different readouts must never be pooled, and a number
   without its provenance is not evidence.

Everything these crates refuse, they refuse loudly: an unrecognised chat template, a backend
that cannot prove the exact-slot route, a schema with two fields, an image in the criterion. The
alternative — guessing — produces plausible numbers that mean something else.

## Acknowledgements

The `direct-options-v1` contract that `weighd` implements, and the measurement discipline
these crates inherit, come from the **SemIf** project — an independent research effort on reading
decisions out of a model's option logits. Thanks to it.

This repository is standalone: it shares no git history, no release process and no build with that
project, and nothing here is a fork of it.

## License

MPL-2.0. See `LICENSE`.
