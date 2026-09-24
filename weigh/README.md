# weigh

Ask an OpenAI-compatible inference server for the **exact** logprobs of the tokens *you* name,
and get a subset softmax over them plus the provenance to attribute the number.

> **Positioning.** Jev returns typed decisions from one vendor's API, with calibrated
> probabilities. `weigh` gives you the same *kind* of readout — a subset-softmax over named answer
> slots, **not** a calibrated confidence — from vLLM, SGLang, or anything OpenAI-compatible, with
> the provenance to attribute every number. Jev is TypeSafe's System One model; this project is
> independent and is **not** affiliated with or endorsed by TypeSafe. Jev, TypeSafe and other names
> are the property of their respective owners.

```rust
use std::time::Duration;
use weigh::{BackendChoice, Client, Config, MediaPolicy, Readout, TemplateChoice};

let (client, metadata) = Client::new(&Config {
    urls: vec!["http://127.0.0.1:8000".to_string()],
    backend: BackendChoice::Auto,
    readout: Readout::ExactSlot,
    template: TemplateChoice::Auto,
    timeout: Duration::from_secs(120),
    max_tokens: usize::MAX,                  // tightened to `max_model_len - 1`
    source: "/models/Qwen3-8B".to_string(),  // a directory containing `tokenizer.json`
    allow_tokenizer_mismatch: false,
    media: MediaPolicy::Off,
})?;

// The caller renders the prompt: this library renders no chat template.
let options = vec!["negative".to_string(), "positive".to_string()];
let prepared = client.prepare("row-1", "…the finished prompt…", &options)?;
// `fetch_any` has its own error type: it carries whether the budget ran out, which is the one
// backend failure a caller may reasonably retry.
let (scored, endpoint) = client.fetch_any(&prepared).map_err(|error| error.to_string())?;

// `scored.probabilities[i]` is the conditional score of `options[i]`, normalized over the
// declared slots. It is not calibrated decision confidence, and the crate never says it is.
for (label, probability) in scored.option_ids.iter().zip(&scored.probabilities) {
    println!("{label}: {probability}");
}
```

That is the whole loop. `examples/score_slots.rs` is the same code as a compilable target, so it
cannot drift from the documented API.

## Why name the slots instead of reading `top_logprobs`

A top-N readout only reports the slots that happen to land in the top N, and whether a given
slot is there depends on the prompt and on the server's state. A missing slot then has to be
recovered some other way — and **any recovery is a different readout**, so the numbers stop
being comparable within one result set. This is not hypothetical: on a real deployment the
recovery rate for a vLLM top-N setup was measured to climb from 0.1% to 97.9% over a run.

Naming the slots by token id removes the question. vLLM's `logprob_token_ids` (with
`return_tokens_as_token_ids`) and SGLang's `token_ids_logprob` return every requested slot **by
construction**, so the client either gets all of them or fails — it never silently renormalizes
over whatever happened to come back.

## What the library does and does not own

| Owns | Does not own |
|---|---|
| Which token ids are read, and the single-token boundary proof | The prompt. `prepare` takes a finished prompt string; the library renders no chat template |
| The subset softmax and its ordering | What the numbers mean for your product |
| Probing the backend: dialect, exact-slot support, media support | Your HTTP API shape |
| Fleet homogeneity checks (so two endpoints cannot be pooled by accident) | Sampling. This is a readout, not a generation |
| Provenance: prompt hash, request hash, readout description, timings | |

## Compatibility

| Backend | Exact slots | Notes |
|---|---|---|
| vLLM ≥ 0.26.0 | ✅ `logprob_token_ids` | The intended target |
| vLLM < 0.26.0 | ⚠️ top-N only | `Readout::TopN`; recovery rate varies with load, so numbers from the two readouts must not be pooled |
| SGLang | ✅ native `/generate` + `token_ids_logprob` | |
| Anything else OpenAI-compatible | ⚠️ top-N only | No exact-slot route, so the same caveat applies |

`Client::new` probes; it never guesses. If you ask for `Readout::ExactSlot` against a build that
cannot prove it, startup fails instead of serving numbers under a readout that did not run. A
refused connection or a 5xx is **not** treated as a negative capability answer.

## Building

Needs a Rust toolchain and a C/C++ compiler: `tokenizers` pulls `onig` (regex pre-tokenizers)
and `esaxx-rs` through its default features.

```bash
cargo test                 # hermetic: starts fake servers on 127.0.0.1, needs no network
cargo run --example score_slots   # needs a real server and a tokenizer directory
```

## Limitations, stated rather than discovered

- **No TLS.** Plain HTTP by design, which is what keeps the dependency tree small. Put it behind
  something that terminates TLS, or keep it on loopback.
- **No prompt rendering.** Chat templates are a moving target (\~140 names in one 17 KB Jinja
  template). The library takes finished prompt text, and `template::resolve` will tell you which
  wrapper a model directory declares so your own renderer can match it.
- **Images are a degraded readout.** With media, the backend's processor expands the image into
  placeholder tokens this client never sees, so the answer boundary is *not* proven locally. The
  provenance says so: `server_tokenized: true`, and `request_sha256` replaces `input_ids_sha256`.
  Media numbers and text numbers must not be pooled.
- **Slots are letters `A..P`.** Sixteen is a deliberate ceiling: the contract is only
  constructive while every slot letter is a single round-trip token in your tokenizer, and that
  is checked at startup rather than assumed.

## License

MPL-2.0. See `LICENSE`.
