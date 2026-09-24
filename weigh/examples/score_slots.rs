// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Score one decision against a real server.
//!
//! `cargo run --example score_slots -- <model-dir> <url>`
//!
//! This is the README's example, kept as a compilable target so it cannot drift from the
//! documented API. It needs a live server and a real tokenizer directory to do anything, which
//! is why the test suite does not run it; `cargo clippy --all-targets` does build it.

use std::time::Duration;

use weigh::{BackendChoice, Client, Config, MediaPolicy, Readout, TemplateChoice};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .unwrap_or_else(|| "/models/Qwen3-8B".to_string());
    let url = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    score(&model, &url)
}

fn score(model: &str, url: &str) -> Result<(), String> {
    let (client, metadata) = Client::new(&Config {
        urls: vec![url.to_string()],
        backend: BackendChoice::Auto,
        readout: Readout::ExactSlot,
        template: TemplateChoice::Auto,
        timeout: Duration::from_secs(120),
        max_tokens: usize::MAX, // tightened to `max_model_len - 1` by the client
        source: model.to_string(),
        allow_tokenizer_mismatch: false,
        media: MediaPolicy::Off,
    })?;
    eprintln!(
        "backend={} exact_slots={} template={} serving_config={}",
        metadata.backend.as_str(),
        metadata.tier_a,
        metadata.prompt_template.as_str(),
        weigh::serving_config_for(metadata.backend, metadata.tier_a),
    );

    // The caller renders the prompt: this library deliberately renders no chat template.
    let prompt = "Apply the supplied criterion to the supplied evidence.\n\n<your evidence>";
    let options = vec!["negative".to_string(), "positive".to_string()];

    let prepared = client.prepare("row-1", prompt, &options)?;
    // `fetch_any` has its own error type: it carries whether the budget ran out, which is the
    // one backend failure a caller may reasonably retry.
    let (scored, endpoint) = client
        .fetch_any(&prepared)
        .map_err(|error| error.to_string())?;

    for (label, probability) in scored.option_ids.iter().zip(&scored.probabilities) {
        println!("{label}: {probability:.6}");
    }
    // Provenance travels with the number, not in a separate log: this is what makes a result
    // attributable to a readout later.
    println!("endpoint={endpoint} readout={}", scored.readout);
    println!("prompt_sha256={:?}", scored.prompt_sha256);
    println!("input_tokens={:?}", scored.input_tokens);
    Ok(())
}
