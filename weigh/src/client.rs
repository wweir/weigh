// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The vLLM/SGLang scoring client: prompt encoding, the answer-boundary check, the HTTP
//! readout, and the subset softmax.
//!
//! Three text readouts, decided once at startup and recorded in the artifact rather than
//! re-decided per row:
//!
//! * **vLLM exact-slot** (`logprob_token_ids` + `return_tokens_as_token_ids`): the declared
//!   slots are named by token id and come back under `token_id:<id>` keys, so no slot can miss
//!   a top-N and there is nothing to recover. This is what `Readout::ExactSlot` requires, and
//!   [`Client::new`] refuses that readout when the build cannot prove it.
//! * **vLLM top-N** (`Readout::TopN`, for a build that ignores `logprob_token_ids`):
//!   `/v1/completions` returns `top_logprobs` keyed by *decoded string*, so option slots are
//!   matched by the text their token decodes to, and a slot outside the returned top-N is
//!   recovered through `/generative_scoring`, which has no top-N cap but returns probabilities
//!   already normalized over the supplied subset. Measured on the build this path was written
//!   for (vLLM 0.23.0): pre-tokenized prompts are accepted, `logprobs` above 20 is a hard HTTP
//!   400, and that recovery rate rises with load (0.1% -> 97.9%) -- which is why numbers from
//!   the two readouts must never be pooled.
//! * **SGLang exact-slot** (native `/generate` + `token_ids_logprob`): slots named by token id,
//!   no top-N dependence, no fallback.
//!
//! An image moves the row to the chat route instead (see [`crate::media`]): the backend's own
//! processor tokenizes the prompt, so the boundary proof does not apply, and the response says
//! so rather than implying an `input_ids_sha256` it never had.
//!
//! ## Serving across several vLLM endpoints
//!
//! A *fleet* of endpoints that serve the **same checkpoint** can be scored together for
//! throughput. Two rules keep that safe:
//!
//! 1. Every endpoint is verified at startup, and homogeneity is enforced rather than
//!    assumed: identical checkpoint root, identical first served-model id, identical
//!    server version. A mixed fleet is exactly the silent-mismatch class this pipeline
//!    is built to refuse.
//! 2. A row's **whole lifecycle stays on one endpoint**. Its `/v1/completions` request
//!    AND any `/generative_scoring` fallback must hit the same server, because the
//!    fallback exists precisely because that server's top-N omitted a slot; taking the
//!    fallback from a different replica would combine two engines' decisions.
//!
//! Fanning out spends reproducibility to buy throughput: this deployment is measured
//! nondeterministic run-to-run even at concurrency 1 (same client, same parameters,
//! 1329/4000 identical lines, max |dP| 0.412 on the 4000-row workload), so with N
//! replicas a row's value also depends on *which* replica served it. Single-endpoint
//! output is byte-identical to the reference Python implementation; multi-endpoint output adds the
//! per-row `endpoint` field so that dependence is recorded instead of hidden.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tokenizers::Tokenizer;

use crate::media::{self, Media};
use crate::pyjson::dumps_int_list;
use crate::slots::LETTERS;
use crate::template::{self, ChatTemplate, TemplateChoice};

/// Measured on the deployed build: `logprobs` above this is a hard HTTP 400.
pub const MAX_LOGPROBS: usize = 20;

/// Which serving dialect one endpoint speaks. Selected per endpoint by `--backend`
/// (explicit) or by probing (`auto`), never assumed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Vllm,
    Sglang,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Vllm => "vllm",
            Backend::Sglang => "sglang",
        }
    }
}

/// The `--backend` flag before probing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendChoice {
    Auto,
    Vllm,
    Sglang,
}

impl BackendChoice {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "auto" => Ok(BackendChoice::Auto),
            "vllm" => Ok(BackendChoice::Vllm),
            "sglang" => Ok(BackendChoice::Sglang),
            other => Err(format!(
                "unknown backend {:?}; expected auto, vllm, or sglang",
                other
            )),
        }
    }
}

/// How the readout is selected, independent of which dialect the endpoint speaks.
///
/// The default is deliberately the already-measured vLLM path: switching a run to Tier A
/// changes the numbers on every row whose slot would have missed the top-N, so it is an
/// explicit choice, not something a probe decides on the user's behalf.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Readout {
    /// Preserve each backend's measured default: vLLM top-N + fallback, SGLang exact-slot.
    Auto,
    /// Require the exact-slot path (vLLM `logprob_token_ids`); error if unavailable.
    ExactSlot,
    /// Force the vLLM top-N + fallback path.
    TopN,
}

impl Readout {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "auto" => Ok(Readout::Auto),
            "exact-slot" => Ok(Readout::ExactSlot),
            "top-n" => Ok(Readout::TopN),
            other => Err(format!(
                "unknown readout {:?}; expected auto, exact-slot, or top-n",
                other
            )),
        }
    }
}

/// The serving recipe recorded in the artifact, per backend and readout tier.
///
/// Two different values mean the numbers are not comparable and must not be pooled.
pub fn serving_config_for(backend: Backend, tier_a: bool) -> &'static str {
    match (backend, tier_a) {
        (Backend::Vllm, true) => "vllm-openai-logprob-token-ids-v1",
        (Backend::Vllm, false) => "vllm-openai-completions-subset-softmax-v1",
        (Backend::Sglang, _) => "sglang-native-generate-token-ids-v1",
    }
}

/// What a multimodal request can get from an endpoint, decided by a startup probe.
///
/// Deliberately a **separate axis** from `tier_a`: the text readout and the media readout
/// are different routes with different proofs. A server
/// can be Tier A for text and have no media path at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MediaSupport {
    /// The probe was refused, the model has no vision path, or media was not opted into.
    None,
    /// `/v1/chat/completions` accepted the image but answered under decoded-text keys, so a
    /// slot outside the returned top-N has **no recovery**: the text path's
    /// `/generative_scoring` fallback takes token ids and cannot carry an image.
    ChatTopN,
    /// `/v1/chat/completions` honoured `logprob_token_ids`, so every declared slot is named
    /// by id and returned by construction -- the only media tier that can promise that.
    ChatExactSlot,
}

impl MediaSupport {
    pub fn as_str(self) -> &'static str {
        match self {
            MediaSupport::None => "none",
            MediaSupport::ChatTopN => "chat-top-n",
            MediaSupport::ChatExactSlot => "chat-exact-slot",
        }
    }
}

/// `--allow-media` / `--allow-media-topn`, before probing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MediaPolicy {
    /// No media at all: an image_url part is refused exactly as it was before this feature.
    Off,
    /// Accept media only where the probe proved the exact-slot route.
    ExactSlot,
    /// Additionally accept the best-effort top-N route (hard-fails on a missing slot).
    TopN,
}

impl MediaPolicy {
    pub fn enabled(self) -> bool {
        self != MediaPolicy::Off
    }
}

/// The serving recipe recorded for a media request.
///
/// Distinct from every text recipe on purpose: media numbers and text numbers must never
/// be pooled, because the media prompt is server-tokenized and its boundary is unproven.
pub fn media_serving_config_for(backend: Backend, support: MediaSupport) -> Option<&'static str> {
    match (backend, support) {
        (Backend::Vllm, MediaSupport::ChatExactSlot) => Some("vllm-openai-chat-media-token-ids-v1"),
        (Backend::Vllm, MediaSupport::ChatTopN) => Some("vllm-openai-chat-media-topn-v1"),
        // Every other pair is refused before a media readout can run (`prepare_media` and
        // `fetch_chat_media` both return an error), so there is no recipe to name. `None`
        // keeps that in the type instead of inventing a string that looks like a recipe a
        // consumer could group by.
        _ => None,
    }
}

/// Local prompt-length ceiling: `--max-tokens` capped by the server's context, minus one
/// token for the single scored position (`max_tokens: 1` / `max_new_tokens: 1`).
pub fn context_limit_for(max_tokens: usize, context: Option<i64>) -> usize {
    match context {
        Some(context) => max_tokens.min((context.max(0) as usize).saturating_sub(1)),
        None => max_tokens,
    }
}

/// The boundary check depends only on the tokenized end of the prompt plus the slot
/// ids, so it repeats across a batch. Matches `direct._BOUNDARY_WINDOW`.
const BOUNDARY_WINDOW: usize = 8;

/// Upper bound on the boundary-proof memo. In a long-running server the key space is
/// every distinct prompt suffix, so an unbounded memo is a leak. Dropping entries is
/// always safe: the check it memoizes is a pure function of the tokenizer and the ids.
const BOUNDARY_CACHE_CAP: usize = 4096;

/// One endpoint's identity, as recorded in the artifact's provenance.
#[derive(Clone)]
pub struct EndpointInfo {
    pub url: String,
    pub served_models: Vec<String>,
    pub model_root: Option<String>,
    pub version: Option<serde_json::Value>,
}

pub struct ServerMetadata {
    pub backend: Backend,
    /// True when every endpoint's readout is the exact-slot Tier A path (vLLM's
    /// `logprob_token_ids`). Decided by a startup probe, not by a version guess.
    pub tier_a: bool,
    /// What every endpoint can do with an image, probed at startup. `None` on a fleet
    /// where media was not opted into.
    pub media_support: MediaSupport,
    /// Always at least one. Order matches `--url` order, which is also the routing
    /// order, so a recorded `endpoint` index is stable and interpretable.
    pub endpoints: Vec<EndpointInfo>,
    pub max_model_len: Option<i64>,
    pub tokenizer_matches: Option<bool>,
    /// Which chat wrapper was rendered, and how that was decided (`explicit`,
    /// `detected:<path>`, or `default:no-chat-template`). The wrapper is part of what the
    /// numbers mean, so it is provenance, not a detail.
    pub prompt_template: ChatTemplate,
    pub prompt_template_source: String,
}

pub struct Config {
    /// One or more base URLs of servers that all serve `source`.
    pub urls: Vec<String>,
    /// `--backend`: `Auto` probes each endpoint; an explicit value skips detection
    /// but still verifies identity.
    pub backend: BackendChoice,
    /// `--readout`: how the exact-slot vs top-N trade-off is chosen.
    pub readout: Readout,
    /// `--template`: which chat wrapper to render (`Auto` reads the tokenizer's own
    /// `chat_template`).
    pub template: TemplateChoice,
    pub timeout: Duration,
    pub max_tokens: usize,
    pub source: String,
    pub allow_tokenizer_mismatch: bool,
    /// `--allow-media` / `--allow-media-topn`: whether to probe for, and then accept,
    /// multimodal requests.
    pub media: MediaPolicy,
}

/// One row, encoded and boundary-verified, ready to fetch.
///
/// Exactly one of `ids` / `chat` is set: the text path carries locally tokenized `ids`, the
/// media path carries server-tokenized `chat` messages.
#[derive(Clone)]
pub struct Prepared {
    pub row_id: String,
    pub option_ids: Vec<String>,
    ids: Vec<u32>,
    slots: Vec<u32>,
    keys: Vec<String>,
    prompt_hash: String,
    /// Media only: the OpenAI `messages` array handed to `/v1/chat/completions`.
    chat: Option<serde_json::Value>,
    /// Media only: one provenance object per attached image, recorded in the response.
    media: Vec<serde_json::Value>,
}

impl Prepared {
    /// True when this row must go through the media (server-tokenized chat) readout.
    pub fn is_media(&self) -> bool {
        self.chat.is_some()
    }
}

/// One scored decision.
///
/// The HTTP service is the only consumer, so the readout returns this directly instead of
/// being serialized into a batch row and parsed back.
#[derive(Debug, Clone)]
pub struct Scored {
    pub probabilities: Vec<f64>,
    pub option_logprobs: Vec<f64>,
    pub answer_token_ids: Vec<u32>,
    pub option_ids: Vec<String>,
    /// Client-counted on the text path; the backend's `usage.prompt_tokens` on the media
    /// path (or `None` when the backend did not report it).
    pub input_tokens: Option<usize>,
    /// Text path only: sha256 of `json.dumps(ids)`.
    pub input_ids_sha256: Option<String>,
    /// Text path only: sha256 of the rendered prompt string.
    pub prompt_sha256: Option<String>,
    /// Media path only: sha256 of the request body sent to the backend.
    pub request_sha256: Option<String>,
    /// `"text"` or `"text+image"`; recorded so the two can never be pooled silently.
    pub modality: &'static str,
    /// True when the backend tokenized the prompt (media path): the boundary is unproven.
    pub server_tokenized: bool,
    /// Media path only: one provenance object per attached image.
    pub media: Vec<serde_json::Value>,
    pub readout: String,
    pub fallback_used: bool,
    pub engine_seconds: f64,
    pub fallback_seconds: f64,
    pub total_seconds: f64,
}

/// Build one scored decision.
///
/// Shared by every readout path so the checks cannot drift between vLLM Tier A, vLLM
/// B-text, and SGLang.
#[allow(clippy::too_many_arguments)]
fn scored(
    item: &Prepared,
    probabilities: Vec<f64>,
    option_logprobs: Vec<f64>,
    readout: String,
    fallback: bool,
    fallback_seconds: f64,
    engine_seconds: f64,
    total_seconds: f64,
) -> Result<Scored, String> {
    if option_logprobs.len() != item.slots.len() {
        return Err(format!("Row {}: incomplete option logprobs", item.row_id));
    }
    if probabilities.len() != item.slots.len() {
        return Err(format!("Row {}: incomplete probabilities", item.row_id));
    }
    // Python's `hashlib.sha256(json.dumps(item["ids"]).encode())`: the preimage includes
    // the spaces json.dumps puts after each comma. Kept even though the service emits no
    // JSONL, because this hash is how a served decision is matched to a recorded row.
    let input_ids_sha256 = media::sha256_hex(dumps_int_list(&item.ids).as_bytes());
    Ok(Scored {
        probabilities,
        option_logprobs,
        answer_token_ids: item.slots.clone(),
        option_ids: item.option_ids.clone(),
        input_tokens: Some(item.ids.len()),
        input_ids_sha256: Some(input_ids_sha256),
        prompt_sha256: Some(item.prompt_hash.clone()),
        request_sha256: None,
        modality: "text",
        server_tokenized: false,
        media: Vec::new(),
        readout,
        fallback_used: fallback,
        engine_seconds,
        fallback_seconds,
        total_seconds,
    })
}

/// Build one multimodal decision.
///
/// Mirrors `scored` but records the media provenance: the promise that `input_ids_sha256`
/// meant ("the server saw these exact ids") is replaced by `request_sha256` ("the backend
/// saw this exact request"), and `server_tokenized` says the boundary was not proven here.
#[allow(clippy::too_many_arguments)]
fn scored_media(
    item: &Prepared,
    probabilities: Vec<f64>,
    option_logprobs: Vec<f64>,
    input_tokens: Option<usize>,
    request_sha256: String,
    readout: String,
    engine_seconds: f64,
    total_seconds: f64,
) -> Result<Scored, String> {
    if option_logprobs.len() != item.slots.len() {
        return Err(format!("Row {}: incomplete option logprobs", item.row_id));
    }
    if probabilities.len() != item.slots.len() {
        return Err(format!("Row {}: incomplete probabilities", item.row_id));
    }
    Ok(Scored {
        probabilities,
        option_logprobs,
        answer_token_ids: item.slots.clone(),
        option_ids: item.option_ids.clone(),
        input_tokens,
        input_ids_sha256: None,
        prompt_sha256: None,
        request_sha256: Some(request_sha256),
        modality: "text+image",
        server_tokenized: true,
        media: item.media.clone(),
        readout,
        fallback_used: false,
        engine_seconds,
        fallback_seconds: 0.0,
        total_seconds,
    })
}

/// Why a served readout could not be produced.
///
/// `timed_out` is carried to the HTTP layer because it is the one backend failure with a
/// distinct, actionable status: 504, where a caller may reasonably retry, rather than the
/// 502 that covers "the backend refused or broke".
///
/// Every string-returning helper in this module converts through `From<String>` with the
/// flag unset, which is why widening this error type did not touch their `?` sites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutError {
    pub message: String,
    pub timed_out: bool,
}

impl std::fmt::Display for ReadoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl From<String> for ReadoutError {
    fn from(message: String) -> Self {
        ReadoutError {
            message,
            timed_out: false,
        }
    }
}

/// A transport-layer failure that must not be mistaken for a server's negative answer.
///
/// The startup probes decide a backend and a readout tier; a timeout or a refused
/// connection is not evidence about either, so it is propagated and the run fails instead
/// of silently picking a dialect and a readout path.
#[derive(Debug)]
enum CallError {
    /// The server answered with a non-2xx status: a real answer, even when negative.
    Status { code: u16, detail: String },
    /// No usable answer (connect/read/write failure, or a non-JSON body).
    Transport(String),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Status { code, detail } => {
                write!(formatter, "HTTP {} from server: {}", code, detail)
            }
            CallError::Transport(detail) => formatter.write_str(detail),
        }
    }
}

/// POST a JSON body and parse the JSON response; used by both the row readouts and the
/// startup capability probes.
fn post_json(
    agent: &ureq::Agent,
    base_url: &str,
    path: &str,
    body: &str,
) -> Result<serde_json::Value, CallError> {
    let url = format!("{}{}", base_url, path);
    let request = agent.post(&url).set("Content-Type", "application/json");
    match request.send_string(body) {
        Ok(response) => response
            .into_string()
            .map_err(|error| {
                CallError::Transport(format!("Cannot read {} response: {}", path, error))
            })
            .and_then(|text| {
                serde_json::from_str(&text).map_err(|error| {
                    CallError::Transport(format!("{} returned invalid JSON: {}", path, error))
                })
            }),
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            let detail: String = detail.chars().take(500).collect();
            Err(CallError::Status { code, detail })
        }
        Err(error) => Err(CallError::Transport(format!(
            "Cannot reach {}: {}",
            url, error
        ))),
    }
}

/// Mirror of `core.softmax`, operation for operation.
///
/// The subtraction of the maximum, the left-to-right summation, and `f64::exp` (which
/// lowers to libm's `exp`, the same function CPython's `math.exp` calls) are all
/// reproduced deliberately: reordering any of them would move the last bits.
fn softmax(values: &[f64]) -> Result<Vec<f64>, String> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return Err("Need at least two finite scores".to_string());
    }
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    Ok(weights.iter().map(|weight| weight / total).collect())
}

/// Read the pinned slots out of a vLLM Tier A `top_logprobs` object.
///
/// `return_tokens_as_token_ids` keys each entry by `token_id:<id>`, which is
/// deterministic in the id -- exactly the property the B-text path lacks, where the key
/// is the server's detokenization of the token. A missing slot here is a contract
/// violation, not a condition to paper over: Tier A exists to make "every declared slot
/// was returned" true by construction.
fn parse_vllm_token_ids(
    top: &serde_json::Map<String, serde_json::Value>,
    slots: &[u32],
    row_id: &str,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        let key = format!("token_id:{}", slot);
        let value = top
            .get(&key)
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                format!(
                    "Row {}: Tier A response omitted requested slot {} (key {:?}); the server \
                 accepted `logprob_token_ids` but did not echo it",
                    row_id, slot, key
                )
            })?;
        if !value.is_finite() {
            return Err(format!(
                "Row {}: non-finite logprob for slot {}",
                row_id, slot
            ));
        }
        out.push(value);
    }
    Ok(out)
}

/// Read the pinned slots out of an SGLang `/generate` response.
///
/// The external key is `meta_info.output_token_ids_logprobs`, a list per output position
/// whose entries are `[logprob, token_id, token_text|null]`; `_val`/`_idx` are internal
/// scheduler IPC names and never appear in the HTTP response. Slot matching is by
/// `token_id`, so it does not depend on detokenization.
fn parse_sglang_token_ids(
    response: &serde_json::Value,
    slots: &[u32],
    row_id: &str,
) -> Result<Vec<f64>, String> {
    let entries = response
        .get("meta_info")
        .and_then(|value| value.get("output_token_ids_logprobs"))
        .and_then(|value| value.as_array())
        .and_then(|positions| positions.first())
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            format!(
                "Row {}: /generate returned no meta_info.output_token_ids_logprobs",
                row_id
            )
        })?;
    let mut by_token_id: HashMap<i64, f64> = HashMap::with_capacity(entries.len());
    for entry in entries {
        let pair = entry
            .as_array()
            .ok_or_else(|| format!("Row {}: token-logprob entry was not an array", row_id))?;
        let value = pair.first().and_then(|value| value.as_f64());
        let token_id = pair.get(1).and_then(|value| value.as_i64());
        match (value, token_id) {
            (Some(value), Some(token_id)) if value.is_finite() => {
                by_token_id.insert(token_id, value);
            }
            _ => {
                return Err(format!(
                    "Row {}: unusable token-logprob entry {:?}",
                    row_id, entry
                ))
            }
        }
    }
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        let value = by_token_id.get(&(*slot as i64)).ok_or_else(|| {
            format!(
                "Row {}: SGLang omitted requested slot {} from output_token_ids_logprobs",
                row_id, slot
            )
        })?;
        out.push(*value);
    }
    Ok(out)
}

/// A single serving endpoint: its URL, its own connection pool, its dialect, and its
/// identity.
struct Endpoint {
    url: String,
    agent: ureq::Agent,
    served_model: String,
    backend: Backend,
    /// vLLM only: the startup probe proved `/v1/completions` honors
    /// `logprob_token_ids`, so the readout is exact-slot (Tier A) with no fallback.
    tier_a: bool,
    /// What this same endpoint does with an image on `/v1/chat/completions`.
    media: MediaSupport,
}

impl Endpoint {
    fn post_json(&self, path: &str, body: &str) -> Result<serde_json::Value, String> {
        post_json(&self.agent, &self.url, path, body).map_err(|error| error.to_string())
    }

    /// Route one prepared row to the dialect this endpoint speaks.
    ///
    /// The readout (vLLM Tier A exact-slot vs the measured B-text fallback path, or
    /// SGLang's native exact-slot path) is decided once at startup and recorded in the
    /// artifact; it is never re-decided per row. Every failure is returned, never
    /// papered over.
    fn fetch(&self, item: &Prepared, timeout: Duration) -> Result<Scored, ReadoutError> {
        let started = Instant::now();
        // The dialect helpers keep returning `String`; the flag is added here, where the
        // budget is known. `ureq`'s transport error does not expose the inner I/O error, so
        // a read that consumed the whole budget and then failed is the timeout case. A
        // refused connection or a bad status fails immediately and never reaches it.
        let result = if item.is_media() {
            // The media readout is a different route (`/v1/chat/completions`), so it is
            // selected by the *request*, not by the endpoint's text dialect.
            self.fetch_chat_media(item, started)
        } else {
            match self.backend {
                Backend::Vllm if self.tier_a => self.fetch_vllm_tier_a(item, started),
                Backend::Vllm => self.fetch_vllm(item, started),
                Backend::Sglang => self.fetch_sglang(item, started),
            }
        };
        result.map_err(|message| ReadoutError {
            message,
            timed_out: !timeout.is_zero() && started.elapsed() >= timeout,
        })
    }

    /// vLLM Tier B-text: `/v1/completions` top-20 keyed by decoded string, with the
    /// measured `/generative_scoring` recovery for a slot outside the top-N.
    fn fetch_vllm(&self, item: &Prepared, started: Instant) -> Result<Scored, String> {
        // Always request the server's cap: an option slot's rank among ALL vocabulary
        // entries is unrelated to the option count (measured: one binary row needed
        // rank 18, so a len(options)+4 heuristic failed 14/777 rows).
        let payload = format!(
            "{{\"model\": {}, \"prompt\": {}, \"max_tokens\": 1, \"temperature\": 0, \"logprobs\": {}}}",
            serde_json::Value::String(self.served_model.clone()),
            dumps_int_list(&item.ids),
            MAX_LOGPROBS
        );
        let send = Instant::now();
        let response = self
            .post_json("/v1/completions", &payload)
            .map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let received = Instant::now();

        let top = response
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .ok_or_else(|| format!("Row {}: response carried no choices", item.row_id))?
            .get("logprobs")
            .and_then(|value| value.get("top_logprobs"))
            .and_then(|value| value.as_array())
            .and_then(|entries| entries.first())
            .cloned()
            .ok_or_else(|| format!("Row {}: response carried no top_logprobs", item.row_id))?;

        let mut selected: Vec<Option<f64>> = item
            .keys
            .iter()
            .map(|key| top.get(key).and_then(|value| value.as_f64()))
            .collect();

        let mut readout = "vLLM /v1/completions max_tokens=1 logprobs=N; subset softmax \
                           over declared answer slots matched by decoded token text"
            .to_string();
        let mut fallback = false;
        let mut fallback_seconds = 0.0f64;

        let probabilities = if selected.iter().any(Option::is_none) {
            // A slot outside THIS server's returned top-N. /generative_scoring honors
            // logprob_token_ids, so it has no top-N cap, and it returns PROBABILITIES
            // already normalized over the option subset -- exactly this readout's
            // quantity, so they must NOT be passed through softmax() again. It is
            // requested from `self` on purpose: the fallback only makes sense against
            // the same server whose top-N missed the slot.
            let fallback_started = Instant::now();
            let recovered = self.fetch_via_generative_scoring(item)?;
            fallback_seconds = fallback_started.elapsed().as_secs_f64();
            readout = "vLLM /generative_scoring subset softmax over declared answer slots \
                       (fallback: option slots outside the /v1/completions top-N)"
                .to_string();
            fallback = true;
            // Rebuild the logprob companion from the probabilities: log(p) is exactly
            // what the /v1/completions path would have put in that field.
            selected = recovered.iter().map(|value| Some(value.ln())).collect();
            recovered
        } else {
            let values: Vec<f64> = selected.iter().filter_map(|value| *value).collect();
            softmax(&values).map_err(|error| format!("Row {}: {}", item.row_id, error))?
        };

        let option_logprobs: Vec<f64> = selected.iter().filter_map(|value| *value).collect();
        let engine_seconds = (received - send).as_secs_f64() + fallback_seconds;
        scored(
            item,
            probabilities,
            option_logprobs,
            readout,
            fallback,
            fallback_seconds,
            engine_seconds,
            started.elapsed().as_secs_f64(),
        )
    }

    /// vLLM Tier A: one request names the answer slots by token id (`logprob_token_ids`)
    /// and asks for them back keyed by id (`return_tokens_as_token_ids`), so there is no
    /// top-N dependence and no decoded-string matching -- hence no fallback.
    ///
    /// A slot missing from this response is a contract violation, not a condition to
    /// paper over: Tier A exists precisely to make "every declared slot was returned"
    /// true by construction.
    fn fetch_vllm_tier_a(&self, item: &Prepared, started: Instant) -> Result<Scored, String> {
        let payload = format!(
            "{{\"model\": {}, \"prompt\": {}, \"max_tokens\": 1, \"temperature\": 0, \"logprobs\": true, \"logprob_token_ids\": {}, \"return_tokens_as_token_ids\": true}}",
            serde_json::Value::String(self.served_model.clone()),
            dumps_int_list(&item.ids),
            dumps_int_list(&item.slots)
        );
        let send = Instant::now();
        let response = self
            .post_json("/v1/completions", &payload)
            .map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let received = Instant::now();

        let top = response
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .ok_or_else(|| format!("Row {}: response carried no choices", item.row_id))?
            .get("logprobs")
            .and_then(|value| value.get("top_logprobs"))
            .and_then(|value| value.as_array())
            .and_then(|entries| entries.first())
            .cloned()
            .ok_or_else(|| format!("Row {}: response carried no top_logprobs", item.row_id))?;
        let top = top
            .as_object()
            .ok_or_else(|| format!("Row {}: top_logprobs was not an object", item.row_id))?;
        let option_logprobs = parse_vllm_token_ids(top, &item.slots, &item.row_id)?;
        let probabilities =
            softmax(&option_logprobs).map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let readout = "vLLM /v1/completions logprob_token_ids: exact answer slots keyed by \
                       token_id, no top-N dependence and no fallback"
            .to_string();
        let engine_seconds = (received - send).as_secs_f64();
        scored(
            item,
            probabilities,
            option_logprobs,
            readout,
            false,
            0.0,
            engine_seconds,
            started.elapsed().as_secs_f64(),
        )
    }

    /// SGLang native `/generate`: `token_ids_logprob` names the answer slots by token id
    /// and `return_logprob` + `logprob_start_len=-1` return, per output position, the
    /// logprob of exactly those ids (`meta_info.output_token_ids_logprobs`). The engine
    /// expands images only for multimodal prompts, which this text contract does not use.
    ///
    /// Sampling parameters are sent explicitly and neutrally: SGLang defaults come from
    /// the model's `generation_config.json`, and any penalty or truncation applied before
    /// the logprobs would silently move the numbers this readout normalizes.
    fn fetch_sglang(&self, item: &Prepared, started: Instant) -> Result<Scored, String> {
        let payload = format!(
            "{{\"input_ids\": {}, \"sampling_params\": {{\"max_new_tokens\": 1, \"temperature\": 0.0, \"top_p\": 1.0, \"top_k\": -1, \"min_p\": 0.0, \"frequency_penalty\": 0.0, \"presence_penalty\": 0.0, \"repetition_penalty\": 1.0}}, \"return_logprob\": true, \"logprob_start_len\": -1, \"token_ids_logprob\": {}}}",
            dumps_int_list(&item.ids),
            dumps_int_list(&item.slots)
        );
        let send = Instant::now();
        let response = self
            .post_json("/generate", &payload)
            .map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let received = Instant::now();

        // meta_info.output_token_ids_logprobs is the exact-slot field; `_val`/`_idx` are
        // internal scheduler IPC names and never appear in the HTTP response.
        let option_logprobs = parse_sglang_token_ids(&response, &item.slots, &item.row_id)?;
        let probabilities =
            softmax(&option_logprobs).map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let readout = "SGLang native /generate token_ids_logprob: exact answer slots keyed by \
                       token_id, no top-N dependence and no fallback"
            .to_string();
        let engine_seconds = (received - send).as_secs_f64();
        scored(
            item,
            probabilities,
            option_logprobs,
            readout,
            false,
            0.0,
            engine_seconds,
            started.elapsed().as_secs_f64(),
        )
    }

    /// Wrap a prepared media row's messages into the request the chat route reads.
    ///
    /// `pin_slots` adds `logprob_token_ids` + `return_tokens_as_token_ids`, which is what makes
    /// the readout exact; the best-effort tier omits them and matches by decoded text instead.
    fn chat_media_request(
        served_model: &str,
        item: &Prepared,
        pin_slots: bool,
    ) -> Result<serde_json::Value, String> {
        let chat = item
            .chat
            .as_ref()
            .ok_or_else(|| format!("Row {}: media readout without chat messages", item.row_id))?;
        let mut request = serde_json::json!({
            "model": served_model,
            "messages": chat,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": true,
            "top_logprobs": MAX_LOGPROBS,
        });
        if pin_slots {
            request["logprob_token_ids"] = serde_json::json!(item.slots);
            request["return_tokens_as_token_ids"] = serde_json::json!(true);
        }
        Ok(request)
    }

    /// Turn a media chat response into a `Scored`, or explain exactly which slot was missing.
    fn media_decision(
        item: &Prepared,
        response: &serde_json::Value,
        readout: String,
        request_sha256: String,
        expect_pinned: bool,
        engine_seconds: f64,
        started: Instant,
    ) -> Result<Scored, String> {
        let top = response
            .pointer("/choices/0/logprobs/top_logprobs/0")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                format!(
                    "Row {}: the chat route returned no `logprobs.top_logprobs` object for the \
                     image request; `logprob_token_ids` on /v1/chat/completions is not \
                     supported by this build",
                    item.row_id
                )
            })?;
        let pinned = top.keys().any(|key| key.starts_with("token_id:"));
        // An exact-slot endpoint must not silently become the best-effort one: the probe
        // proved `logprob_token_ids` here, so a response without the pinned keys is a contract
        // violation (a rolling upgrade, a proxy stripping unknown fields), and the recorded
        // `serving_config` would otherwise name a recipe that never ran.
        if expect_pinned && !pinned {
            return Err(format!(
                "Row {}: this endpoint was probed as returning exact answer slots keyed by \
                 `token_id:`, but this response carried none; refusing rather than reading \
                 decoded text under the exact-slot serving_config",
                item.row_id
            ));
        }
        // Exact-slot answers under `token_id:<id>`; the top-N route under decoded text.
        let option_logprobs = if pinned {
            parse_vllm_token_ids(top, &item.slots, &item.row_id)?
        } else {
            let mut values = Vec::with_capacity(item.keys.len());
            for key in &item.keys {
                let value = top
                    .get(key)
                    .and_then(|value| value.as_f64())
                    .ok_or_else(|| {
                        format!(
                            "Row {}: answer slot {:?} is outside the returned top-N, and a media \
                         prompt has no /generative_scoring fallback (that route takes token ids \
                         and cannot carry an image). Refusing rather than renormalizing over \
                         the slots that happened to appear.",
                            item.row_id, key
                        )
                    })?;
                if !value.is_finite() {
                    return Err(format!(
                        "Row {}: non-finite logprob for slot {:?}",
                        item.row_id, key
                    ));
                }
                values.push(value);
            }
            values
        };
        let probabilities =
            softmax(&option_logprobs).map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let input_tokens = response
            .pointer("/usage/prompt_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        scored_media(
            item,
            probabilities,
            option_logprobs,
            input_tokens,
            request_sha256,
            readout,
            engine_seconds,
            started.elapsed().as_secs_f64(),
        )
    }

    /// vLLM media readout: `/v1/chat/completions` with an `image_url` part.
    ///
    /// The *chat* route is used because that is where a processor expands the image, so the
    /// prompt is server-tokenized and `input_ids_sha256` / the boundary proof do not apply.
    /// The answer-slot contract does: `item.slots` are still the exact A..P token ids.
    fn fetch_chat_media(&self, item: &Prepared, started: Instant) -> Result<Scored, String> {
        match self.media {
            MediaSupport::ChatExactSlot => self.fetch_chat_media_exact(item, started),
            MediaSupport::ChatTopN => self.fetch_chat_media_topn(item, started),
            MediaSupport::None => Err(format!(
                "Row {}: this endpoint was not probed as able to score an image; start it with \
                 --allow-media against a multimodal model",
                item.row_id
            )),
        }
    }

    /// The chat payload for a media request. Media parts are forwarded exactly as the caller
    /// sent them -- `Media::content_part` re-emits `Media::reference`, never a re-encode.
    fn chat_media_payload(&self, item: &Prepared, pin_slots: bool) -> Result<String, String> {
        Ok(Self::chat_media_request(&self.served_model, item, pin_slots)?.to_string())
    }

    fn fetch_chat_media_exact(&self, item: &Prepared, started: Instant) -> Result<Scored, String> {
        let payload = self.chat_media_payload(item, true)?;
        let send = Instant::now();
        let response = self
            .post_json("/v1/chat/completions", &payload)
            .map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let received = Instant::now();
        let readout = "vLLM /v1/chat/completions logprob_token_ids: exact answer slots keyed \
                       by token_id over a server-tokenized image prompt"
            .to_string();
        Self::media_decision(
            item,
            &response,
            readout,
            // The body that was actually sent, pin fields included: the two media tiers send
            // different bodies, so a hash of the messages alone could not tell them apart.
            media::sha256_hex(payload.as_bytes()),
            true,
            (received - send).as_secs_f64(),
            started,
        )
    }

    /// Best-effort media readout: the decoded-text top-N.
    ///
    /// A slot outside the returned top-N is a **hard failure**, because a media prompt has no
    /// `/generative_scoring` fallback -- that route takes token ids and cannot carry an image.
    fn fetch_chat_media_topn(&self, item: &Prepared, started: Instant) -> Result<Scored, String> {
        let payload = self.chat_media_payload(item, false)?;
        let send = Instant::now();
        let response = self
            .post_json("/v1/chat/completions", &payload)
            .map_err(|error| format!("Row {}: {}", item.row_id, error))?;
        let received = Instant::now();
        let readout = "vLLM /v1/chat/completions top-N over a server-tokenized image prompt: \
                       answer slots matched by decoded text, no fallback"
            .to_string();
        Self::media_decision(
            item,
            &response,
            readout,
            media::sha256_hex(payload.as_bytes()),
            false,
            (received - send).as_secs_f64(),
            started,
        )
    }

    /// Mirror of `_fetch_via_generative_scoring`.
    fn fetch_via_generative_scoring(&self, item: &Prepared) -> Result<Vec<f64>, String> {
        let mut probabilities = Vec::with_capacity(item.slots.len());
        for (index, slot) in item.slots.iter().enumerate() {
            // Put the wanted slot first; the remaining options are the normalizer for
            // the subset softmax.
            let mut order: Vec<u32> = vec![*slot];
            order.extend(
                item.slots
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, value)| *value),
            );
            let payload = format!(
                "{{\"model\": {}, \"query\": {}, \"items\": [\"\"], \"label_token_ids\": {}, \
                  \"apply_softmax\": true}}",
                serde_json::Value::String(self.served_model.clone()),
                dumps_int_list(&item.ids),
                dumps_int_list(&order)
            );
            let response = self
                .post_json("/generative_scoring", &payload)
                .map_err(|error| format!("Row {}: {}", item.row_id, error))?;
            let score = response
                .get("data")
                .and_then(|value| value.as_array())
                .and_then(|data| data.first())
                .and_then(|entry| entry.get("score"))
                .and_then(|value| value.as_f64())
                .ok_or_else(|| {
                    format!(
                        "Row {}: /generative_scoring returned no scores",
                        item.row_id
                    )
                })?;
            probabilities.push(score);
        }
        let total: f64 = probabilities.iter().sum();
        if probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || total <= 0.0
        {
            return Err(format!(
                "Row {}: /generative_scoring returned unusable scores {:?}",
                item.row_id, probabilities
            ));
        }
        // Renormalize defensively: each request normalizes over its own subset, so the
        // values should already sum to 1, but rounding could drift.
        Ok(probabilities.iter().map(|value| value / total).collect())
    }
}

/// What one endpoint's startup probe produced, before the cross-endpoint checks.
struct Probe {
    backend: Backend,
    tier_a: bool,
    media_support: MediaSupport,
    served_models: Vec<String>,
    model_root: Option<String>,
    version: Option<serde_json::Value>,
    max_model_len: Option<i64>,
}

/// Which dialect an endpoint speaks, when `--backend auto` left it open.
///
/// A positive answer decides: `GET /get_model_info` returning a `model_path` means
/// SGLang, and a successful answer on a vLLM-shaped route (`/version`, `/v1/models`)
/// means vLLM. A transport failure on every route is **not** a negative answer, so it is
/// reported instead of guessed -- otherwise one timeout would silently label an SGLang
/// server as vLLM and record that wrong dialect in the artifact.
fn detect_backend(agent: &ureq::Agent, url: &str) -> Result<Backend, String> {
    let model_info = get_json(agent, &format!("{}/get_model_info", url));
    if let Ok(info) = &model_info {
        if info
            .get("model_path")
            .and_then(|value| value.as_str())
            .is_some()
        {
            return Ok(Backend::Sglang);
        }
    }
    if get_json(agent, &format!("{}/version", url)).is_ok()
        || get_json(agent, &format!("{}/v1/models", url)).is_ok()
    {
        return Ok(Backend::Vllm);
    }
    Err(format!(
        "cannot determine the backend at {}: neither /get_model_info nor the vLLM routes \
         (/version, /v1/models) answered (last error: {})",
        url,
        match &model_info {
            Err(error) => error.to_string(),
            Ok(_) => "an HTTP 200 without a `model_path`".to_string(),
        }
    ))
}

fn probe_endpoint(
    agent: &ureq::Agent,
    url: &str,
    choice: BackendChoice,
    readout: Readout,
    media: MediaPolicy,
    tokenizer: &Tokenizer,
    probe_slots: &[u32],
) -> Result<Probe, String> {
    let backend = match choice {
        BackendChoice::Vllm => Backend::Vllm,
        BackendChoice::Sglang => Backend::Sglang,
        BackendChoice::Auto => detect_backend(agent, url)?,
    };
    match backend {
        Backend::Vllm => probe_vllm(agent, url, readout, media, tokenizer, probe_slots),
        Backend::Sglang => probe_sglang(agent, url, readout),
    }
}

fn probe_vllm(
    agent: &ureq::Agent,
    url: &str,
    readout: Readout,
    media: MediaPolicy,
    tokenizer: &Tokenizer,
    probe_slots: &[u32],
) -> Result<Probe, String> {
    // Both metadata routes are GET; a POST is answered 405. `/version` may legitimately
    // be absent (a proxy, an older build); a single endpoint tolerates that, a fleet does
    // not, because a fleet is required to prove its builds agree.
    let version = get_json(agent, &format!("{}/version", url)).ok();
    let models =
        get_json(agent, &format!("{}/v1/models", url)).map_err(|error| error.to_string())?;
    let entries = models
        .get("data")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    let served_models: Vec<String> = entries
        .iter()
        .filter_map(|entry| {
            entry
                .get("id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .collect();
    if served_models.is_empty() {
        return Err(format!(
            "No served models reported by {}; is the server up?",
            url
        ));
    }
    let model_root = entries
        .first()
        .and_then(|entry| entry.get("root"))
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let max_model_len = entries
        .first()
        .and_then(|entry| entry.get("max_model_len"))
        .and_then(|value| value.as_i64());
    // Refuse to guess: a non-positive context length is not a limit the client can enforce,
    // and accepting it would fail every row with a misleading per-row error.
    if let Some(context) = max_model_len {
        if context <= 0 {
            return Err(format!(
                "/v1/models reported an unusable max_model_len ({})",
                context
            ));
        }
    }
    // Tier A is opt-in on purpose: it changes the numbers on every row whose slot would
    // have missed the top-N, so the already-measured top-N path stays the default.
    let tier_a = match readout {
        Readout::ExactSlot => {
            let supported =
                probe_vllm_tier_a(agent, url, &served_models[0], tokenizer, probe_slots)
                    .map_err(|error| error.to_string())?;
            if !supported {
                return Err(
                    "--readout exact-slot was requested, but this build did not honor \
                     `logprob_token_ids` on /v1/completions (no `token_id:` keys came back). \
                     Use --readout auto (top-N + fallback) or a build that supports it."
                        .to_string(),
                );
            }
            true
        }
        _ => false,
    };
    Ok(Probe {
        backend: Backend::Vllm,
        tier_a,
        media_support: if media.enabled() {
            probe_vllm_media(agent, url, &served_models[0], probe_slots, media)?
        } else {
            MediaSupport::None
        },
        served_models,
        model_root,
        version,
        max_model_len,
    })
}

/// Ask the server whether it honors `logprob_token_ids` on `/v1/completions`.
///
/// This is a probe, not a version check: a build can accept the field and ignore it, and
/// older builds do exactly that. The signal is the response key space -- only a server
/// that actually honored the pinned ids answers under `token_id:<id>` keys, because that
/// key is produced by `return_tokens_as_token_ids` for the explicitly selected ids. A
/// server that ignored the field answers under decoded-text keys and is read as B-text.
fn probe_vllm_tier_a(
    agent: &ureq::Agent,
    url: &str,
    served_model: &str,
    tokenizer: &Tokenizer,
    probe_slots: &[u32],
) -> Result<bool, String> {
    let encoded = tokenizer
        .encode(".", false)
        .map(|value| value.get_ids().to_vec())
        .unwrap_or_default();
    let prompt: Vec<u32> = if encoded.is_empty() { vec![0] } else { encoded };
    let probe_slot = probe_slots.first().copied().unwrap_or(0);
    let payload = format!(
        "{{\"model\": {}, \"prompt\": {}, \"max_tokens\": 1, \"temperature\": 0, \"logprobs\": true, \"logprob_token_ids\": [{}], \"return_tokens_as_token_ids\": true}}",
        serde_json::Value::String(served_model.to_string()),
        dumps_int_list(&prompt),
        probe_slot
    );
    match post_json(agent, url, "/v1/completions", &payload) {
        Ok(response) => Ok(response
            .pointer("/choices/0/logprobs/top_logprobs/0")
            .and_then(|value| value.as_object())
            .map(|top| top.keys().any(|key| key.starts_with("token_id:")))
            .unwrap_or(false)),
        // The server understood the request and refused the field: a real negative signal
        // for this build. Only validation statuses qualify -- a 5xx, a 429, or a 401 says
        // nothing about whether the field is supported, and must not be recorded as
        // "this build does not support it".
        Err(CallError::Status { code, .. }) if code == 400 || code == 422 => Ok(false),
        Err(error) => Err(format!("Tier A probe on /v1/completions failed: {}", error)),
    }
}

/// Ask whether `/v1/chat/completions` can score an image, and at which tier.
///
/// The probe uses a real (1x1 PNG) data URI, because a text-only checkpoint or a build
/// without a vision path is expected to answer 400/422 -- a real negative answer. The tier
/// comes from the response key space, exactly as for `probe_vllm_tier_a`: `token_id:` keys
/// mean the pinned ids were honoured; any other object is the best-effort top-N route.
fn probe_vllm_media(
    agent: &ureq::Agent,
    url: &str,
    served_model: &str,
    probe_slots: &[u32],
    policy: MediaPolicy,
) -> Result<MediaSupport, String> {
    let probe_slot = probe_slots.first().copied().unwrap_or(0);
    // Read the capability off the request shape the selected tier will actually send: the
    // exact tier needs `logprob_token_ids`, while the best-effort tier must not require a
    // field a build might reject for reasons that have nothing to do with vision.
    let payload = |pinned: bool| {
        let mut request = serde_json::json!({
            "model": served_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "."},
                    {"type": "image_url", "image_url": {"url": media::PROBE_IMAGE_DATA_URI}},
                ],
            }],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": true,
            "top_logprobs": MAX_LOGPROBS,
        });
        if pinned {
            request["logprob_token_ids"] = serde_json::json!([probe_slot]);
            request["return_tokens_as_token_ids"] = serde_json::json!(true);
        }
        request.to_string()
    };
    let read_support = |response: &serde_json::Value| -> Result<MediaSupport, String> {
        match response
            .pointer("/choices/0/logprobs/top_logprobs/0")
            .and_then(|v| v.as_object())
        {
            Some(top) if top.keys().any(|key| key.starts_with("token_id:")) => {
                Ok(MediaSupport::ChatExactSlot)
            }
            Some(_) => Ok(MediaSupport::ChatTopN),
            None => Err(format!(
                "the media probe on {} answered without a `logprobs.top_logprobs` object, so \
                 this build's response cannot be read as a decision",
                url
            )),
        }
    };
    let support = match post_json(agent, url, "/v1/chat/completions", &payload(true)) {
        Ok(response) => read_support(&response)?,
        // A validation refusal is a real negative answer *for this request shape*: this build
        // will not take `logprob_token_ids` on the chat route. That says nothing yet about
        // whether it can score an image, so when the best-effort tier was opted into, ask the
        // question that tier actually sends before recording "no vision".
        Err(CallError::Status { code, .. })
            if (code == 400 || code == 422) && policy == MediaPolicy::TopN =>
        {
            match post_json(agent, url, "/v1/chat/completions", &payload(false)) {
                Ok(response) => read_support(&response)?,
                Err(CallError::Status { code, .. }) if code == 400 || code == 422 => {
                    eprintln!(
                        "weigh: media disabled at {}: the /v1/chat/completions image \
                         probe was refused without `logprob_token_ids` too (HTTP {})",
                        url, code
                    );
                    MediaSupport::None
                }
                Err(error) => {
                    return Err(format!(
                        "media probe on /v1/chat/completions failed: {}",
                        error
                    ))
                }
            }
        }
        // Only 400/422 qualify as a real negative answer; a 5xx or a transport failure says
        // nothing about that ability and must not be recorded as "unsupported".
        Err(CallError::Status { code, detail }) if code == 400 || code == 422 => {
            eprintln!(
                "weigh: media disabled at {}: the /v1/chat/completions image probe was \
                 refused (HTTP {}): {}",
                url,
                code,
                detail.chars().take(200).collect::<String>()
            );
            MediaSupport::None
        }
        Err(error) => {
            return Err(format!(
                "media probe on /v1/chat/completions failed: {}",
                error
            ))
        }
    };
    // `--allow-media` asks for a guarantee; a best-effort route cannot make it, so it is
    // reported as unavailable rather than silently accepted.
    if policy == MediaPolicy::ExactSlot && support == MediaSupport::ChatTopN {
        eprintln!(
            "weigh: media disabled at {}: --allow-media requires the exact-slot media \
             route, but this build answers under decoded text. Pass --allow-media-topn to \
             accept the best-effort route.",
            url
        );
        return Ok(MediaSupport::None);
    }
    Ok(support)
}

fn probe_sglang(agent: &ureq::Agent, url: &str, readout: Readout) -> Result<Probe, String> {
    if readout == Readout::TopN {
        return Err(
            "--readout top-n is not implemented for SGLang; its adapter is exact-slot only"
                .to_string(),
        );
    }
    let info = get_json(agent, &format!("{}/get_model_info", url))
        .map_err(|error| format!("/get_model_info: {}", error))?;
    let model_root = info
        .get("model_path")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    if model_root.is_none() {
        return Err("/get_model_info reported no `model_path`; not an SGLang server".to_string());
    }
    // The native /generate route does not need a model id, but identity does: prefer the
    // OpenAI-compatible model list, fall back to the checkpoint path.
    let listed: Vec<String> = match get_json(agent, &format!("{}/v1/models", url)) {
        Ok(models) => models
            .get("data")
            .and_then(|value| value.as_array())
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| {
                        entry
                            .get("id")
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    let served_models: Vec<String> = if listed.is_empty() {
        model_root.clone().into_iter().collect()
    } else {
        listed
    };
    let version = info.get("version").cloned();
    let max_model_len = info.get("context_length").and_then(|value| value.as_i64());
    if let Some(context) = max_model_len {
        if context <= 0 {
            return Err(format!(
                "/get_model_info reported an unusable context_length ({})",
                context
            ));
        }
    }
    // SGLang's native `token_ids_logprob` is exact-slot by construction, so it is Tier A.
    // Media is not wired for SGLang yet: its OpenAI route has no token-id readout and its
    // native `/generate` needs `input_ids`, which is exactly what an image destroys.
    Ok(Probe {
        backend: Backend::Sglang,
        tier_a: true,
        media_support: MediaSupport::None,
        served_models,
        model_root,
        version,
        max_model_len,
    })
}

/// Mirror of `direct._slot_ids`, as a free function so the startup probe can use it
/// before a `Client` exists. Stricter than the Python copy in exactly one place: a count
/// past the alphabet is refused rather than truncated to it, because a short slot vector is
/// a silently under-scored decision, not a token-id disagreement.
fn slot_ids_for(tokenizer: &Tokenizer, count: usize) -> Result<Vec<u32>, String> {
    // `.take(count)` alone would return a short vector, leaving a `Prepared` whose slots do
    // not line up with its options: a decision scored one value short, with nothing saying so.
    if count > LETTERS.len() {
        return Err(format!(
            "{} options exceed the {} answer slots",
            count,
            LETTERS.len()
        ));
    }
    let mut result = Vec::with_capacity(count);
    for letter in LETTERS.chars().take(count) {
        let encoded = tokenizer
            .encode(letter.to_string(), false)
            .map_err(|error| format!("Cannot encode slot {:?}: {}", letter, error))?;
        let ids = encoded.get_ids();
        if ids.len() != 1
            || tokenizer
                .decode(ids, false)
                .map_err(|error| error.to_string())?
                != letter.to_string()
        {
            return Err(format!(
                "Answer slot {:?} is not one exact round-trip token",
                letter
            ));
        }
        result.push(ids[0]);
    }
    let unique: HashSet<u32> = result.iter().copied().collect();
    if unique.len() != result.len() {
        return Err("Answer-slot tokens collide".to_string());
    }
    Ok(result)
}

pub struct Client {
    endpoints: Vec<Endpoint>,
    /// Round-robin cursor: each row is assigned an endpoint BEFORE its request is
    /// issued, so replicas receive an even share (measured 389/388 across 777 rows).
    ///
    /// This is NOT work-stealing, despite being the obvious place to claim it. A worker
    /// then blocks on the endpoint it was handed, so a slow replica still receives its
    /// full 1/N share and stretches the tail latency; balancing by completion speed
    /// would need one consumer per endpoint with its own queue instead of a shared
    /// cursor. The comment says round-robin because that is what the code does.
    next_endpoint: AtomicUsize,
    timeout: Duration,
    tokenizer: Tokenizer,
    max_tokens: usize,
    /// Smallest server-reported context limit across endpoints. A prompt longer than
    /// this would be truncated or rejected server-side, so the client refuses it first
    /// instead of relying on the server's truncation policy.
    context_limit: Option<i64>,
    /// Boundary verdicts already proven, keyed by the prompt's token-id tail and its slots.
    /// them. Without this every row would re-encode the full prompt once per option,
    /// which measured ~6.5 ms/row of pure overhead.
    boundary: Mutex<HashSet<(Vec<u32>, Vec<u32>)>>,
    slot_cache: Mutex<HashMap<usize, Vec<u32>>>,
    /// What every endpoint can do with an image. Decided once at startup by the probe, so a
    /// media request is never re-decided per row.
    media: MediaSupport,
}

impl Client {
    pub fn new(config: &Config) -> Result<(Client, ServerMetadata), String> {
        if config.urls.is_empty() {
            return Err("At least one --url is required".to_string());
        }
        let tokenizer_path = format!("{}/tokenizer.json", config.source.trim_end_matches('/'));
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("Cannot load tokenizer from {}: {}", tokenizer_path, error))?;

        // The chat template is resolved before any server traffic: a contradiction between
        // an explicit --template and the tokenizer's own chat_template must fail before a
        // single request is sent.
        let resolved_template = template::resolve(config.template, &config.source)?;

        let tokenizer_name = std::path::Path::new(&config.source)
            .file_name()
            .map(|n| n.to_owned());
        // One probe slot is enough to ask a vLLM server whether it honors
        // `logprob_token_ids`; the answer is a property of the build, not of the slots.
        let probe_slots = slot_ids_for(&tokenizer, 2)?;

        let fleet = config.urls.len() > 1;
        let mut endpoints = Vec::with_capacity(config.urls.len());
        let mut infos = Vec::with_capacity(config.urls.len());
        let mut backend: Option<Backend> = None;
        let mut tier_a: Option<bool> = None;
        let mut media_support: Option<MediaSupport> = None;
        let mut max_model_len: Option<i64> = None;
        let mut tokenizer_matches = None;

        // Compare the checkpoint's *name*, not its path: the same checkpoint is
        // routinely mounted at different paths on different hosts, and demanding path
        // equality would reject exactly the cross-machine fleet this feature exists for.
        // Known limit: two different checkpoints whose directories share a name compare equal.
        let checkpoint_name = |root: &Option<String>| {
            root.as_deref().and_then(|path| {
                std::path::Path::new(path)
                    .file_name()
                    .map(|name| name.to_owned())
            })
        };

        for (index, raw_url) in config.urls.iter().enumerate() {
            let url = raw_url.trim().trim_end_matches('/').to_string();
            let agent = ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(15))
                .timeout_read(config.timeout)
                .timeout_write(Duration::from_secs(120))
                .build();

            let probe = probe_endpoint(
                &agent,
                &url,
                config.backend,
                config.readout,
                config.media,
                &tokenizer,
                &probe_slots,
            )
            .map_err(|error| format!("endpoint {} ({}): {}", index, url, error))?;

            // A fleet must prove that every endpoint agrees on the fields it compares, so a
            // field missing on ANY endpoint -- including the first -- is refused instead of
            // being compared as `None == None`.
            if fleet && probe.backend == Backend::Vllm {
                if probe.version.is_none() {
                    return Err(format!(
                        "endpoint {} ({}): /version did not answer, so this endpoint's build \
                         cannot be compared with the rest of the fleet. Refusing to score \
                         rather than assume the endpoints agree.",
                        index, url
                    ));
                }
                if probe.max_model_len.is_none() {
                    return Err(format!(
                        "endpoint {} ({}): /v1/models reported no `max_model_len`, so its \
                         configuration cannot be compared with the rest of the fleet.",
                        index, url
                    ));
                }
            }

            // Homogeneity, checked rather than assumed. `None == None` is never a pass
            // for a field the fleet is required to agree on.
            if let Some(first) = infos.first() {
                let first: &EndpointInfo = first;
                if Some(probe.backend) != backend {
                    return Err(format!(
                        "Fleet mismatch at endpoint {} ({}): this endpoint speaks {} but \
                         endpoint 0 speaks {}. A mixed fleet would silently mix two readouts \
                         under one artifact.",
                        index,
                        url,
                        probe.backend.as_str(),
                        backend.map(Backend::as_str).unwrap_or("?")
                    ));
                }
                if first.served_models.first() != probe.served_models.first() {
                    return Err(format!(
                        "Fleet mismatch at endpoint {} ({}): first served model id is {:?} but \
                         endpoint 0 reported {:?}. Requests are sent with that id, so the \
                         endpoints must agree on it.",
                        index,
                        url,
                        probe.served_models.first(),
                        first.served_models.first()
                    ));
                }
                if checkpoint_name(&first.model_root).is_none()
                    || checkpoint_name(&first.model_root) != checkpoint_name(&probe.model_root)
                {
                    return Err(format!(
                        "Fleet mismatch at endpoint {} ({}): checkpoint directory is {:?} but \
                         endpoint 0 reported {:?}. Scoring across different checkpoints would \
                         mix two models' probabilities under one artifact. If these are the \
                         same checkpoint on different mounts, the served root cannot prove \
                         it -- align the directory names or verify tokenizer fingerprints \
                         by hand.",
                        index, url, probe.model_root, first.model_root
                    ));
                }
                if tier_a != Some(probe.tier_a) {
                    return Err(format!(
                        "Fleet mismatch at endpoint {} ({}): readout tier differs (Tier A={}). \
                         Every endpoint must reach the same readout, or a row's value would \
                         also depend on which replica served it.",
                        index, url, probe.tier_a
                    ));
                }
                if media_support != Some(probe.media_support) {
                    return Err(format!(
                        "Fleet mismatch at endpoint {} ({}): media support differs ({}). Every \
                         endpoint must reach the same media readout, or a media row's \
                         provenance would depend on which replica served it.",
                        index,
                        url,
                        probe.media_support.as_str()
                    ));
                }
                if probe.backend == Backend::Vllm {
                    // Different builds batch and quantize differently, and this deployment
                    // is already measured not to be bit-reproducible across configurations.
                    // Both sides are proven present by the fleet check above.
                    if first.version != probe.version {
                        return Err(format!(
                            "Fleet mismatch at endpoint {} ({}): server version is {:?} but \
                             endpoint 0 reported {:?}.",
                            index, url, probe.version, first.version
                        ));
                    }
                }
            } else {
                backend = Some(probe.backend);
                tier_a = Some(probe.tier_a);
                media_support = Some(probe.media_support);
            }

            // The tokenizer guard is checked per endpoint. With one endpoint this is the
            // Python client's behaviour exactly; with N it is the same refusal applied N
            // times, because "the models are identical" must be verified, not assumed.
            if let Some(served_root) = &probe.model_root {
                let same =
                    std::path::Path::new(served_root).file_name() == tokenizer_name.as_deref();
                tokenizer_matches = Some(tokenizer_matches.unwrap_or(true) && same);
                if !same && !config.allow_tokenizer_mismatch {
                    return Err(format!(
                        "Tokenizer mismatch at endpoint {} ({}): --model is {:?} but the server \
                         reports it serves {:?}. This client tokenizes the prompt and the answer \
                         slots locally, so a different tokenizer yields plausible but wrong \
                         probabilities. Point --model at the served checkpoint, or pass \
                         --allow-tokenizer-mismatch.",
                        index, url, config.source, served_root
                    ));
                }
            }

            max_model_len = match (max_model_len, probe.max_model_len) {
                (Some(previous), Some(current)) if previous != current => {
                    return Err(format!(
                        "Fleet mismatch at endpoint {} ({}): max_model_len is {} but endpoint 0 \
                         reported {}. All endpoints must serve the same configuration.",
                        index, url, current, previous
                    ))
                }
                (Some(previous), _) => Some(previous),
                (None, current) => current,
            };

            infos.push(EndpointInfo {
                url: url.clone(),
                served_models: probe.served_models.clone(),
                model_root: probe.model_root,
                version: probe.version,
            });
            endpoints.push(Endpoint {
                url,
                agent,
                served_model: probe.served_models[0].clone(),
                backend: probe.backend,
                tier_a: probe.tier_a,
                media: probe.media_support,
            });
        }

        let client = Client {
            endpoints,
            next_endpoint: AtomicUsize::new(0),
            timeout: config.timeout,
            tokenizer,
            max_tokens: config.max_tokens,
            context_limit: max_model_len,
            boundary: Mutex::new(HashSet::new()),
            slot_cache: Mutex::new(HashMap::new()),
            media: media_support.unwrap_or(MediaSupport::None),
        };
        Ok((
            client,
            ServerMetadata {
                backend: backend.unwrap_or(Backend::Vllm),
                tier_a: tier_a.unwrap_or(false),
                media_support: media_support.unwrap_or(MediaSupport::None),
                endpoints: infos,
                max_model_len,
                tokenizer_matches,
                prompt_template: resolved_template.template,
                prompt_template_source: resolved_template.source,
            },
        ))
    }

    /// Score one prepared row, returning the endpoint index that served it.
    ///
    /// The endpoint is chosen once per row, by round-robin, and then used for the row's
    /// entire lifecycle, so the readout and its fallback cannot straddle two servers.
    pub fn fetch_any(&self, item: &Prepared) -> Result<(Scored, usize), ReadoutError> {
        let index = self.next_endpoint.fetch_add(1, Ordering::Relaxed) % self.endpoints.len();
        let scored = self.endpoints[index].fetch(item, self.timeout)?;
        Ok((scored, index))
    }

    /// Mirror of `direct._slot_ids`, memoized per option count.
    fn slot_ids(&self, count: usize) -> Result<Vec<u32>, String> {
        if let Some(cached) = self.slot_cache.lock().unwrap().get(&count) {
            return Ok(cached.clone());
        }
        let result = slot_ids_for(&self.tokenizer, count)?;
        self.slot_cache
            .lock()
            .unwrap()
            .insert(count, result.clone());
        Ok(result)
    }

    /// Prepare a text readout: tokenize the caller's prompt, prove the answer boundary, and
    /// bind every option to one slot token.
    ///
    /// The caller owns the prompt text -- this library renders no chat template of its own.
    /// What is proven here is the property the readout depends on: appending any slot letter
    /// appends exactly one token, so the position the server scores is the position meant.
    pub fn prepare(
        &self,
        row_id: &str,
        prompt: &str,
        option_ids: &[String],
    ) -> Result<Prepared, String> {
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| format!("Cannot encode prompt for {}: {}", row_id, error))?;
        let ids: Vec<u32> = encoded.get_ids().to_vec();
        // The server's own context limit is part of the no-truncation contract: a prompt inside
        // `--max-tokens` but above `max_model_len` would be truncated or rejected server-side,
        // and which one happens is the server's policy, not ours. One token is reserved for the
        // single scored position, since every readout sends `max_tokens: 1`.
        let limit = context_limit_for(self.max_tokens, self.context_limit);
        if ids.is_empty() {
            return Err(format!("{}: prompt tokenizes to zero tokens", row_id));
        }
        if ids.len() > limit {
            return Err(format!(
                "{}: {} input tokens exceed limit {}; no truncation allowed",
                row_id,
                ids.len(),
                limit
            ));
        }
        let slots = self.slot_ids(option_ids.len())?;

        let key = (
            ids[ids.len().saturating_sub(BOUNDARY_WINDOW)..].to_vec(),
            slots.clone(),
        );
        let already_proven = self.boundary.lock().unwrap().contains(&key);
        if !already_proven {
            // The answer must append one exact token: encode(prompt + letter) has to be
            // ids + [slot]. If the boundary merges characters the server would score a
            // different position than the one the caller believes it is reading.
            for (letter, token) in LETTERS.chars().zip(slots.iter()) {
                let mut with_letter = prompt.to_string();
                with_letter.push(letter);
                let appended = self
                    .tokenizer
                    .encode(with_letter.as_str(), false)
                    .map_err(|error| format!("Cannot encode boundary for {}: {}", letter, error))?;
                let mut expected = ids.clone();
                expected.push(*token);
                if appended.get_ids() != expected.as_slice() {
                    return Err(format!(
                        "Answer boundary changes tokenization for slot {}",
                        letter
                    ));
                }
            }
            // A concurrent row may have proven it first; inserting again is harmless.
            let mut cache = self.boundary.lock().unwrap();
            if cache.len() >= BOUNDARY_CACHE_CAP {
                cache.clear();
            }
            cache.insert(key);
        }

        let keys = self.slot_keys(&slots)?;

        Ok(Prepared {
            row_id: row_id.to_string(),
            option_ids: option_ids.to_vec(),
            ids,
            slots,
            keys,
            prompt_hash: media::sha256_hex(prompt.as_bytes()),
            chat: None,
            media: Vec::new(),
        })
    }

    /// The decoded text of each answer slot, with the uniqueness check the top-N media
    /// matcher depends on. Shared by `prepare` and `prepare_media` so the check cannot drift.
    fn slot_keys(&self, slots: &[u32]) -> Result<Vec<String>, String> {
        let keys: Vec<String> = slots
            .iter()
            .map(|slot| {
                self.tokenizer
                    .decode(&[*slot], false)
                    .map_err(|error| format!("Cannot decode slot {}: {}", slot, error))
            })
            .collect::<Result<_, _>>()?;
        let unique: HashSet<&String> = keys.iter().collect();
        if unique.len() != keys.len() {
            return Err(
                "Option slots decode to duplicate strings; cannot match by text".to_string(),
            );
        }
        Ok(keys)
    }

    /// Prepare a media readout: one text part carrying the caller's prompt, then the images in
    /// the order given, scored through the backend's chat route.
    ///
    /// The caller supplies `system` and `text_payload` -- this library invents no prompt. There
    /// is **no** boundary proof and **no** local `ids`, because the backend's processor expands
    /// an image into placeholder tokens this client cannot see. What is still proven locally is
    /// the answer-slot contract: `slots` are the exact token ids of the slot letters.
    pub fn prepare_media(
        &self,
        row_id: &str,
        system: &str,
        text_payload: &str,
        media: &[Media],
        option_ids: &[String],
    ) -> Result<Prepared, String> {
        if self.media == MediaSupport::None {
            return Err(
                "this endpoint was not probed as able to score an image; start the server \
                 against a multimodal model and opt in"
                    .to_string(),
            );
        }
        if media.is_empty() {
            return Err("a media readout needs at least one media part".to_string());
        }
        let slots = self.slot_ids(option_ids.len())?;
        let keys = self.slot_keys(&slots)?;

        // One text part carries the caller's prompt; the media parts follow it in the order
        // given, forwarded byte for byte (`Media::content_part`).
        let mut content: Vec<serde_json::Value> =
            vec![serde_json::json!({"type": "text", "text": text_payload})];
        for item in media {
            content.push(item.content_part());
        }
        let chat = serde_json::json!([
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]);
        // One provenance object per image: what the caller declared and, where this process
        // could see the bytes, the fingerprint of exactly those bytes.
        let media: Vec<serde_json::Value> = media.iter().map(Media::provenance).collect();
        Ok(Prepared {
            row_id: row_id.to_string(),
            option_ids: option_ids.to_vec(),
            ids: Vec::new(),
            slots,
            keys,
            prompt_hash: String::new(),
            chat: Some(chat),
            media,
        })
    }
}

fn get_json(agent: &ureq::Agent, url: &str) -> Result<serde_json::Value, CallError> {
    match agent.get(url).call() {
        Ok(response) => response
            .into_string()
            .map_err(|error| CallError::Transport(format!("Cannot read {}: {}", url, error)))
            .and_then(|text| {
                serde_json::from_str(&text).map_err(|error| {
                    CallError::Transport(format!("{} returned invalid JSON: {}", url, error))
                })
            }),
        Err(ureq::Error::Status(code, response)) => {
            let detail = response.into_string().unwrap_or_default();
            let detail: String = detail.chars().take(500).collect();
            Err(CallError::Status { code, detail })
        }
        Err(error) => Err(CallError::Transport(format!(
            "Cannot reach {}: {}",
            url, error
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- pure response parsers ------------------------------------------------

    #[test]
    fn vllm_tier_a_reads_slots_by_token_id() {
        let top = json!({"token_id:1": -0.5, "token_id:2": -1.5, "token_id:99": -3.0});
        let top = top.as_object().unwrap();
        let got = parse_vllm_token_ids(top, &[1, 2], "row").unwrap();
        assert_eq!(got, vec![-0.5, -1.5]);
    }

    #[test]
    fn vllm_tier_a_rejects_a_missing_slot_instead_of_substituting() {
        let top = json!({"token_id:1": -0.5});
        let top = top.as_object().unwrap();
        let error = parse_vllm_token_ids(top, &[1, 2], "row").unwrap_err();
        assert!(error.contains("omitted requested slot 2"), "{}", error);
    }

    #[test]
    fn vllm_tier_a_rejects_a_non_finite_logprob() {
        // null parses as no value, so it must be reported, not silently skipped.
        let top = json!({"token_id:7": null});
        let top = top.as_object().unwrap();
        assert!(parse_vllm_token_ids(top, &[7], "row").is_err());
    }

    #[test]
    fn sglang_reads_the_external_meta_info_key_by_token_id() {
        // Shape from tokenizer_manager: a list per output position whose entries are
        // (logprob, token_id, token_text|null).
        let response = json!({
            "meta_info": {"output_token_ids_logprobs": [[[-0.25, 11, "A"], [-2.0, 22, null]]]}
        });
        let got = parse_sglang_token_ids(&response, &[22, 11], "row").unwrap();
        assert_eq!(got, vec![-2.0, -0.25]);
    }

    #[test]
    fn sglang_without_the_key_is_an_error_not_a_zero() {
        let response = json!({"meta_info": {"output_token_logprobs": [[-0.25, 11, "A"]]}});
        let error = parse_sglang_token_ids(&response, &[11], "row").unwrap_err();
        assert!(error.contains("output_token_ids_logprobs"), "{}", error);
    }

    #[test]
    fn sglang_rejects_a_missing_slot() {
        let response = json!({"meta_info": {"output_token_ids_logprobs": [[[-0.25, 11, "A"]]]}});
        let error = parse_sglang_token_ids(&response, &[11, 22], "row").unwrap_err();
        assert!(error.contains("omitted requested slot 22"), "{}", error);
    }

    // -- pure decision helpers ------------------------------------------------

    #[test]
    fn softmax_is_a_distribution_over_the_slots() {
        let probabilities = softmax(&[-1.0, -2.0, -3.0]).unwrap();
        let total: f64 = probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
    }

    /// The minimal tokenizer the slot contract needs: one distinct round-trip token per
    /// letter, which a tiny vocab gives without a model download.
    fn letter_tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".to_string(), json!(0));
        for (index, letter) in LETTERS.chars().enumerate() {
            vocab.insert(letter.to_string(), json!(index + 1));
        }
        let model = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
        });
        Tokenizer::from_bytes(model.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn more_options_than_answer_slots_is_refused_not_truncated() {
        let tokenizer = letter_tokenizer();
        // The whole alphabet binds: every letter is one distinct round-trip token.
        let slots = slot_ids_for(&tokenizer, LETTERS.len()).unwrap();
        assert_eq!(slots.len(), LETTERS.len());
        assert_eq!(
            slots.iter().copied().collect::<HashSet<u32>>().len(),
            LETTERS.len()
        );
        // One past the alphabet has to say so rather than hand back a short vector that
        // leaves a row with fewer scored slots than options.
        let error = slot_ids_for(&tokenizer, LETTERS.len() + 1).unwrap_err();
        assert!(error.contains("answer slots"), "{}", error);
    }

    #[test]
    fn serving_config_distinguishes_backend_and_tier() {
        let values = [
            serving_config_for(Backend::Vllm, false),
            serving_config_for(Backend::Vllm, true),
            serving_config_for(Backend::Sglang, true),
        ];
        let unique: HashSet<&str> = values.iter().copied().collect();
        assert_eq!(unique.len(), 3, "recipes must not collide: {:?}", values);
        assert_eq!(
            serving_config_for(Backend::Vllm, false),
            "vllm-openai-completions-subset-softmax-v1"
        );
    }

    #[test]
    fn media_serving_config_is_distinct_from_every_text_recipe() {
        let text = [
            serving_config_for(Backend::Vllm, false),
            serving_config_for(Backend::Vllm, true),
            serving_config_for(Backend::Sglang, true),
        ];
        let media = [
            media_serving_config_for(Backend::Vllm, MediaSupport::ChatExactSlot).unwrap(),
            media_serving_config_for(Backend::Vllm, MediaSupport::ChatTopN).unwrap(),
        ];
        let unique: HashSet<&str> = text.iter().chain(media.iter()).copied().collect();
        assert_eq!(unique.len(), 5, "media and text recipes must not collide");
        // The pairs that cannot produce a media readout have no recipe at all, rather than a
        // string that looks like one.
        assert_eq!(
            media_serving_config_for(Backend::Vllm, MediaSupport::None),
            None
        );
        assert_eq!(
            media_serving_config_for(Backend::Sglang, MediaSupport::ChatTopN),
            None
        );
    }

    /// A media `Prepared` with just enough shape to exercise the readout helpers.
    fn media_prepared(slots: Vec<u32>, keys: Vec<String>, reference: &str) -> Prepared {
        let chat = serde_json::json!([
            {"role": "system", "content": "criterion"},
            {"role": "user", "content": [
                {"type": "text", "text": "{\"evidence\":\"\"}"},
                {"type": "image_url", "image_url": {"url": reference}},
            ]},
        ]);
        Prepared {
            row_id: "row".to_string(),
            option_ids: vec!["yes".to_string(), "no".to_string()],
            ids: Vec::new(),
            slots,
            keys,
            prompt_hash: String::new(),
            chat: Some(chat),
            media: Vec::new(),
        }
    }

    #[test]
    fn a_media_request_forwards_the_image_reference_verbatim() {
        // Pass-through: whatever the caller put in `image_url.url` is what the backend gets.
        let reference = crate::media::PROBE_IMAGE_DATA_URI;
        let item = media_prepared(
            vec![1, 2],
            vec!["A".to_string(), "B".to_string()],
            reference,
        );
        let request = Endpoint::chat_media_request("served", &item, true).unwrap();
        assert_eq!(
            request["messages"][1]["content"][1]["image_url"]["url"],
            reference
        );
        assert_eq!(request["logprob_token_ids"], serde_json::json!([1, 2]));
        assert_eq!(
            request["return_tokens_as_token_ids"],
            serde_json::json!(true)
        );
        assert_eq!(request["max_tokens"], serde_json::json!(1));
    }

    #[test]
    fn a_media_exact_slot_response_is_read_by_token_id() {
        let item = media_prepared(
            vec![1, 2],
            vec!["A".to_string(), "B".to_string()],
            "data:image/png;base64,AA",
        );
        let response = serde_json::json!({
            "choices": [{"logprobs": {"top_logprobs": [{"token_id:1": -0.5, "token_id:2": -1.5}]}}],
            "usage": {"prompt_tokens": 42}
        });
        let scored = Endpoint::media_decision(
            &item,
            &response,
            "readout".to_string(),
            "body-sha".to_string(),
            true,
            0.0,
            Instant::now(),
        )
        .unwrap();
        assert_eq!(scored.input_tokens, Some(42));
        assert_eq!(scored.modality, "text+image");
        assert!(scored.server_tokenized);
        // The two text proofs are gone; the request hash replaces them.
        assert_eq!(scored.input_ids_sha256, None);
        assert_eq!(scored.prompt_sha256, None);
        assert!(scored.request_sha256.is_some());
        assert!((scored.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_media_slot_outside_the_top_n_is_a_hard_failure_not_a_renormalization() {
        let item = media_prepared(
            vec![1, 2],
            vec!["A".to_string(), "B".to_string()],
            "data:image/png;base64,AA",
        );
        // Only one of the two declared slots came back, under decoded-text keys.
        let response =
            serde_json::json!({"choices": [{"logprobs": {"top_logprobs": [{"A": -0.5}]}}]});
        let error = Endpoint::media_decision(
            &item,
            &response,
            "readout".to_string(),
            "body-sha".to_string(),
            false,
            0.0,
            Instant::now(),
        )
        .unwrap_err();
        assert!(error.contains("outside the returned top-N"), "{}", error);
        assert!(
            error.contains("no /generative_scoring fallback"),
            "{}",
            error
        );
    }

    #[test]
    fn a_media_response_without_top_logprobs_is_an_error_not_a_zero() {
        let item = media_prepared(
            vec![1, 2],
            vec!["A".to_string(), "B".to_string()],
            "data:image/png;base64,AA",
        );
        let response = serde_json::json!({"choices": [{}]});
        assert!(Endpoint::media_decision(
            &item,
            &response,
            "readout".to_string(),
            "body-sha".to_string(),
            false,
            0.0,
            Instant::now()
        )
        .is_err());
    }

    /// The exact media tier must not silently degrade into the best-effort one: the probe
    /// proved `logprob_token_ids`, so a response without the pinned keys means the recorded
    /// `serving_config` would name a recipe that did not run.
    #[test]
    fn an_exact_media_response_without_pinned_keys_is_refused() {
        let item = media_prepared(
            vec![1, 2],
            vec!["A".to_string(), "B".to_string()],
            "data:image/png;base64,AA",
        );
        // Both slots present, but under decoded-text keys: the best-effort shape.
        let response = serde_json::json!({"choices": [{"logprobs": {"top_logprobs": [{"A": -0.5, "B": -1.5}]}}]});
        let error = Endpoint::media_decision(
            &item,
            &response,
            "readout".to_string(),
            "body-sha".to_string(),
            true,
            0.0,
            Instant::now(),
        )
        .unwrap_err();
        assert!(error.contains("keyed by `token_id:`"), "{}", error);
        // The same response is legitimate on the endpoint that was probed for top-N.
        assert!(Endpoint::media_decision(
            &item,
            &response,
            "readout".to_string(),
            "body-sha".to_string(),
            false,
            0.0,
            Instant::now(),
        )
        .is_ok());
    }

    #[test]
    fn context_limit_reserves_one_token_for_the_scored_position() {
        // prompt + 1 <= max_model_len: a prompt exactly at the context length is rejected.
        assert_eq!(context_limit_for(4096, Some(2048)), 2047);
        assert_eq!(context_limit_for(100, Some(2048)), 100);
        assert_eq!(context_limit_for(4096, None), 4096);
        assert_eq!(context_limit_for(4096, Some(0)), 0);
    }

    #[test]
    fn flag_parsers_reject_unknown_values() {
        assert!(BackendChoice::parse("sglang").is_ok());
        assert!(BackendChoice::parse("vllm").is_ok());
        assert!(BackendChoice::parse("tgi").is_err());
        assert!(Readout::parse("exact-slot").is_ok());
        assert!(Readout::parse("top-n").is_ok());
        assert!(Readout::parse("legacy").is_err());
    }

    // -- startup probing over real HTTP --------------------------------------

    /// Serve a fixed route table on an ephemeral port; unmatched paths answer 404.
    fn mock_server(routes: &'static [(&'static str, u16, &'static str)]) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..8 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut buffer = [0u8; 8192];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]);
                let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (status, body) = routes
                    .iter()
                    .find(|(route, _, _)| *route == path)
                    .map(|(_, status, body)| (*status, *body))
                    .unwrap_or((404, "{\"error\":\"not found\"}"));
                let response = format!(
                    "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        format!("http://{}", address)
    }

    #[test]
    fn auto_detects_sglang_from_get_model_info() {
        let url = mock_server(&[("/get_model_info", 200, "{\"model_path\":\"/models/gemma\"}")]);
        assert_eq!(
            detect_backend(&ureq::agent(), &url).unwrap(),
            Backend::Sglang
        );
    }

    #[test]
    fn auto_detects_vllm_from_a_positive_answer_on_a_vllm_route() {
        // /get_model_info is absent (404); /version answers, which is a real answer and
        // therefore a real vLLM signal.
        let url = mock_server(&[("/version", 200, "{\"version\":\"0.23.0\"}")]);
        assert_eq!(detect_backend(&ureq::agent(), &url).unwrap(), Backend::Vllm);
    }

    #[test]
    fn auto_refuses_to_guess_a_backend_when_nothing_answers() {
        // Bind then drop, so the port is closed: every route fails at the transport layer.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let url = format!("http://{}", address);
        let error = detect_backend(&ureq::agent(), &url).unwrap_err();
        assert!(error.contains("cannot determine the backend"), "{}", error);
    }

    #[test]
    fn transport_failures_are_not_http_statuses() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buffer = [0u8; 4096];
                let _ = stream.read(&mut buffer);
                let _ = stream.write_all(
                    b"HTTP/1.1 400 X\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                );
            }
        });
        let url = format!("http://{}", address);
        match get_json(&ureq::agent(), &url) {
            Err(CallError::Status { code, .. }) => assert_eq!(code, 400),
            other => panic!("expected a Status error, got {:?}", other),
        }
    }
}
