// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Exact per-slot logprobs from an OpenAI-compatible inference server.
//!
//! This library answers one question: *what is the logprob of each of these `N` tokens at this
//! position?* It asks a vLLM (>= 0.26.0) or SGLang server for the exact logprobs of
//! caller-named token ids (`logprob_token_ids` / `token_ids_logprob`), normalizes them into a
//! subset softmax over the declared slots, and returns the provenance needed to attribute the
//! number.
//!
//! ## Why not just read `top_logprobs`?
//!
//! A top-N readout can only report slots that happen to land in the top N. Whether a given slot
//! is there depends on the prompt and on the server's state, so a missing slot has to be
//! recovered some other way -- and any recovery is a **different readout**, which means the
//! numbers stop being comparable within one result set. Naming the slots by token id removes
//! the question: every declared slot comes back by construction, or the call fails.
//!
//! ## The library renders no prompt
//!
//! [`Client::prepare`] takes the finished prompt text and the option labels;
//! [`Client::prepare_media`] takes the system text, the text payload, and the images. Deciding
//! what the model is asked is the caller's contract, not this crate's. What this crate owns is
//! the part that has to be exact: which token ids get read, and what the answer means.
//!
//! ## What is checked at startup rather than assumed
//!
//! [`Client::new`] loads the tokenizer and probes every endpoint, and refuses to run when it
//! cannot prove the readout it would be using: the backend dialect, whether the build honours
//! `logprob_token_ids`, whether it can score an image, and -- across a fleet -- that every
//! endpoint agrees on the fields a result set would pool. A failure there is an error, not a
//! silent downgrade.

pub mod client;
pub mod media;
pub mod pyjson;
pub mod schema;
pub mod slots;
pub mod template;

pub use client::{
    media_serving_config_for, serving_config_for, Backend, BackendChoice, Client, Config,
    EndpointInfo, MediaPolicy, MediaSupport, Prepared, Readout, ReadoutError, Scored,
    ServerMetadata,
};
pub use media::{sha256_hex, Content, Media, MediaLimits};
pub use schema::{DecisionSchema, SchemaError};
pub use slots::LETTERS;
pub use template::{ChatTemplate, TemplateChoice};
