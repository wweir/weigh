// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The crate documentation IS the README, so the code a reader sees is the code `cargo test`
// compiles as a doc test: the documented API and the tested API cannot drift apart.
#![doc = include_str!("../README.md")]

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
