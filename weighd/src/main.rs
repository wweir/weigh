// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `weighd`: SemIf's decision readout as an OpenAI-shaped HTTP service.
//!
//! This binary is one consumer of the `weigh` library. It owns SemIf's contract -- the
//! `direct-options-v1` prompt, the criterion/evidence fold, and the `semif` response block --
//! and the HTTP surface. Everything that decides *which token ids are read and what the answer
//! means* lives in the library.
//!
//! This binary serves, and only serves. The batch JSONL scorer that used to live here was
//! removed deliberately: the service form is the supported interface, and the readout the
//! service can answer faithfully (exact-slot) is the one it enforces at startup.
//!
//! Everything the process needs is decided before it binds a port: the backend dialect and
//! readout tier (probed), the checkpoint identity (verified), and the chat template (resolved
//! against the tokenizer's own `chat_template`). A failure there exits non-zero instead of
//! serving numbers the artifact could not describe.

mod prompt;
mod serve;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = serve::run(&argv) {
        eprintln!("weighd: error: {}", error);
        std::process::exit(1);
    }
}
