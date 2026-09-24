// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The answer-slot model: `N` options bound to `N` distinct single-token letters.
//!
//! A decision is read out as a distribution over a handful of *answer slots*. Binding each
//! option to an uppercase letter keeps the readout constructive: the option text can be any
//! string, but what the model is asked for is one token from a set the client chose, so the
//! client can request that token's logprob **by id** instead of searching a top-N list. That
//! is the whole reason this crate exists: `logprob_token_ids` with `return_tokens_as_token_ids`
//! names every slot, so no slot can be missing from the response.
//!
//! The letters are only a naming device. Three properties are load-bearing, and every one of
//! them is checked on the tokenizer at startup rather than assumed:
//!
//! 1. each letter must be **exactly one token**,
//! 2. that token must round-trip (encode -> decode -> the same letter),
//! 3. the letters must be **distinct** tokens, so a slot id identifies one option.

/// The option letters this crate can bind, in slot order: `A` is slot 0, `B` is slot 1, ...
///
/// Sixteen is a deliberate ceiling rather than a limit of the protocol: the contract is only
/// constructive while every letter is a single round-trip token, and that is a property of
/// the tokenizer, not of the letter set. Sixteen covers the decision shapes this crate is
/// built for; a wider set would need the per-tokenizer check before it could be promised.
pub const LETTERS: &str = "ABCDEFGHIJKLMNOP";
