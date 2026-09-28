// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Multimodal request parts: parsing OpenAI `image_url` content, decoding `data:` URIs, and
//! fingerprinting the exact bytes this server hands to the backend.
//!
//! ## Why a media part is not "one more content type"
//!
//! The text contract is deliberately narrow: the client renders the chat template, tokenizes
//! it locally, and proves the answer boundary ([`crate::client::Client::prepare`]). An image
//! breaks all three, because the backend's processor expands it into placeholder tokens the
//! client never sees. A request that carries media therefore switches to a **degraded**
//! readout whose guarantees are recorded per response.
//!
//! ## Pass-through is the rule for what reaches the backend
//!
//! `Media::reference` is the caller's `image_url.url` **byte for byte**, and that exact
//! string is what the readout forwards. Nothing is re-encoded, re-serialized, or rewritten
//! to a canonical form: a re-encode is a silent transformation of the bytes the model sees,
//! and the whole point of recording a sha256 is that the recorded value describes what was
//! actually sent.
//!
//! Decoding is a **side channel**, used only for two things the design requires -- a size
//! ceiling and a byte fingerprint -- and it never feeds the forwarded reference. The MIME is
//! recorded as the caller declared it, not judged here: whether a type is a usable image is
//! the backend processor's decision, and inventing a second allow-list would reject images
//! a newer checkpoint handles while adding nothing the backend does not already enforce.
//!
//! ## Refusal, never silent degradation
//!
//! Every way an image can be unacceptable is reported by name -- an unparseable `data:` URI,
//! an oversized payload, a remote URL (SSRF), media in the criterion -- because the
//! alternative (dropping the image and scoring the text) answers a question the caller did
//! not ask. A caller that folds messages into a prompt applies the same rule to unsupported parts.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// What the operator allowed a request to carry.
///
/// `allow_media` is the master switch; when it is false an `image_url` part is refused
/// exactly as it was before this module existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    pub allow_media: bool,
    /// Accept `http(s)://` references. Off by default: the backend would fetch an arbitrary
    /// URL on the caller's behalf (SSRF), and this server can never hash the bytes it gets.
    pub allow_remote: bool,
    pub max_images: usize,
    /// Per-image decoded ceiling. The request body limit is a separate, coarser guard.
    pub max_media_bytes: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        // The disabled default: media off, and a limit no real image reaches. Used by tests
        // that only exercise the text path.
        MediaLimits {
            allow_media: false,
            allow_remote: false,
            max_images: 1,
            max_media_bytes: 8 * 1024 * 1024,
        }
    }
}

/// One image the caller attached to the evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Media {
    pub mime: String,
    /// sha256 of the decoded bytes. `None` for a remote URL this server never fetched --
    /// the fingerprint would be of the URL, not of what the backend actually loads.
    pub sha256: Option<String>,
    /// Decoded size in bytes. `None` for a remote URL.
    pub size: Option<usize>,
    /// The exact `image_url.url` string handed to the backend, byte for byte. This is the
    /// only field the readout reads when it builds the backend request: the decode above is
    /// evidence about the bytes, never a source for them.
    pub reference: String,
}

impl Media {
    /// The `image_url` part for the backend request, forwarding `reference` unchanged.
    ///
    /// Built by interpolation rather than re-serialization so the reference string is the
    /// same characters the caller sent; a JSON round-trip would still be correct, but this
    /// makes the pass-through property visible at the call site.
    pub fn content_part(&self) -> Value {
        json!({"type": "image_url", "image_url": {"url": self.reference}})
    }

    /// The provenance object recorded in a response.
    ///
    /// `kind` is emitted as a constant because this crate produces one kind of media, and the
    /// URL is restated from the reference exactly when there is no local fingerprint -- a
    /// remote URL is the only case that hashes nothing, so the two cannot disagree.
    pub fn provenance(&self) -> Value {
        json!({
            "kind": "image",
            "mime": self.mime,
            "bytes": self.size,
            "sha256": self.sha256,
            "url": self.sha256.is_none().then_some(self.reference.as_str()),
        })
    }
}

/// One message's content, folded from a string or a content-parts array.
///
/// Text and media are kept separate and in order: a JSON `evidence` string cannot represent
/// "an image between these two paragraphs", so the fold must not try.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Content {
    pub text: String,
    pub media: Vec<Media>,
}

impl Content {
    fn text_only(text: String) -> Self {
        Content {
            text,
            media: Vec::new(),
        }
    }
}

/// The `evidence_renderer` value for a canonical (`system`+`user`) request that carries media.
///
/// A distinct value from the text renderers on purpose: the media fold changes the prompt, so
/// a recorded decision must say which fold produced it.
pub const EVIDENCE_CANONICAL_MEDIA: &str = "system+user+media";

/// The `evidence_renderer` value for a role-labelled transcript that carries media.
pub const EVIDENCE_TRANSCRIPT_MEDIA: &str = "transcript+media";

/// A real 1x1 PNG as a `data:` URI, used by the startup media probe.
///
/// It is a real image on purpose: a probe that sent something the processor rejects would
/// report "no media support" for a server that merely disliked the probe, and a build with
/// no vision path is expected to answer 400/422 -- that is the negative signal being read.
pub const PROBE_IMAGE_DATA_URI: &str = "data:image/png;base64,\
     iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==";

/// Fold one message's `content` into text + media.
///
/// A part that is neither `text` nor an accepted `image_url` is refused **by name** rather
/// than dropped, and the index of the offending message is always in the message.
pub fn read_content(
    message: &Value,
    index: usize,
    limits: &MediaLimits,
) -> Result<Content, (&'static str, String)> {
    match message.get("content") {
        Some(Value::String(text)) => Ok(Content::text_only(text.clone())),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            let mut media = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        // A `text` part whose `text` is missing or not a string is refused, not
                        // skipped: skipping it answers a question the caller did not ask, and the
                        // response would carry no sign that a part was lost.
                        let fragment =
                            part.get("text").and_then(Value::as_str).ok_or_else(|| {
                                (
                                    "unsupported_content_part",
                                    format!(
                                        "messages[{}] has a `text` content part whose `text` is \
                                     missing or not a string",
                                        index
                                    ),
                                )
                            })?;
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(fragment);
                    }
                    Some("image_url") => {
                        if !limits.allow_media {
                            return Err((
                                "media_not_enabled",
                                format!(
                                    "messages[{}] carries an image_url part, but this server \
                                     was started without --allow-media; media scoring is a \
                                     degraded readout and must be opted into.",
                                    index
                                ),
                            ));
                        }
                        if media.len() >= limits.max_images {
                            return Err((
                                "too_many_images",
                                format!(
                                    "messages[{}] carries more than {} images",
                                    index, limits.max_images
                                ),
                            ));
                        }
                        media.push(read_image(part, index, limits)?);
                    }
                    Some(other) => {
                        return Err((
                            "unsupported_content_part",
                            format!(
                                "messages[{}] carries a {:?} content part; this endpoint scores \
                                 text and image decisions",
                                index, other
                            ),
                        ))
                    }
                    None => {
                        return Err((
                            "unsupported_content_part",
                            format!("messages[{}] has a content part with no \"type\"", index),
                        ))
                    }
                }
            }
            Ok(Content { text, media })
        }
        _ => Err((
            "messages_not_semif_contract",
            format!(
                "messages[{}] needs `content`: a string, or an array of text/image_url parts",
                index
            ),
        )),
    }
}

/// Parse one `{"type":"image_url","image_url":{"url": …}}` part.
fn read_image(
    part: &Value,
    index: usize,
    limits: &MediaLimits,
) -> Result<Media, (&'static str, String)> {
    let reference = part
        .get("image_url")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            (
                "invalid_media",
                format!("messages[{}] image_url part needs `image_url.url`", index),
            )
        })?;
    // Only `url` is forwarded, and the other keys OpenAI defines there (`detail` above all)
    // select *how* the backend renders the image -- i.e. how many image tokens the model sees.
    // Dropping one would answer at the default resolution while the response looked exactly
    // like a request that asked for it.
    if let Some(unknown) = reference.keys().find(|key| key.as_str() != "url") {
        return Err((
            "unsupported_content_part",
            format!(
                "messages[{}] image_url uses `{}`; this endpoint forwards only `image_url.url`, \
                 so any other key would be silently ignored",
                index, unknown
            ),
        ));
    }
    let url = reference
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            (
                "invalid_media",
                format!("messages[{}] image_url part needs `image_url.url`", index),
            )
        })?;

    if let Some(rest) = url.strip_prefix("data:") {
        return decode_data_uri(rest, url, index, limits);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if !limits.allow_remote {
            return Err((
                "media_remote_not_allowed",
                format!(
                    "messages[{}] references a remote image URL; this server will not make the \
                     backend fetch an arbitrary URL, and it cannot fingerprint the bytes that \
                     come back. Enable --allow-remote-media to accept remote URLs, or send the \
                     image as a `data:` URI so its hash can be recorded.",
                    index
                ),
            ));
        }
        return Ok(Media {
            mime: "unknown".to_string(),
            sha256: None,
            size: None,
            reference: url.to_string(),
        });
    }
    Err((
        "invalid_media",
        format!(
            "messages[{}] image URL must be a `data:` URI or an http(s) URL; a filesystem path \
             would be a file-read primitive on the backend host.",
            index
        ),
    ))
}

/// Decode `data:<mime>;base64,<payload>` and fingerprint the bytes.
fn decode_data_uri(
    rest: &str,
    url: &str,
    index: usize,
    limits: &MediaLimits,
) -> Result<Media, (&'static str, String)> {
    let (header, payload) = rest.split_once(',').ok_or_else(|| {
        (
            "invalid_media",
            format!(
                "messages[{}] data URI has no comma before the payload",
                index
            ),
        )
    })?;
    let mut fields = header.split(';');
    let declared = fields.next().unwrap_or("").trim().to_ascii_lowercase();
    let mime = if declared.is_empty() {
        "unknown".to_string()
    } else {
        declared
    };
    let is_base64 = fields.any(|field| field.eq_ignore_ascii_case("base64"));
    // A non-base64 data URI is refused rather than forwarded: without base64 this server
    // cannot bound the decoded size, so the ceiling would be unenforceable. That is a
    // deliberate narrowing of pass-through, not a transformation of the reference.
    if !is_base64 {
        return Err((
            "invalid_media",
            format!(
                "messages[{}] data URI must be base64-encoded (a percent-encoded payload has \
                 no enforceable size ceiling)",
                index
            ),
        ));
    }
    // Reject on the encoded length before allocating, so a huge payload cannot make this
    // process allocate 3/4 of it just to then fail the decoded-size check. Saturating
    // arithmetic: an extreme `--max-media-bytes` must not wrap to a tiny ceiling (which would
    // reject every image) or overflow in a debug build.
    let ceiling = limits
        .max_media_bytes
        .saturating_div(3)
        .saturating_mul(4)
        .saturating_add(4);
    if payload.len() > ceiling {
        return Err((
            "media_too_large",
            format!(
                "messages[{}] image payload is {} base64 bytes; the limit is {} decoded bytes",
                index,
                payload.len(),
                limits.max_media_bytes
            ),
        ));
    }
    let bytes = BASE64.decode(payload.trim()).map_err(|error| {
        (
            "invalid_media",
            format!("messages[{}] image base64 is invalid: {}", index, error),
        )
    })?;
    if bytes.len() > limits.max_media_bytes {
        return Err((
            "media_too_large",
            format!(
                "messages[{}] image decodes to {} bytes; the limit is {}",
                index,
                bytes.len(),
                limits.max_media_bytes
            ),
        ));
    }
    if bytes.is_empty() {
        return Err((
            "invalid_media",
            format!("messages[{}] image decodes to zero bytes", index),
        ));
    }
    // `reference` is the caller's string verbatim; the decoded `bytes` are dropped after
    // being fingerprinted. Pass-through is what reaches the backend.
    Ok(Media {
        mime,
        sha256: Some(sha256_hex(&bytes)),
        size: Some(bytes.len()),
        reference: url.to_string(),
    })
}

/// Lowercase hex sha256, matching the rest of this crate's fingerprint format.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn limits() -> MediaLimits {
        MediaLimits {
            allow_media: true,
            allow_remote: false,
            max_images: 2,
            max_media_bytes: 1024,
        }
    }

    /// The probe image, reused so the test's idea of a valid PNG cannot drift from the one
    /// the startup probe sends.
    fn png_data_uri() -> String {
        PROBE_IMAGE_DATA_URI.to_string()
    }

    #[test]
    fn a_string_content_is_text_only() {
        let message = json!({"role": "user", "content": "hello"});
        let content = read_content(&message, 0, &limits()).unwrap();
        assert_eq!(content.text, "hello");
        assert!(content.media.is_empty());
    }

    #[test]
    fn a_data_uri_is_decoded_and_fingerprinted() {
        let message = json!({"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": png_data_uri()}},
        ]});
        let content = read_content(&message, 0, &limits()).unwrap();
        assert_eq!(content.text, "what is this?");
        assert_eq!(content.media.len(), 1);
        let image = &content.media[0];
        assert_eq!(image.mime, "image/png");
        assert_eq!(image.size, Some(67));
        assert_eq!(
            image.sha256.as_deref().unwrap(),
            "ebf4f635a17d10d6eb46ba680b70142419aa3220f228001a036d311a22ee9d2a"
        );
        assert_eq!(image.provenance()["url"], json!(null));
    }

    #[test]
    fn a_remote_url_is_refused_without_the_opt_in() {
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}},
        ]});
        let error = read_content(&message, 3, &limits()).unwrap_err();
        assert_eq!(error.0, "media_remote_not_allowed");
        assert!(error.1.contains("messages[3]"), "{}", error.1);
    }

    #[test]
    fn a_remote_url_is_accepted_with_the_opt_in_but_carries_no_byte_hash() {
        let mut allowed = limits();
        allowed.allow_remote = true;
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}},
        ]});
        let content = read_content(&message, 0, &allowed).unwrap();
        assert_eq!(content.media[0].sha256, None);
        assert_eq!(content.media[0].size, None);
        assert_eq!(
            content.media[0].provenance()["url"],
            json!("https://example.invalid/x.png")
        );
    }

    #[test]
    fn media_without_the_master_switch_is_refused_by_name() {
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": png_data_uri()}},
        ]});
        let error = read_content(&message, 1, &MediaLimits::default()).unwrap_err();
        assert_eq!(error.0, "media_not_enabled");
    }

    #[test]
    fn a_reference_that_is_neither_a_data_uri_nor_http_is_refused() {
        for (reference, expected) in [
            ("file:///etc/passwd", "filesystem path"),
            ("data:image/png,notbase64", "base64"),
        ] {
            let message = json!({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": reference}},
            ]});
            let error = read_content(&message, 0, &limits()).unwrap_err();
            assert_eq!(error.0, "invalid_media", "{}", error.1);
            assert!(error.1.contains(expected), "{}", error.1);
        }
    }

    #[test]
    fn a_mime_the_backend_may_reject_is_still_passed_through() {
        // MIME judgement belongs to the backend processor; this server records what the
        // caller declared and forwards the reference unchanged.
        let reference = "data:application/pdf;base64,AAAA";
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": reference}},
        ]});
        let content = read_content(&message, 0, &limits()).unwrap();
        assert_eq!(content.media[0].mime, "application/pdf");
        assert_eq!(content.media[0].reference, reference);
    }

    #[test]
    fn the_forwarded_part_is_the_reference_verbatim() {
        let reference = png_data_uri();
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": reference}},
        ]});
        let media = read_content(&message, 0, &limits())
            .unwrap()
            .media
            .remove(0);
        assert_eq!(media.content_part()["image_url"]["url"], json!(reference));
        assert_eq!(media.reference, reference);
    }

    #[test]
    fn an_oversized_image_is_refused_before_allocation_spreads() {
        let mut small = limits();
        small.max_media_bytes = 8;
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": png_data_uri()}},
        ]});
        assert_eq!(
            read_content(&message, 0, &small).unwrap_err().0,
            "media_too_large"
        );
    }

    #[test]
    fn more_images_than_allowed_is_refused() {
        let mut one = limits();
        one.max_images = 1;
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": png_data_uri()}},
            {"type": "image_url", "image_url": {"url": png_data_uri()}},
        ]});
        assert_eq!(
            read_content(&message, 0, &one).unwrap_err().0,
            "too_many_images"
        );
    }

    #[test]
    fn a_part_this_server_cannot_read_is_refused_by_index_and_named() {
        // Refused rather than dropped, at every shape: skipping a part would answer a
        // question the caller did not ask, with nothing in the response saying so.
        let cases = [
            (json!([{"text": "orphan"}]), 2, "no \"type\""),
            (
                json!([{"type": "text", "text": 42}]),
                0,
                "missing or not a string",
            ),
            (
                json!([{"type": "image_url", "image_url": {"url": png_data_uri(), "detail": "high"}}]),
                0,
                "detail",
            ),
        ];
        for (content, index, expected) in cases {
            let message = json!({"role": "user", "content": content});
            let error = read_content(&message, index, &limits()).unwrap_err();
            assert_eq!(error.0, "unsupported_content_part", "{}", error.1);
            assert!(
                error.1.contains(&format!("messages[{}]", index)),
                "{}",
                error.1
            );
            assert!(error.1.contains(expected), "{}", error.1);
        }
    }

    #[test]
    fn text_parts_are_joined_with_a_newline_exactly_as_before() {
        let message = json!({"role": "user", "content": [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]});
        assert_eq!(
            read_content(&message, 0, &limits()).unwrap().text,
            "line one\nline two"
        );
    }
}
