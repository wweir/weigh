// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Python-compatible JSON output.
//!
//! These payloads are compared byte for byte with the Python client's: `input_ids_sha256` is
//! literally the SHA-256 of `json.dumps(ids)`, spaces included, and a `prompt_sha256` only
//! means something if the prompt bytes match. `serde_json` differs from `json.dumps` in three
//! ways that matter here -- its separators, its array spacing, and its float rendering (it
//! disagreed with CPython's `repr` on 4 of 12 sampled values before this module existed), so
//! the payloads are rendered here instead.
//!
//! Built with `allow_nan=False` semantics, and with `ensure_ascii=False` like the Python
//! client's prompt payload: non-ASCII evidence stays in the payload as written, because the
//! payload is a UTF-8 HTTP body.

use std::fmt::Write as _;

/// Split a shortest-decimal string (`"-1.25"`, `"1.25e-7"`, `"123.0"`) into its
/// significant digits and the decimal-point position, such that the value equals
/// `0.<digits> * 10^decpt`.
///
/// Accepts both notations because the shortest-formatter's notation is not part of its
/// contract; only the digits are.
fn digits_and_decpt(text: &str) -> Option<(String, i32)> {
    let text = text.trim_start_matches('-');
    let (mantissa, exponent) = match text.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()?),
        None => (text, 0),
    };
    let (integer_part, fraction_part) = match mantissa.split_once('.') {
        Some((integer_part, fraction_part)) => (integer_part, fraction_part),
        None => (mantissa, ""),
    };
    let raw = format!("{}{}", integer_part, fraction_part);
    // The value is `raw` read as an integer times 10^(exponent - fraction_part.len()).
    let point = integer_part.len() as i32 + exponent;

    let trimmed = raw.trim_start_matches('0');
    let leading_zeros = raw.len() as i32 - trimmed.len() as i32;
    let without_trailing = trimmed.trim_end_matches('0');
    if without_trailing.is_empty() {
        // The value was zero (already handled by the caller, but stay total).
        return Some(("0".to_string(), 1));
    }
    Some((without_trailing.to_string(), point - leading_zeros))
}

/// Format an `f64` exactly as CPython's `float_repr` does.
///
/// CPython takes the shortest digit string that round-trips, computes `decpt` (the
/// position of the decimal point, so the value is `0.<digits> * 10^decpt`), and then
/// uses scientific notation iff `decpt <= -4 || decpt > 16`, with an exponent of at
/// least two digits. `-0.0` keeps its sign (`repr(-0.0) == '-0.0'`).
///
/// The digits come from `ryu` rather than from Rust's `{:e}`. Both are
/// shortest-round-trip, but they resolve an exact tie differently: on
/// `810553441865041.25` (a double that is exactly midway between two 16-digit
/// decimals, both of which round-trip) Rust's Grisu emits `...041.3` where CPython
/// emits `...041.2`. Measured on 20,000 random bit patterns, `{:e}` disagreed with
/// CPython on 5 and `ryu` on 0.
///
/// Returns `None` for NaN and infinities: the caller passes `allow_nan=False`, which
/// makes Python raise, so emitting them here would be silently wrong.
fn py_float(value: f64) -> Option<String> {
    if !value.is_finite() {
        return None;
    }
    if value == 0.0 {
        return Some(
            if value.is_sign_negative() {
                "-0.0"
            } else {
                "0.0"
            }
            .to_string(),
        );
    }

    let sign = if value < 0.0 { "-" } else { "" };
    let mut buffer = ryu::Buffer::new();
    let (digits, decpt) = digits_and_decpt(buffer.format_finite(value.abs()))?;

    let mut out = String::with_capacity(digits.len() + 8);
    out.push_str(sign);
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let exponent_out = decpt - 1;
        out.push(if exponent_out < 0 { '-' } else { '+' });
        let magnitude = exponent_out.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        let _ = write!(out, "{}", magnitude);
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        // Integral value: CPython's Py_DTSF_ADD_DOT_0 appends ".0" so that a float
        // never reads back as an int.
        out.push_str(&digits);
        for _ in 0..(decpt as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    Some(out)
}

/// Append `text` as a Python `json.dumps` string literal.
///
/// Non-ASCII is written as-is (`ensure_ascii=False`), which is what the Python client's
/// prompt payload does; control characters are escaped either way, because that is what
/// `json.dumps`'s ESCAPE regex does regardless of the flag.
fn write_py_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            // Python's ESCAPE_ASCII pattern is `([\\"]|[^\ -~])`, applied whatever
            // `ensure_ascii` is: everything below 0x20 is escaped, and everything from 0x20
            // up (including DEL and astral characters) is emitted as written here.
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// An ordered JSON value, used to build the prompt's evidence payload.
///
/// Objects are a `Vec` of pairs because `json.dumps` preserves insertion order, and the
/// payload's byte layout is part of the prompt: it must match the Python renderer exactly
/// for the two clients to send the server the same string.
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
    /// Pre-rendered JSON text emitted verbatim.
    ///
    /// Used for integers lifted out of the evidence payload: `serde_json`'s `Number`
    /// already holds the exact source text, so passing it through avoids any
    /// i64/u64 range decision of my own, and matches `json.dumps` byte for byte.
    Raw(String),
}

impl Json {
    /// Render the value the way the Python client's `json.dumps(..., ensure_ascii=False)`
    /// does, separators included.
    pub fn dumps(&self) -> Result<String, String> {
        let mut out = String::new();
        self.write(&mut out)?;
        Ok(out)
    }

    fn write(&self, out: &mut String) -> Result<(), String> {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Float(value) => match py_float(*value) {
                Some(text) => out.push_str(&text),
                // Mirrors `allow_nan=False`: Python raises ValueError rather than
                // writing a non-standard NaN literal into the artifact.
                None => {
                    return Err(format!(
                        "out-of-range float cannot be serialized: {}",
                        value
                    ))
                }
            },
            Json::Str(text) => write_py_string(out, text),
            Json::Raw(text) => out.push_str(text),
            Json::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    item.write(out)?;
                }
                out.push(']');
            }
            Json::Object(pairs) => {
                out.push('{');
                for (index, (key, value)) in pairs.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    write_py_string(out, key);
                    out.push_str(": ");
                    value.write(out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
}

/// `json.dumps(ids)` for a list of token ids -- the exact preimage of
/// `input_ids_sha256`, spaces after commas included.
pub fn dumps_int_list(values: &[u32]) -> String {
    let mut out = String::with_capacity(values.len() * 6 + 2);
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{}", value);
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte parity with `json.dumps` is the whole point of this module, and it had no tests.
    /// These expected strings were produced by CPython 3.13:
    /// `python3 -c 'import json; print(json.dumps(v, ensure_ascii=False))'`.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn floats_render_like_cpython_repr() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.25, "-1.25"),
            // `repr` switches to scientific notation iff decpt <= -4 or decpt > 16.
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (5e-324, "5e-324"),
            (1e300, "1e+300"),
            // An exact tie between two 16-digit decimals, where Rust's own `{:e}` (Grisu)
            // disagrees with CPython and `ryu` does not. Written with the trailing `.25` on
            // purpose: that spells the double sitting exactly at the tie, where `.2` and `.3`
            // both round-trip to it, so the case does not depend on which of the two shorter
            // decimals a parser happens to pick.
            (810553441865041.25, "810553441865041.2"),
        ];
        for (value, expected) in cases {
            assert_eq!(py_float(*value).as_deref(), Some(*expected), "{}", value);
        }
        // `allow_nan=False`: Python raises, so this must not invent a literal.
        assert_eq!(py_float(f64::NAN), None);
        assert_eq!(py_float(f64::INFINITY), None);
        assert_eq!(py_float(f64::NEG_INFINITY), None);
    }

    #[test]
    fn a_payload_uses_the_python_separators_and_keeps_non_ascii_raw() {
        let value = Json::Object(vec![
            (
                "evidence".to_string(),
                Json::Str("café\u{1f600}".to_string()),
            ),
            (
                "ids".to_string(),
                Json::Array(vec![Json::Float(1.5), Json::Null]),
            ),
            ("ok".to_string(), Json::Bool(true)),
        ]);
        // `separators=(', ', ': ')` and `ensure_ascii=False`: the non-ASCII stays as UTF-8
        // (no \u escapes, no surrogate pairs).
        assert_eq!(
            value.dumps().unwrap(),
            "{\"evidence\": \"caf\u{e9}\u{1f600}\", \"ids\": [1.5, null], \"ok\": true}"
        );
    }

    #[test]
    fn control_characters_and_quotes_are_escaped_in_every_mode() {
        // `json.dumps`'s ESCAPE regex applies whatever `ensure_ascii` is.
        let value = Json::Str("a\"b\\c\nd\te\u{1}f".to_string());
        assert_eq!(value.dumps().unwrap(), "\"a\\\"b\\\\c\\nd\\te\\u0001f\"");
        // DEL (0x7f) is not a control character for JSON escaping: it stays raw here, which
        // is what `ensure_ascii=False` does.
        assert_eq!(
            Json::Str("\u{7f}".to_string()).dumps().unwrap(),
            "\"\u{7f}\""
        );
    }

    #[test]
    fn a_non_finite_float_is_an_error_rather_than_a_bare_literal() {
        assert!(Json::Float(f64::NAN).dumps().is_err());
        assert!(Json::Array(vec![Json::Float(f64::INFINITY)])
            .dumps()
            .is_err());
    }

    #[test]
    fn dumps_int_list_is_the_exact_hash_preimage() {
        // `json.dumps([1, 23]) == "[1, 23]"`, spaces included: this string is hashed into
        // `input_ids_sha256`.
        assert_eq!(dumps_int_list(&[1, 23]), "[1, 23]");
        assert_eq!(dumps_int_list(&[]), "[]");
        assert_eq!(dumps_int_list(&[0]), "[0]");
    }
}
