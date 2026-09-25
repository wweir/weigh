// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Row validation, message construction, and gemma-4 chat-template rendering.
//!
//! ## Why there is no Jinja engine here
//!
//! The gemma-4 `chat_template.jinja` is 17,336 bytes and references ~140 names, and
//! the obvious Rust port (`minijinja`) **fails** on it: the template calls `.get(` 14
//! times and minijinja does not implement that method
//! (`unknown method: map has no method named get (in chat:238)`).
//!
//! But the served contract only ever sends one message shape -- a plain system message plus a
//! plain user message, no tools, no image/audio/video -- so the reachable subgraph of the
//! template is three lines. Rendering it directly was verified byte-identical to
//! `transformers`' Jinja output on **777/777** fixture rows. `enable_thinking` was also
//! verified to make no difference for this shape.

use serde_json::Value;
use weigh::pyjson::Json;
use weigh::template::ChatTemplate;

// The answer-slot model belongs to the library, which is where the token-id contract is
// enforced. Re-exported here because the rest of SemIf's contract (the option-count bound and
// the letters in the payload) has to agree with it, and two copies would not.
pub use weigh::LETTERS;

pub const PROMPT_VERSION: &str = "direct-options-v1";

/// How a request's `messages` array was folded into the scored (criterion, evidence) pair.
///
/// `system+user` is the canonical shape and reproduces the historical prompt byte for byte:
/// the evidence is the single user message verbatim. `transcript` is the general fold used
/// for every other array shape (multi-turn history, tool results, several system messages):
/// the evidence is the non-criterion messages rendered as `role: text`, blank-line joined.
///
/// The two produce different prompt strings for the same request, so every response records
/// which one ran. Without this key a served decision could not be reproduced after a release
/// that widened the accepted message shapes.
pub const EVIDENCE_CANONICAL: &str = "system+user";
pub const EVIDENCE_TRANSCRIPT: &str = "transcript";

/// The `direct-options-v1` system prompt, verbatim.
pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. \
Choose exactly one listed option. Respond with only its uppercase letter, with no \
explanation or reasoning.";

pub struct Option {
    pub id: String,
    pub description: String,
}

pub struct Row {
    pub id: String,
    pub state: Value,
    pub question: String,
    pub options: Vec<Option>,
}

/// Validate one `direct-options-v1` row.
///
/// Deliberately strict: the accepted row shapes are part of the published contract, so a
/// caller cannot get a scored answer out of a row that the contract does not describe.
pub fn validate_row(value: &Value) -> Result<Row, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Row must be a JSON object".to_string())?;

    for key in ["id", "state", "question", "options"] {
        if !object.contains_key(key) {
            return Err(format!("Row is missing fields: [\"{}\"]", key));
        }
    }

    let id = object["id"]
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "id and question must be nonempty strings".to_string())?
        .to_string();
    let question = object["question"]
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "id and question must be nonempty strings".to_string())?
        .to_string();

    let state = &object["state"];
    let state_ok = match state {
        Value::String(text) => !text.is_empty(),
        Value::Object(map) => !map.is_empty(),
        Value::Array(items) => !items.is_empty(),
        _ => false,
    };
    if !state_ok {
        return Err("state must be a nonempty string, object, or array".to_string());
    }

    let option_values = object["options"]
        .as_array()
        .ok_or_else(|| "options must contain 2-16 entries".to_string())?;
    if option_values.len() < 2 || option_values.len() > LETTERS.len() {
        return Err("options must contain 2-16 entries".to_string());
    }

    let mut options = Vec::with_capacity(option_values.len());
    let mut seen = std::collections::HashSet::new();
    for entry in option_values {
        let entry = entry
            .as_object()
            .ok_or_else(|| "Each option needs string id and description fields".to_string())?;
        let option_id = entry
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "Each option needs string id and description fields".to_string())?;
        let description = entry
            .get("description")
            .and_then(Value::as_str)
            .ok_or_else(|| "Each option needs string id and description fields".to_string())?;
        if !seen.insert(option_id.to_string()) {
            return Err("Option IDs must be unique".to_string());
        }
        options.push(Option {
            id: option_id.to_string(),
            description: description.to_string(),
        });
    }

    Ok(Row {
        id,
        state: state.clone(),
        question,
        options,
    })
}

/// Convert a `serde_json::Value` into the ordered form, preserving object key order
/// (`serde_json` is built with `preserve_order` so the evidence payload round-trips in
/// the same order `json.dumps` would emit).
pub fn to_json(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(flag) => Json::Bool(*flag),
        // `Number`'s own text is already exact, so integers bypass any range decision.
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => Json::Float(float),
            _ => Json::Raw(number.to_string()),
        },
        Value::String(text) => Json::Str(text.clone()),
        Value::Array(items) => Json::Array(items.iter().map(to_json).collect()),
        Value::Object(map) => Json::Object(
            map.iter()
                .map(|(key, item)| (key.clone(), to_json(item)))
                .collect(),
        ),
    }
}

/// Build the user message: `json.dumps(payload, ensure_ascii=False)` over the same
/// three fields `core.direct_messages` emits, in the same order.
pub fn user_message(row: &Row) -> Result<String, String> {
    let payload = Json::Object(vec![
        ("evidence".to_string(), to_json(&row.state)),
        ("criterion".to_string(), Json::Str(row.question.clone())),
        (
            "options".to_string(),
            Json::Array(
                row.options
                    .iter()
                    .zip(LETTERS.chars())
                    .map(|(option, letter)| {
                        Json::Object(vec![
                            ("letter".to_string(), Json::Str(letter.to_string())),
                            (
                                "description".to_string(),
                                Json::Str(option.description.clone()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    // ensure_ascii=False: the prompt carries the evidence as written, not escaped.
    payload.dumps()
}

/// The text part of a media request: the same `{evidence, criterion, options}` JSON the text
/// path sends, carrying the *text* evidence. The images become sibling content parts after it.
pub fn media_user_payload(
    evidence_text: &str,
    criterion: &str,
    option_ids: &[String],
) -> Result<String, String> {
    let payload = Json::Object(vec![
        ("evidence".to_string(), Json::Str(evidence_text.to_string())),
        ("criterion".to_string(), Json::Str(criterion.to_string())),
        (
            "options".to_string(),
            Json::Array(
                option_ids
                    .iter()
                    .zip(LETTERS.chars())
                    .map(|(label, letter)| {
                        Json::Object(vec![
                            ("letter".to_string(), Json::Str(letter.to_string())),
                            ("description".to_string(), Json::Str(label.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    payload.dumps()
}

/// Render the chat template for one system + user turn.
///
/// The gemma-4 branch is byte-identical to `tokenizer.apply_chat_template(messages,
/// tokenize=False, add_generation_prompt=True, enable_thinking=False)` on all 777 fixture
/// rows; every family's reachable subgraph lives in [`weigh::template`]. `prompt_version`
/// versions this content and `prompt_template` versions the wrapper, so only the pair
/// identifies the string the server actually receives.
pub fn render_prompt_with(template: ChatTemplate, row: &Row) -> Result<String, String> {
    let user = user_message(row)?;
    Ok(template.render(DIRECT_SYSTEM, &user))
}
