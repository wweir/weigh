// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Row validation, message construction, and prompt assembly.
//!
//! The rendered wrapper is not this module's business: [`weigh::template`] owns the family
//! registry -- and records why it is a registry rather than a Jinja engine -- while this module
//! owns the *content*: the row that validates, the `{evidence, criterion, options}` JSON, and
//! the option-to-letter binding. `prompt_version` here plus `prompt_template` there are what
//! identify the string the server actually received.

use serde_json::Value;
use weigh::pyjson::Json;
use weigh::schema::{MAX_VALUES, MIN_VALUES};
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
        .ok_or_else(|| format!("options must contain {}-{} entries", MIN_VALUES, MAX_VALUES))?;
    if option_values.len() < MIN_VALUES || option_values.len() > MAX_VALUES {
        return Err(format!(
            "options must contain {}-{} entries",
            MIN_VALUES, MAX_VALUES
        ));
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
        // A number that fit the JSON parser's i64/u64 range keeps its exact text; a magnitude
        // beyond it was already an f64 before this module saw it, so either branch emits what
        // `json.dumps(json.loads(...))` would.
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

/// The `direct-options-v1` user payload: `{evidence, criterion, options}`, in that order, with
/// every option bound to its slot letter.
///
/// One builder for both paths, because the media payload is the same JSON with a different
/// *evidence* source -- not a second contract.
fn user_payload(
    evidence: Json,
    criterion: &str,
    descriptions: &[String],
) -> Result<String, String> {
    let payload = Json::Object(vec![
        ("evidence".to_string(), evidence),
        ("criterion".to_string(), Json::Str(criterion.to_string())),
        (
            "options".to_string(),
            Json::Array(
                descriptions
                    .iter()
                    .zip(LETTERS.chars())
                    .map(|(description, letter)| {
                        Json::Object(vec![
                            ("letter".to_string(), Json::Str(letter.to_string())),
                            ("description".to_string(), Json::Str(description.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    // ensure_ascii=False: the prompt carries the evidence as written, not escaped.
    payload.dumps()
}

/// Build the user message for one validated row.
pub fn user_message(row: &Row) -> Result<String, String> {
    let descriptions: Vec<String> = row
        .options
        .iter()
        .map(|option| option.description.clone())
        .collect();
    user_payload(to_json(&row.state), &row.question, &descriptions)
}

/// The text part of a media request: the same payload the text path sends, carrying the *text*
/// evidence. The images become sibling content parts after it.
pub fn media_user_payload(
    evidence_text: &str,
    criterion: &str,
    option_ids: &[String],
) -> Result<String, String> {
    user_payload(Json::Str(evidence_text.to_string()), criterion, option_ids)
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
