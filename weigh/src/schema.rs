// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The decision schema every serving request must carry.
//!
//! The reader computes exactly one softmax over one finite answer-slot set, so the schema is
//! required to describe exactly one field with exactly one JSON type. Anything else is
//! refused at the door rather than answered halfway: a multi-field schema would need
//! several readouts and a cross-field consistency rule this server does not implement,
//! and a multi-type field would span heterogeneous tokenizations.
//!
//! Accepted shapes, exactly one per request:
//!
//! ```json
//! "response_format": {"type": "json_schema", "json_schema": {"schema": { ... }}}
//! ```
//!
//! or a top-level `"schema": { ... }`, or `"guided_json": { ... }` (vLLM/SGLang's own
//! key, which [`DecisionSchema::from_request`] reads). More than one of them is an error
//! rather than a silent preference.

use serde_json::{Map, Value};

/// Minimum option count: one option is not a decision.
pub const MIN_VALUES: usize = 2;
/// Maximum option count: the answer slot is one uppercase letter, and `LETTERS` is
/// `ABCDEFGHIJKLMNOP`.
pub const MAX_VALUES: usize = crate::slots::LETTERS.len();

/// Keywords a field schema may use. Anything else is refused instead of ignored: a
/// `pattern` or a `minimum` that is not enforced would let the response violate the
/// schema the caller sent.
const FIELD_KEYWORDS: &[&str] = &["type", "enum", "description", "title"];

const UNION_KEYWORDS: &[&str] = &["anyOf", "oneOf", "allOf", "not", "$ref"];

#[derive(Debug)]
pub struct SchemaError {
    pub code: &'static str,
    pub message: String,
}

impl SchemaError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        SchemaError {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug)]
pub struct DecisionSchema {
    /// The single property name, echoed as the JSON key of the response content.
    pub field: String,
    /// Allowed values in schema order: `values[0]` is answer slot `A`.
    pub values: Vec<Value>,
}

impl DecisionSchema {
    /// Pull the schema out of a request body, rejecting an ambiguous or missing one.
    pub fn from_request(body: &Map<String, Value>) -> Result<Self, SchemaError> {
        Self::from_json_schema(locate(body)?)
    }

    pub fn from_json_schema(schema: &Value) -> Result<Self, SchemaError> {
        let object = schema.as_object().ok_or_else(|| {
            SchemaError::new(
                "schema_not_object",
                "the decision schema must be a JSON object",
            )
        })?;
        let schema_type = object.get("type").and_then(Value::as_str);
        if schema_type != Some("object") {
            return Err(SchemaError::new(
                "schema_not_object",
                format!(
                    "the decision schema must have type \"object\", found {:?}",
                    schema_type
                ),
            ));
        }
        reject_unions(object, "the decision schema")?;

        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                SchemaError::new(
                    "schema_not_single_field",
                    "the decision schema needs a `properties` object",
                )
            })?;
        if properties.len() != 1 {
            return Err(SchemaError::new(
                "schema_not_single_field",
                format!(
                    "the decision schema must declare exactly one field; it declares {}. The \
                     service scores one distribution over one answer-slot set, so a multi-field \
                     decision is not supported.",
                    properties.len()
                ),
            ));
        }
        let (field, field_schema) = properties.iter().next().expect("len == 1");

        let required = object
            .get("required")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                SchemaError::new(
                    "schema_field_not_required",
                    "the decision schema needs `required` naming its single field",
                )
            })?;
        // Every entry must be a string: dropping a malformed one would accept
        // `["verdict", 5]` as if it named exactly one field.
        let names: Vec<&str> = required.iter().filter_map(Value::as_str).collect();
        if names.len() != required.len() || names.len() != 1 || names[0] != field.as_str() {
            return Err(SchemaError::new(
                "schema_field_not_required",
                format!("`required` must be exactly [{:?}]", field),
            ));
        }
        if object.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(SchemaError::new(
                "schema_allows_extra_fields",
                "the decision schema must set `additionalProperties: false`; extra fields cannot \
                 be scored",
            ));
        }

        let values = field_values(field, field_schema)?;
        Ok(DecisionSchema {
            field: field.clone(),
            values,
        })
    }

    /// The model-facing option text and the label echoed back to the caller. For a scalar
    /// enum they are the same string; the value itself is what the caller gets back, so
    /// no id/description mapping can drift.
    pub fn label(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        }
    }

    /// Strict JSON content for the winning value: `{"<field>": <value>}`.
    pub fn content(&self, choice: &Value) -> String {
        let mut object = Map::new();
        object.insert(self.field.clone(), choice.clone());
        // A single scalar key cannot fail to serialize.
        Value::Object(object).to_string()
    }

    pub fn labels(&self) -> Vec<String> {
        self.values.iter().map(Self::label).collect()
    }
}

/// Find the one decision schema in a request body.
///
/// Three sources are accepted, because three ecosystems carry it differently: the OpenAI
/// SDK's `response_format.json_schema.schema`, a top-level `schema`, and the vLLM/SGLang
/// native `guided_json`. More than one is an error rather than a silent precedence rule.
fn locate(body: &Map<String, Value>) -> Result<&Value, SchemaError> {
    let mut sources: Vec<(&str, &Value)> = Vec::new();
    if let Some(schema) = body.get("schema") {
        sources.push(("`schema`", schema));
    }
    if let Some(format) = body.get("response_format") {
        let kind = format.get("type").and_then(Value::as_str);
        if kind != Some("json_schema") {
            return Err(SchemaError::new(
                "unsupported_response_format",
                format!(
                    "response_format.type is {:?}; served decisions implement `json_schema` only",
                    kind
                ),
            ));
        }
        let schema = format
            .get("json_schema")
            .and_then(|entry| entry.get("schema"))
            .ok_or_else(|| {
                SchemaError::new(
                    "missing_schema",
                    "response_format.json_schema.schema is missing; a decision schema is required",
                )
            })?;
        sources.push(("`response_format.json_schema.schema`", schema));
    }
    if let Some(guided) = body.get("guided_json") {
        match guided {
            Value::Object(_) => sources.push(("`guided_json`", guided)),
            // vLLM also accepts a path or a registered name here. Neither is resolvable from
            // this process, so it is refused by name instead of being guessed at.
            _ => {
                return Err(SchemaError::new(
                    "unsupported_guided_json",
                    "`guided_json` must be the schema object itself; a path or a registered name \
                     cannot be resolved by this server",
                ))
            }
        }
    }
    match sources.len() {
        0 => Err(SchemaError::new(
            "missing_schema",
            "a decision schema is required: send response_format.json_schema.schema, a \
             top-level `schema`, or `guided_json`",
        )),
        1 => Ok(sources[0].1),
        _ => {
            let names: Vec<&str> = sources.iter().map(|(name, _)| *name).collect();
            Err(SchemaError::new(
                "ambiguous_schema",
                format!(
                    "send the decision schema once; found it in {}",
                    names.join(", ")
                ),
            ))
        }
    }
}

fn reject_unions(object: &Map<String, Value>, what: &str) -> Result<(), SchemaError> {
    for keyword in UNION_KEYWORDS {
        if object.contains_key(*keyword) {
            return Err(SchemaError::new(
                "schema_union_unsupported",
                format!(
                    "{} uses `{}`; a decision must have exactly one type",
                    what, keyword
                ),
            ));
        }
    }
    Ok(())
}

fn field_values(field: &str, field_schema: &Value) -> Result<Vec<Value>, SchemaError> {
    let object = field_schema.as_object().ok_or_else(|| {
        SchemaError::new(
            "schema_field_type_unsupported",
            format!("field {:?} must be a JSON object", field),
        )
    })?;
    reject_unions(object, &format!("field {:?}", field))?;
    for keyword in object.keys() {
        if !FIELD_KEYWORDS.contains(&keyword.as_str()) {
            return Err(SchemaError::new(
                "schema_keyword_unsupported",
                format!(
                    "field {:?} uses `{}`; this server honors only {}",
                    field,
                    keyword,
                    FIELD_KEYWORDS.join(", ")
                ),
            ));
        }
    }

    let field_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        SchemaError::new(
            "schema_field_type_unsupported",
            format!("field {:?} must declare exactly one `type`", field),
        )
    })?;
    let enumeration = object.get("enum").and_then(Value::as_array);

    match field_type {
        "boolean" => {
            if let Some(enumeration) = enumeration {
                let values: Vec<bool> = enumeration.iter().filter_map(Value::as_bool).collect();
                // Exactly `false` and `true`: `[false, false]` would declare one option while
                // this endpoint answers with two slots, which is the "one field, one type"
                // escape this module exists to close.
                if values.len() != enumeration.len() || values.len() != 2 || values[0] == values[1]
                {
                    return Err(SchemaError::new(
                        "schema_enum_type_mismatch",
                        format!(
                            "a boolean field's `enum` must be exactly [false, true]; found {:?}",
                            enumeration
                        ),
                    ));
                }
            }
            // `false` is slot A, `true` is slot B, so the slot order is fixed and does not
            // depend on the order an `enum` happened to be written in.
            Ok(vec![Value::Bool(false), Value::Bool(true)])
        }
        "string" | "integer" => {
            let enumeration = enumeration.ok_or_else(|| {
                SchemaError::new(
                    "schema_enum_required",
                    format!(
                        "a {} field must list its allowed values in `enum`",
                        field_type
                    ),
                )
            })?;
            if enumeration.len() < MIN_VALUES || enumeration.len() > MAX_VALUES {
                return Err(SchemaError::new(
                    "schema_value_count",
                    format!(
                        "`enum` must hold {}..={} values; found {}",
                        MIN_VALUES,
                        MAX_VALUES,
                        enumeration.len()
                    ),
                ));
            }
            let mut values: Vec<Value> = Vec::with_capacity(enumeration.len());
            for value in enumeration {
                let typed = match field_type {
                    "string" => value.is_string(),
                    // `as_i64` alone refuses every legal JSON integer above i64::MAX, which
                    // serde_json stores as u64; a float is still not an integer.
                    _ => value.is_i64() || value.is_u64(),
                };
                if !typed {
                    return Err(SchemaError::new(
                        "schema_enum_type_mismatch",
                        format!(
                            "field {:?} has type {:?} but `enum` contains {:?}; one field must \
                             have one type",
                            field, field_type, value
                        ),
                    ));
                }
                if values.contains(value) {
                    return Err(SchemaError::new(
                        "schema_value_duplicate",
                        format!("`enum` repeats the value {:?}", value),
                    ));
                }
                values.push(value.clone());
            }
            Ok(values)
        }
        other => Err(SchemaError::new(
            "schema_field_type_unsupported",
            format!(
                "field {:?} has type {:?}; supported types are string, integer, boolean",
                field, other
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn from(schema: Value) -> Result<DecisionSchema, SchemaError> {
        DecisionSchema::from_json_schema(&schema)
    }

    fn object_schema(field: Value) -> Value {
        json!({
            "type": "object",
            "properties": {"verdict": field},
            "required": ["verdict"],
            "additionalProperties": false
        })
    }

    #[test]
    fn a_string_enum_maps_to_slots_in_schema_order() {
        let schema = from(object_schema(
            json!({"type": "string", "enum": ["yes", "no"]}),
        ))
        .unwrap();
        assert_eq!(schema.field, "verdict");
        assert_eq!(schema.labels(), vec!["yes", "no"]);
        assert_eq!(schema.content(&schema.values[1]), r#"{"verdict":"no"}"#);
    }

    #[test]
    fn integer_and_boolean_values_keep_their_json_type() {
        let integers = from(object_schema(json!({"type": "integer", "enum": [1, 2, 3]}))).unwrap();
        assert_eq!(integers.content(&integers.values[2]), r#"{"verdict":3}"#);
        let booleans = from(object_schema(json!({"type": "boolean"}))).unwrap();
        assert_eq!(booleans.labels(), vec!["false", "true"]);
    }

    #[test]
    fn a_second_field_is_refused() {
        let error = from(json!({
            "type": "object",
            "properties": {"a": {"type": "boolean"}, "b": {"type": "boolean"}},
            "required": ["a"], "additionalProperties": false
        }))
        .unwrap_err();
        assert_eq!(error.code, "schema_not_single_field");
    }

    #[test]
    fn a_multi_type_field_is_refused() {
        let error = from(object_schema(json!({"type": ["string", "integer"]}))).unwrap_err();
        assert_eq!(error.code, "schema_field_type_unsupported");
        let error = from(object_schema(json!({"anyOf": [{"type": "string"}]}))).unwrap_err();
        assert_eq!(error.code, "schema_union_unsupported");
    }

    #[test]
    fn a_string_without_an_enum_is_refused() {
        let error = from(object_schema(json!({"type": "string"}))).unwrap_err();
        assert_eq!(error.code, "schema_enum_required");
    }

    #[test]
    fn mixed_enum_types_and_duplicates_are_refused() {
        let error = from(object_schema(json!({"type": "string", "enum": ["a", 1]}))).unwrap_err();
        assert_eq!(error.code, "schema_enum_type_mismatch");
        let error = from(object_schema(json!({"type": "string", "enum": ["a", "a"]}))).unwrap_err();
        assert_eq!(error.code, "schema_value_duplicate");
    }

    #[test]
    fn option_count_stays_within_the_letter_slots() {
        let many: Vec<String> = (0..17).map(|index| format!("v{}", index)).collect();
        let error = from(object_schema(json!({"type": "string", "enum": many}))).unwrap_err();
        assert_eq!(error.code, "schema_value_count");
    }

    #[test]
    fn extra_properties_and_missing_required_are_refused() {
        let error = from(json!({
            "type": "object", "properties": {"a": {"type": "boolean"}},
            "required": ["a"]
        }))
        .unwrap_err();
        assert_eq!(error.code, "schema_allows_extra_fields");
        let error = from(json!({
            "type": "object", "properties": {"a": {"type": "boolean"}},
            "required": [], "additionalProperties": false
        }))
        .unwrap_err();
        assert_eq!(error.code, "schema_field_not_required");
    }

    #[test]
    fn a_boolean_enum_must_be_exactly_false_and_true() {
        // `[false, false]` declares one value but this endpoint answers with two slots.
        assert!(from(object_schema(
            json!({"type": "boolean", "enum": [false, false]})
        ))
        .is_err());
        assert!(from(object_schema(
            json!({"type": "boolean", "enum": [true, true]})
        ))
        .is_err());
        // Listing both is fine in either order: the slot order is fixed by the type, not by
        // the order the enum happened to be written in.
        for enumeration in [json!([false, true]), json!([true, false])] {
            let schema = from(object_schema(
                json!({"type": "boolean", "enum": enumeration}),
            ))
            .expect("both booleans, once each");
            assert_eq!(schema.values, vec![Value::Bool(false), Value::Bool(true)]);
        }
    }

    #[test]
    fn integer_enums_accept_every_json_integer() {
        // serde_json stores integers above i64::MAX as u64.
        let big = from(object_schema(
            json!({"type": "integer", "enum": [18446744073709551615u64, 2]}),
        ));
        assert!(big.is_ok(), "{:?}", big.err());
        assert_eq!(
            from(object_schema(json!({"type": "integer", "enum": [-1, 2]})))
                .unwrap()
                .values,
            vec![json!(-1), json!(2)]
        );
        // A float is not an integer, even when its value is integral.
        assert!(from(object_schema(json!({"type": "integer", "enum": [1.0, 2]}))).is_err());
    }

    #[test]
    fn a_malformed_required_list_is_refused_rather_than_pruned() {
        let pruned = json!({
            "type": "object",
            "properties": {"verdict": {"type": "boolean"}},
            "required": ["verdict", 5],
            "additionalProperties": false,
        });
        assert!(from(pruned).is_err());
    }

    #[test]
    fn unenforced_keywords_are_refused_rather_than_ignored() {
        let error = from(object_schema(
            json!({"type": "string", "enum": ["a", "b"], "pattern": "^a$"}),
        ))
        .unwrap_err();
        assert_eq!(error.code, "schema_keyword_unsupported");
    }

    #[test]
    fn the_schema_is_located_from_either_placement() {
        let via_response_format = json!({
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "d", "schema": object_schema(json!({"type": "boolean"}))
            }}
        });
        let schema =
            DecisionSchema::from_request(via_response_format.as_object().unwrap()).unwrap();
        assert_eq!(schema.field, "verdict");

        let via_top_level = json!({"schema": object_schema(json!({"type": "boolean"}))});
        assert!(DecisionSchema::from_request(via_top_level.as_object().unwrap()).is_ok());

        let both = json!({
            "schema": object_schema(json!({"type": "boolean"})),
            "response_format": {"type": "json_schema", "json_schema": {"schema": {}}}
        });
        assert_eq!(
            DecisionSchema::from_request(both.as_object().unwrap())
                .unwrap_err()
                .code,
            "ambiguous_schema"
        );

        let missing = json!({"messages": []});
        assert_eq!(
            DecisionSchema::from_request(missing.as_object().unwrap())
                .unwrap_err()
                .code,
            "missing_schema"
        );
    }
}
