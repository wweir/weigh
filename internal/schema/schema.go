// Package schema is the decision schema every serving request must carry.
//
// The reader computes exactly one softmax over one finite answer-slot set, so the schema is
// required to describe exactly one field with exactly one JSON type. Anything else is refused
// at the door rather than answered halfway: a multi-field schema would need several readouts and
// a cross-field consistency rule this server does not implement, and a multi-type field would
// span heterogeneous tokenizations.
//
// The error codes are the published contract; callers and scripts match on them. The message
// text is written for a human and does not attempt to reproduce the previous implementation's
// debug formatting.
package schema

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/slots"
)

// MinValues is the smallest option count: one option is not a decision.
const MinValues = slots.MinValues

// MaxValues is the largest option count: the answer slot is one uppercase letter.
const MaxValues = slots.MaxValues

// fieldKeywords are the only keywords a field schema may use. Anything else is refused instead
// of ignored: a `pattern` or a `minimum` that is not enforced would let the response violate the
// schema the caller sent.
var fieldKeywords = []string{"type", "enum", "description", "title"}

// unionKeywords are refused wherever they appear.
var unionKeywords = []string{"anyOf", "oneOf", "allOf", "not", "$ref"}

// Error is a refusal with the code the contract publishes.
type Error struct {
	Code    string
	Message string
}

func (e *Error) Error() string { return e.Message }

func newError(code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Decision is the parsed schema: one field, and the allowed values in schema order, where
// Values[i] is answer slot slots.Letter(i).
type Decision struct {
	Field  string
	Values []any
}

// FromRequest pulls the decision schema out of a request body.
func FromRequest(body map[string]any) (*Decision, *Error) {
	located, err := locate(body)
	if err != nil {
		return nil, err
	}
	return FromJSONSchema(located)
}

// FromJSONSchema validates and normalises one JSON Schema object.
func FromJSONSchema(value any) (*Decision, *Error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, newError("schema_not_object", "the decision schema must be a JSON object")
	}
	schemaType, hasType := object["type"].(string)
	if !hasType || schemaType != "object" {
		return nil, newError("schema_not_object",
			"the decision schema must have type \"object\", %s", foundType(object["type"]))
	}
	if err := rejectUnions(object, "the decision schema"); err != nil {
		return nil, err
	}

	properties, ok := object["properties"].(map[string]any)
	if !ok {
		return nil, newError("schema_not_single_field", "the decision schema needs a `properties` object")
	}
	if len(properties) != 1 {
		return nil, newError("schema_not_single_field",
			"the decision schema must declare exactly one field; it declares %d. The service scores one distribution over one answer-slot set, so a multi-field decision is not supported",
			len(properties))
	}
	var field string
	var fieldSchema any
	for key, entry := range properties {
		field, fieldSchema = key, entry
	}

	required, ok := object["required"].([]any)
	if !ok {
		return nil, newError("schema_field_not_required",
			"the decision schema needs `required` naming its single field")
	}
	// Every entry must be a string: dropping a malformed one would accept ["verdict", 5] as if
	// it named exactly one field.
	names := make([]string, 0, len(required))
	for _, entry := range required {
		name, ok := entry.(string)
		if !ok {
			return nil, newError("schema_field_not_required", "`required` must be exactly [%q]", field)
		}
		names = append(names, name)
	}
	if len(names) != 1 || names[0] != field {
		return nil, newError("schema_field_not_required", "`required` must be exactly [%q]", field)
	}

	if extra, ok := object["additionalProperties"].(bool); !ok || extra {
		return nil, newError("schema_allows_extra_fields",
			"the decision schema must set `additionalProperties: false`; extra fields cannot be scored")
	}

	values, err := fieldValues(field, fieldSchema)
	if err != nil {
		return nil, err
	}
	return &Decision{Field: field, Values: values}, nil
}

// Param names the request field that carried the schema, for an error's `param`.
func Param(body map[string]any) string {
	if _, present := body["response_format"]; present {
		return "response_format.json_schema.schema"
	}
	if _, present := body["guided_json"]; present {
		return "guided_json"
	}
	return "schema"
}

// Label is the model-facing option text and the label echoed back to the caller. For a scalar
// enum they are the same string; the value itself is what the caller gets back, so no
// id/description mapping can drift.
func Label(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case json.Number:
		return typed.String()
	case bool:
		return strconv.FormatBool(typed)
	default:
		// Unreachable: every value on this path came from fieldValues, which accepts only these
		// three shapes. A panic rather than a `%v` rendering, which would put a non-JSON label
		// into the prompt and into the echoed response.
		panic(fmt.Sprintf("schema: label for an unsupported value %T", value))
	}
}

// Content is the strict JSON content for the winning value: {"<field>": <value>}.
func (d *Decision) Content(choice any) string {
	rendered, err := jsonx.Compact(jsonx.Object{{Key: d.Field, Value: choice}})
	if err != nil {
		// Unreachable: the single value was validated when the schema was parsed.
		panic("schema: content for an unrenderable value: " + err.Error())
	}
	return rendered
}

// Labels returns every option label in slot order.
func (d *Decision) Labels() []string {
	out := make([]string, len(d.Values))
	for i, value := range d.Values {
		out[i] = Label(value)
	}
	return out
}

// locate finds the one decision schema in a request body.
//
// Three sources are accepted, because three ecosystems carry it differently: the OpenAI SDK's
// response_format.json_schema.schema, a top-level `schema`, and the vLLM/SGLang native
// `guided_json`. More than one is an error rather than a silent precedence rule.
func locate(body map[string]any) (any, *Error) {
	type source struct {
		name  string
		value any
	}
	var sources []source

	if value, present := body["schema"]; present {
		sources = append(sources, source{"`schema`", value})
	}
	if format, present := body["response_format"]; present {
		object, ok := format.(map[string]any)
		if !ok {
			return nil, newError("unsupported_response_format",
				"response_format must be an object; served decisions implement `json_schema` only")
		}
		kind, _ := object["type"].(string)
		if kind != "json_schema" {
			return nil, newError("unsupported_response_format",
				"response_format.type is %q; served decisions implement `json_schema` only", kind)
		}
		schema, ok := object["json_schema"].(map[string]any)
		if !ok {
			return nil, newError("missing_schema",
				"response_format.json_schema.schema is missing; a decision schema is required")
		}
		value, present := schema["schema"]
		if !present {
			return nil, newError("missing_schema",
				"response_format.json_schema.schema is missing; a decision schema is required")
		}
		sources = append(sources, source{"`response_format.json_schema.schema`", value})
	}
	if guided, present := body["guided_json"]; present {
		if _, ok := guided.(map[string]any); !ok {
			// vLLM also accepts a path or a registered name here. Neither is resolvable from
			// this process, so it is refused by name instead of being guessed at.
			return nil, newError("unsupported_guided_json",
				"`guided_json` must be the schema object itself; a path or a registered name cannot be resolved by this server")
		}
		sources = append(sources, source{"`guided_json`", guided})
	}

	switch len(sources) {
	case 0:
		return nil, newError("missing_schema",
			"a decision schema is required: send response_format.json_schema.schema, a top-level `schema`, or `guided_json`")
	case 1:
		return sources[0].value, nil
	default:
		names := make([]string, len(sources))
		for i, entry := range sources {
			names[i] = entry.name
		}
		return nil, newError("ambiguous_schema", "send the decision schema once; found it in %s",
			strings.Join(names, ", "))
	}
}

func rejectUnions(object map[string]any, what string) *Error {
	for _, keyword := range unionKeywords {
		if _, present := object[keyword]; present {
			return newError("schema_union_unsupported",
				"%s uses `%s`; a decision must have exactly one type", what, keyword)
		}
	}
	return nil
}

func fieldValues(field string, fieldSchema any) ([]any, *Error) {
	object, ok := fieldSchema.(map[string]any)
	if !ok {
		return nil, newError("schema_field_type_unsupported", "field %q must be a JSON object", field)
	}
	if err := rejectUnions(object, fmt.Sprintf("field %q", field)); err != nil {
		return nil, err
	}
	for keyword := range object {
		if !contains(fieldKeywords, keyword) {
			return nil, newError("schema_keyword_unsupported",
				"field %q uses `%s`; this server honors only %s", field, keyword, strings.Join(fieldKeywords, ", "))
		}
	}

	fieldType, ok := object["type"].(string)
	if !ok {
		return nil, newError("schema_field_type_unsupported",
			"field %q must declare exactly one `type`", field)
	}
	enumeration, hasEnum := object["enum"].([]any)

	switch fieldType {
	case "boolean":
		if hasEnum {
			values := make([]bool, 0, len(enumeration))
			for _, entry := range enumeration {
				flag, ok := entry.(bool)
				if !ok {
					return nil, newError("schema_enum_type_mismatch",
						"a boolean field's `enum` must be exactly [false, true]; found %s", render(enumeration))
				}
				values = append(values, flag)
			}
			// Exactly `false` and `true`: [false, false] would declare one option while this
			// endpoint answers with two slots.
			if len(values) != 2 || values[0] == values[1] {
				return nil, newError("schema_enum_type_mismatch",
					"a boolean field's `enum` must be exactly [false, true]; found %s", render(enumeration))
			}
		}
		// `false` is slot A and `true` is slot B, so the slot order is fixed by the type and
		// does not depend on the order an enum happened to be written in.
		return []any{false, true}, nil

	case "string", "integer":
		if !hasEnum {
			return nil, newError("schema_enum_required",
				"a %s field must list its allowed values in `enum`", fieldType)
		}
		if len(enumeration) < MinValues || len(enumeration) > MaxValues {
			return nil, newError("schema_value_count",
				"`enum` must hold %d..=%d values; found %d", MinValues, MaxValues, len(enumeration))
		}
		values := make([]any, 0, len(enumeration))
		seen := make(map[string]bool, len(enumeration))
		for _, entry := range enumeration {
			key, ok := typedKey(fieldType, entry)
			if !ok {
				return nil, newError("schema_enum_type_mismatch",
					"field %q has type %q but `enum` contains %s; one field must have one type",
					field, fieldType, render(entry))
			}
			if seen[key] {
				return nil, newError("schema_value_duplicate", "`enum` repeats the value %s", render(entry))
			}
			seen[key] = true
			values = append(values, entry)
		}
		return values, nil

	default:
		return nil, newError("schema_field_type_unsupported",
			"field %q has type %q; supported types are string, integer, boolean", field, fieldType)
	}
}

// typedKey reports whether entry matches the field's type, and returns a canonical key for the
// duplicate check.
func typedKey(fieldType string, entry any) (string, bool) {
	switch fieldType {
	case "string":
		text, ok := entry.(string)
		if !ok {
			return "", false
		}
		return "s:" + text, true
	default:
		number, ok := entry.(json.Number)
		if !ok || !isJSONInteger(number) {
			return "", false
		}
		return "n:" + number.String(), true
	}
}

// isJSONInteger accepts every JSON integer literal serde_json would keep as an i64 or a u64.
//
// A float is still not an integer, even when its value is integral: `1.0` names a different
// type, and accepting it would let one field span two tokenizations.
func isJSONInteger(number json.Number) bool {
	text := number.String()
	if strings.ContainsAny(text, ".eE") {
		return false
	}
	if strings.HasPrefix(text, "-") {
		_, err := strconv.ParseInt(text, 10, 64)
		return err == nil
	}
	_, err := strconv.ParseUint(text, 10, 64)
	return err == nil
}

// foundType describes a present-but-unusable `type` for an error message.
func foundType(value any) string {
	if text, ok := value.(string); ok {
		return fmt.Sprintf("found %q", text)
	}
	if value == nil {
		return `found no "type"`
	}
	return fmt.Sprintf("found %s", render(value))
}

func render(value any) string {
	rendered, err := jsonx.Compact(value)
	if err != nil {
		return fmt.Sprintf("%v", value)
	}
	return rendered
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
