package schema

import (
	"strings"
	"testing"

	"github.com/wweir/weigh/internal/jsonx"
)

// from parses a schema written as JSON text, so a test states the caller's view of the wire
// format rather than a Go map that already lost the integer literal.
func from(t *testing.T, raw string) (*Decision, *Error) {
	t.Helper()
	value, err := jsonx.Decode([]byte(raw))
	if err != nil {
		t.Fatalf("test schema is not valid JSON: %v", err)
	}
	return FromJSONSchema(value)
}

func objectSchema(field string) string {
	return `{"type":"object","properties":{"verdict":` + field +
		`},"required":["verdict"],"additionalProperties":false}`
}

func TestStringEnumMapsToSlotsInSchemaOrder(t *testing.T) {
	decision, schemaErr := from(t, objectSchema(`{"type":"string","enum":["yes","no"]}`))
	if schemaErr != nil {
		t.Fatal(schemaErr)
	}
	if decision.Field != "verdict" {
		t.Errorf("field = %q", decision.Field)
	}
	if got := decision.Labels(); len(got) != 2 || got[0] != "yes" || got[1] != "no" {
		t.Errorf("labels = %v", got)
	}
	if got := decision.Content(decision.Values[1]); got != `{"verdict":"no"}` {
		t.Errorf("content = %s", got)
	}
}

func TestIntegerAndBooleanValuesKeepTheirJSONType(t *testing.T) {
	integers, schemaErr := from(t, objectSchema(`{"type":"integer","enum":[1,2,3]}`))
	if schemaErr != nil {
		t.Fatal(schemaErr)
	}
	if got := integers.Content(integers.Values[2]); got != `{"verdict":3}` {
		t.Errorf("integer content = %s", got)
	}
	booleans, schemaErr := from(t, objectSchema(`{"type":"boolean"}`))
	if schemaErr != nil {
		t.Fatal(schemaErr)
	}
	if got := booleans.Labels(); len(got) != 2 || got[0] != "false" || got[1] != "true" {
		t.Errorf("boolean labels = %v", got)
	}
}

func TestRefusalCodes(t *testing.T) {
	cases := []struct {
		name   string
		schema string
		code   string
	}{
		{"second field", `{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"boolean"}},"required":["a"],"additionalProperties":false}`, "schema_not_single_field"},
		{"no properties", `{"type":"object","required":["a"],"additionalProperties":false}`, "schema_not_single_field"},
		{"multi-type field", objectSchema(`{"type":["string","integer"]}`), "schema_field_type_unsupported"},
		{"union field", objectSchema(`{"anyOf":[{"type":"string"}]}`), "schema_union_unsupported"},
		{"string without enum", objectSchema(`{"type":"string"}`), "schema_enum_required"},
		{"mixed enum types", objectSchema(`{"type":"string","enum":["a",1]}`), "schema_enum_type_mismatch"},
		{"duplicate enum", objectSchema(`{"type":"string","enum":["a","a"]}`), "schema_value_duplicate"},
		{"too many options", objectSchema(`{"type":"string","enum":["v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16"]}`), "schema_value_count"},
		{"one option", objectSchema(`{"type":"string","enum":["only"]}`), "schema_value_count"},
		{"extra properties", `{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"]}`, "schema_allows_extra_fields"},
		{"empty required", `{"type":"object","properties":{"a":{"type":"boolean"}},"required":[],"additionalProperties":false}`, "schema_field_not_required"},
		{"malformed required", `{"type":"object","properties":{"verdict":{"type":"boolean"}},"required":["verdict",5],"additionalProperties":false}`, "schema_field_not_required"},
		{"unenforced keyword", objectSchema(`{"type":"string","enum":["a","b"],"pattern":"^a$"}`), "schema_keyword_unsupported"},
		{"not an object", `"nope"`, "schema_not_object"},
		{"wrong type", `{"type":"array"}`, "schema_not_object"},
		{"unsupported field type", objectSchema(`{"type":"number","enum":[1,2]}`), "schema_field_type_unsupported"},
		{"boolean enum with one value twice", objectSchema(`{"type":"boolean","enum":[false,false]}`), "schema_enum_type_mismatch"},
		{"integer enum with a float", objectSchema(`{"type":"integer","enum":[1.0,2]}`), "schema_enum_type_mismatch"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, schemaErr := from(t, tc.schema)
			if schemaErr == nil {
				t.Fatalf("expected a refusal with code %s", tc.code)
			}
			if schemaErr.Code != tc.code {
				t.Fatalf("code = %s, want %s (%s)", schemaErr.Code, tc.code, schemaErr.Message)
			}
		})
	}
}

// Listing both booleans is fine in either order: the slot order is fixed by the type, not by the
// order the enum happened to be written in.
func TestBooleanSlotOrderIsFixedByType(t *testing.T) {
	for _, enumeration := range []string{"[false,true]", "[true,false]"} {
		decision, schemaErr := from(t, objectSchema(`{"type":"boolean","enum":`+enumeration+`}`))
		if schemaErr != nil {
			t.Fatalf("%s: %v", enumeration, schemaErr)
		}
		if got := decision.Labels(); got[0] != "false" || got[1] != "true" {
			t.Errorf("%s: labels = %v, want [false true]", enumeration, got)
		}
	}
}

// serde_json stores integers above i64::MAX as u64; the schema layer must accept them rather
// than silently losing the value to a float.
func TestIntegerEnumsAcceptEveryJSONInteger(t *testing.T) {
	decision, schemaErr := from(t, objectSchema(`{"type":"integer","enum":[18446744073709551615,2]}`))
	if schemaErr != nil {
		t.Fatalf("u64 max must be accepted: %v", schemaErr)
	}
	if got := decision.Content(decision.Values[0]); got != `{"verdict":18446744073709551615}` {
		t.Errorf("content = %s", got)
	}
	negative, schemaErr := from(t, objectSchema(`{"type":"integer","enum":[-1,2]}`))
	if schemaErr != nil {
		t.Fatalf("negative integers must be accepted: %v", schemaErr)
	}
	if got := negative.Content(negative.Values[0]); got != `{"verdict":-1}` {
		t.Errorf("content = %s", got)
	}
}

func TestSchemaIsLocatedFromEitherPlacement(t *testing.T) {
	request := func(raw string) map[string]any {
		t.Helper()
		object, err := jsonx.DecodeObject([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		return object
	}
	viaResponseFormat := request(`{"response_format":{"type":"json_schema","json_schema":{"name":"d","schema":` + objectSchema(`{"type":"boolean"}`) + `}}}`)
	if decision, err := FromRequest(viaResponseFormat); err != nil {
		t.Fatalf("response_format placement: %v", err)
	} else if decision.Field != "verdict" {
		t.Fatalf("field = %q", decision.Field)
	}
	if _, err := FromRequest(request(`{"schema":` + objectSchema(`{"type":"boolean"}`) + `}`)); err != nil {
		t.Fatalf("top-level placement: %v", err)
	}
	if _, err := FromRequest(request(`{"guided_json":` + objectSchema(`{"type":"boolean"}`) + `}`)); err != nil {
		t.Fatalf("guided_json placement: %v", err)
	}

	both := request(`{"schema":` + objectSchema(`{"type":"boolean"}`) + `,"response_format":{"type":"json_schema","json_schema":{"schema":{}}}}`)
	_ = expectCode(t, both, "ambiguous_schema")
	_ = expectCode(t, request(`{"messages":[]}`), "missing_schema")
	// A response_format that is not json_schema is refused even when a `schema` is also sent.
	wrongFormat := request(`{"schema":` + objectSchema(`{"type":"boolean"}`) + `,"response_format":{"type":"text"}}`)
	_ = expectCode(t, wrongFormat, "unsupported_response_format")
	// guided_json as a path or a registered name cannot be resolved by this process, and the
	// message must say so: the remedy differs from an unrecognised schema.
	byName := expectCode(t, request(`{"guided_json":"schema.json"}`), "unsupported_guided_json")
	if !strings.Contains(byName.Message, "cannot be resolved") {
		t.Fatalf("message = %q", byName.Message)
	}
}

func expectCode(t *testing.T, body map[string]any, code string) *Error {
	t.Helper()
	_, schemaErr := FromRequest(body)
	if schemaErr == nil || schemaErr.Code != code {
		t.Fatalf("code = %v, want %s", schemaErr, code)
	}
	return schemaErr
}
