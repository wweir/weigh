package prompt

import (
	"strings"
	"testing"

	"github.com/wweir/weigh/internal/jsonx"
	"github.com/wweir/weigh/internal/template"
)

func decode(t *testing.T, raw string) any {
	t.Helper()
	value, err := jsonx.Decode([]byte(raw))
	if err != nil {
		t.Fatalf("test input is not valid JSON: %v", err)
	}
	return value
}

func TestUserMessageIsThePublishedPayload(t *testing.T) {
	row, err := ValidateRow(decode(t, `{"id":"r1","state":"the evidence","question":"the criterion","options":[{"id":"yes","description":"yes"},{"id":"no","description":"no"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	message, err := UserMessage(row)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"evidence": "the evidence", "criterion": "the criterion", "options": [{"letter": "A", "description": "yes"}, {"letter": "B", "description": "no"}]}`
	if message != want {
		t.Fatalf("user message =\n%s\nwant\n%s", message, want)
	}
}

// The evidence is written as the caller wrote it: non-ASCII stays UTF-8, control characters are
// escaped, and DEL is not a control character for JSON escaping.
func TestEvidenceKeepsItsEncoding(t *testing.T) {
	row, err := ValidateRow(decode(t, `{"id":"r","state":"caf\u00e9","question":"q","options":[{"id":"a","description":"a"},{"id":"b","description":"b"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	message, err := UserMessage(row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(message, `"evidence": "café"`) {
		t.Fatalf("evidence was not written as UTF-8: %s", message)
	}
}

func TestRenderPromptWithUsesTheFamilyWrapper(t *testing.T) {
	row, err := ValidateRow(decode(t, `{"id":"r","state":"e","question":"c","options":[{"id":"yes","description":"yes"},{"id":"no","description":"no"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := RenderPromptWith(template.Gemma4, row)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(rendered, "<bos><|turn>system\n"+DirectSystem) {
		t.Fatalf("rendered prompt does not start with the system turn: %q", rendered)
	}
	if !strings.HasSuffix(rendered, "<turn|>\n<|turn>model\n") {
		t.Fatalf("rendered prompt does not end at the assistant turn opener: %q", rendered)
	}
}

func TestMediaUserPayloadIsTheSameJSON(t *testing.T) {
	payload, err := MediaUserPayload("e", "c", []string{"yes", "no"})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"evidence": "e", "criterion": "c", "options": [{"letter": "A", "description": "yes"}, {"letter": "B", "description": "no"}]}`
	if payload != want {
		t.Fatalf("media payload = %s, want %s", payload, want)
	}
}

func TestValidateRowRefusals(t *testing.T) {
	cases := []struct {
		name string
		row  string
		want string
	}{
		{"not an object", `[]`, "Row must be a JSON object"},
		{"missing options", `{"id":"r","state":"s","question":"q"}`, `Row is missing fields: ["options"]`},
		{"empty id", `{"id":"","state":"s","question":"q","options":[{"id":"a","description":"a"},{"id":"b","description":"b"}]}`, "id and question must be nonempty strings"},
		{"empty state", `{"id":"r","state":"","question":"q","options":[{"id":"a","description":"a"},{"id":"b","description":"b"}]}`, "state must be a nonempty string"},
		// A decoded object has no key order, and the payload's byte layout is part of the
		// prompt, so a non-string state is refused at the door rather than at render time.
		{"object state", `{"id":"r","state":{"a":1},"question":"q","options":[{"id":"a","description":"a"},{"id":"b","description":"b"}]}`, "state must be a nonempty string"},
		{"array state", `{"id":"r","state":[1],"question":"q","options":[{"id":"a","description":"a"},{"id":"b","description":"b"}]}`, "state must be a nonempty string"},
		{"one option", `{"id":"r","state":"s","question":"q","options":[{"id":"a","description":"a"}]}`, "options must contain 2-16 entries"},
		{"seventeen options", `{"id":"r","state":"s","question":"q","options":[{"id":"1","description":"a"},{"id":"2","description":"a"},{"id":"3","description":"a"},{"id":"4","description":"a"},{"id":"5","description":"a"},{"id":"6","description":"a"},{"id":"7","description":"a"},{"id":"8","description":"a"},{"id":"9","description":"a"},{"id":"10","description":"a"},{"id":"11","description":"a"},{"id":"12","description":"a"},{"id":"13","description":"a"},{"id":"14","description":"a"},{"id":"15","description":"a"},{"id":"16","description":"a"},{"id":"17","description":"a"}]}`, "options must contain 2-16 entries"},
		{"option without description", `{"id":"r","state":"s","question":"q","options":[{"id":"a"},{"id":"b","description":"b"}]}`, "Each option needs string id and description fields"},
		{"duplicate option ids", `{"id":"r","state":"s","question":"q","options":[{"id":"a","description":"a"},{"id":"a","description":"b"}]}`, "Option IDs must be unique"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateRow(decode(t, tc.row))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}
