package jsonx

import (
	"testing"
)

func TestDumpUsesThePublishedPromptLayout(t *testing.T) {
	value := Object{
		{"evidence", "café😀"},
		{"criterion", "y"},
		{"options", []any{
			Object{{"letter", "A"}, {"description", "yes"}},
			Object{{"letter", "B"}, {"description", "no"}},
		}},
	}
	got, err := Dump(value)
	if err != nil {
		t.Fatal(err)
	}
	// `", "` and `": "`: the layout the published direct-options-v1 payload used. Non-ASCII
	// stays as UTF-8.
	want := `{"evidence": "café😀", "criterion": "y", "options": [{"letter": "A", "description": "yes"}, {"letter": "B", "description": "no"}]}`
	if got != want {
		t.Fatalf("Dump() = %s, want %s", got, want)
	}
}

func TestCompactHasNoSpaces(t *testing.T) {
	got, err := Compact(Object{{"verdict", "no"}})
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"verdict":"no"}` {
		t.Fatalf("Compact() = %s", got)
	}
}

func TestControlCharactersAreEscapedAndDELIsNot(t *testing.T) {
	cases := map[string]string{
		"a\"b\\c\nd\te\x01f": `"a\"b\\c\nd\te\u0001f"`,
		"\x7f":               "\"\x7f\"",
		"\b\f\r":             `"\b\f\r"`,
	}
	for in, want := range cases {
		got, err := Dump(in)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("Dump(%q) = %s, want %s", in, got, want)
		}
	}
}

// A map has no order, so rendering one would silently reorder a prompt for the same request.
func TestUnorderedObjectIsRefused(t *testing.T) {
	if _, err := Dump(map[string]any{"a": 1}); err == nil {
		t.Fatal("a map must be refused, not sorted")
	}
}

func TestIntListIsTheRequestForm(t *testing.T) {
	if got := IntList([]uint32{1, 23}); got != "[1, 23]" {
		t.Errorf("IntList = %s", got)
	}
	if got := IntList(nil); got != "[]" {
		t.Errorf("IntList(nil) = %s", got)
	}
	if got := IntList([]uint32{0}); got != "[0]" {
		t.Errorf("IntList([0]) = %s", got)
	}
}

// A value this renderer does not know is refused by name rather than guessed at.
func TestUnknownValueIsRefused(t *testing.T) {
	if _, err := Dump([]string{"a"}); err == nil {
		t.Error("an unhandled slice type must be refused")
	}
}

// Decode keeps an integer's literal, and Compact writes it back unchanged.
func TestNumberKeepsItsLiteral(t *testing.T) {
	decoded, err := Decode([]byte(`{"v": 18446744073709551615}`))
	if err != nil {
		t.Fatal(err)
	}
	value := decoded.(map[string]any)["v"]
	got, err := Compact(Object{{"verdict", value}})
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"verdict":18446744073709551615}` {
		t.Fatalf("Compact() = %s", got)
	}
}

// Anything after the first JSON document is refused, whether it is a second value or garbage:
// accepting the first and ignoring the rest would silently drop part of a caller's body.
func TestDecodeRefusesTrailingData(t *testing.T) {
	for _, raw := range []string{`{"a":1} garbage`, `{"a":1} {"b":2}`, `1 2`} {
		if _, err := Decode([]byte(raw)); err == nil {
			t.Errorf("Decode(%q) must be refused", raw)
		}
	}
	if _, err := Decode([]byte("  {\"a\":1}  \n")); err != nil {
		t.Errorf("trailing whitespace is not trailing data: %v", err)
	}
}
