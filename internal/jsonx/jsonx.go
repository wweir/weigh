// Package jsonx renders the JSON text this service puts in a prompt or a response body.
//
// It exists because the rendering is part of the contract, not a formatting detail: the prompt
// a model is asked to apply and the `{"<field>": <value>}` a caller receives must be the same
// strings they were before this service was ported, or `prompt_version: direct-options-v1`
// would name two different prompts. encoding/json cannot express either shape — it emits
// compact separators, escapes HTML, and cannot preserve a caller's object key order — so the
// few shapes this service needs are written out here.
//
// Two separator styles are provided because the two artifacts genuinely differ: the prompt
// payload is rendered with `", "` and `": "` (the layout the published prompt used), while
// response content is compact.
//
// Only the value kinds this service actually carries are rendered: strings, booleans,
// `json.Number` (which preserves a caller's integer literal), ints it constructs itself,
// ordered objects, arrays and pre-rendered text. Anything else is refused by name rather than
// guessed at.
package jsonx

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Field is one member of an ordered object.
type Field struct {
	Key   string
	Value any
}

// Object is a JSON object whose key order is the order the fields were listed in. encoding/json
// would either sort the keys (map) or lose them to a struct, and the payload's byte layout is
// observable.
type Object []Field

// Dump renders v the way the prompt payload is laid out: `", "` between members and `": "`
// after a key.
func Dump(v any) (string, error) {
	var out strings.Builder
	if err := write(&out, v, spaced); err != nil {
		return "", err
	}
	return out.String(), nil
}

// Compact renders v as response content: no spaces after `,` or `:`.
func Compact(v any) (string, error) {
	var out strings.Builder
	if err := write(&out, v, compact); err != nil {
		return "", err
	}
	return out.String(), nil
}

// IntList renders a token-id list for a request body: `[1, 23]`, `[]`.
func IntList(values []uint32) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.FormatUint(uint64(value), 10)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// separators selects the two layouts. They differ only in the space after `,` and `:`.
type separators struct {
	comma, colon string
}

var (
	spaced  = separators{comma: ", ", colon: ": "}
	compact = separators{comma: ",", colon: ":"}
)

func write(out *strings.Builder, v any, sep separators) error {
	switch value := v.(type) {
	case nil:
		out.WriteString("null")
	case bool:
		out.WriteString(strconv.FormatBool(value))
	case string:
		writeString(out, value)
	case json.Number:
		// The literal text the caller sent. Range and syntax were checked by the schema
		// layer, so passing it through cannot invent a value.
		out.WriteString(value.String())
	case int:
		out.WriteString(strconv.Itoa(value))
	case Object:
		out.WriteByte('{')
		for i, field := range value {
			if i > 0 {
				out.WriteString(sep.comma)
			}
			writeString(out, field.Key)
			out.WriteString(sep.colon)
			if err := write(out, field.Value, sep); err != nil {
				return err
			}
		}
		out.WriteByte('}')
	case []any:
		out.WriteByte('[')
		for i, item := range value {
			if i > 0 {
				out.WriteString(sep.comma)
			}
			if err := write(out, item, sep); err != nil {
				return err
			}
		}
		out.WriteByte(']')
	case map[string]any:
		// Unordered input is refused rather than silently sorted: a sorted object would
		// change the prompt bytes for the same request, which is the one thing this package
		// exists to prevent. Callers pass Object, or a raw []byte via Raw.
		return fmt.Errorf("jsonx: unordered object of %d keys; use jsonx.Object to fix the order", len(value))
	case Raw:
		out.Write([]byte(value))
	default:
		return fmt.Errorf("jsonx: cannot render %T", v)
	}
	return nil
}

// Raw is pre-rendered JSON text emitted verbatim, for a caller that already holds the exact
// bytes (an evidence payload read from the request, for example).
type Raw string

// writeString is the published escaping: quote, backslash and every C0 control character are
// escaped; everything else — including DEL and non-ASCII — is written as UTF-8.
func writeString(out *strings.Builder, text string) {
	out.WriteByte('"')
	for i := 0; i < len(text); i++ {
		ch := text[i]
		switch ch {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		default:
			if ch < 0x20 {
				fmt.Fprintf(out, `\u%04x`, ch)
			} else {
				out.WriteByte(ch)
			}
		}
	}
	out.WriteByte('"')
}
