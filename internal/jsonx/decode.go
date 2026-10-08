package jsonx

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Decode parses JSON with numbers kept as json.Number, so an integer keeps its exact literal.
//
// This matters for the schema layer: a u64 above i64::MAX is a legal JSON integer, and decoding
// into float64 would silently lose it. It also matters for rendering: a number carried as
// json.Number is written back as the caller wrote it.
func Decode(raw []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	// Anything after the first document is refused, whether it is a second value or garbage:
	// accepting the first and ignoring the rest would silently drop part of the caller's body.
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("trailing data after the JSON document")
	}
	return value, nil
}

// DecodeObject decodes one JSON object.
func DecodeObject(raw []byte) (map[string]any, error) {
	value, err := Decode(raw)
	if err != nil {
		return nil, err
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("body must be a JSON object")
	}
	return object, nil
}
