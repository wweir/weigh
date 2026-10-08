// Package slots is the answer-slot model: N options bound to N distinct single-token letters.
//
// A decision is read out as a distribution over a handful of answer slots. Binding each option
// to an uppercase letter keeps the readout constructive: the option text can be any string, but
// what the model is asked for is one token from a set the client chose, so the client can
// request that token's logprob by id instead of searching a top-N list.
//
// The letters are only a naming device. Three properties are load-bearing and every one of them
// is checked, by the backend's own tokenizer, before a request is scored:
//
//  1. each letter must tokenize to exactly one token,
//  2. that token must be appended by the letter alone (the answer-boundary proof),
//  3. the letters must be distinct tokens, so a slot id identifies one option.
package slots

// Letters are the option letters this service can bind, in slot order: A is slot 0, B is slot 1.
//
// Sixteen is a deliberate ceiling rather than a protocol limit: the contract is only
// constructive while every letter is a single round-trip token, and a wider set would need the
// per-tokenizer check before it could be promised.
const Letters = "ABCDEFGHIJKLMNOP"

// MinValues is the smallest option count: one option is not a decision.
const MinValues = 2

// MaxValues is the largest option count: one letter per slot.
const MaxValues = len(Letters)

// Letter returns the slot letter for an index. It panics out of range: every index on this path
// is bounded by MaxValues, so an out-of-range index is a divergence inside this module, and
// returning the empty string would report that the model scored nothing.
func Letter(index int) string {
	return Letters[index : index+1]
}
