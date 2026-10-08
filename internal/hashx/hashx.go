// Package hashx computes the provenance digests recorded in every response.
//
// A number without its provenance is not evidence: two different readouts must never be pooled,
// so each response carries the digest of what was actually scored.
package hashx

import (
	"crypto/sha256"
	"encoding/hex"
)

// Sha256Hex is the lowercase hex digest of b.
func Sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
