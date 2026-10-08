package readout

import (
	"fmt"
	"math"
)

// Softmax normalises scores over the declared answer slots.
//
// The maximum is subtracted before exponentiating, both to keep the result finite for large
// scores and to match the arithmetic a reference implementation would perform; the sum is taken
// left to right, so the same input yields the same output.
func Softmax(values []float64) ([]float64, error) {
	if len(values) < 2 {
		return nil, fmt.Errorf("Need at least two finite scores")
	}
	maximum := math.Inf(-1)
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("Need at least two finite scores")
		}
		maximum = math.Max(maximum, value)
	}
	weights := make([]float64, len(values))
	total := 0.0
	for i, value := range values {
		weights[i] = math.Exp(value - maximum)
		total += weights[i]
	}
	for i := range weights {
		weights[i] /= total
	}
	return weights, nil
}
