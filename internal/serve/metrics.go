package serve

import (
	"strconv"
	"strings"
	"sync/atomic"
)

// readoutBuckets are the histogram bounds for semif_readout_seconds, in seconds. A readout is one
// backend call, so these are spaced around LAN round-trip times rather than model latency.
var readoutBuckets = [8]float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0}

// metrics are process-lifetime counters, exposed at /metrics.
//
// Deliberately small: every field is something an operator would act on. There is no per-route
// breakdown because every route goes through the same readout path, and no summary quantiles
// because these histogram buckets already support histogram_quantile.
type metrics struct {
	requests        atomic.Uint64
	inflight        atomic.Int64
	finished        atomic.Uint64
	clientErrors    atomic.Uint64
	serverErrors    atomic.Uint64
	readouts        atomic.Uint64
	fallbacks       atomic.Uint64
	backendTimeouts atomic.Uint64
	backendErrors   atomic.Uint64
	// buckets is cumulative: bucket i counts readouts at or under readoutBuckets[i].
	buckets [len(readoutBuckets)]atomic.Uint64
	// readoutMicros is readout seconds in microseconds, so the sum needs no float atomic.
	readoutMicros atomic.Uint64
}

func (m *metrics) observeReadout(seconds float64, fallback bool) {
	m.readouts.Add(1)
	if fallback {
		m.fallbacks.Add(1)
	}
	if seconds < 0 {
		seconds = 0
	}
	m.readoutMicros.Add(uint64(seconds * 1_000_000))
	for index, bound := range readoutBuckets {
		if seconds <= bound {
			m.buckets[index].Add(1)
		}
	}
}

// Render is the Prometheus text exposition, in full for every family, so a counter sitting at
// zero is still visible rather than inferred from absence.
func (m *metrics) Render() string {
	var out strings.Builder
	pushCounter(&out, "semif_requests_total", "HTTP requests received.", m.requests.Load())
	pushCounter(&out, "semif_responses_total", "HTTP responses sent.", m.finished.Load())
	pushCounter(&out, "semif_client_errors_total", "Responses with a 4xx status.", m.clientErrors.Load())
	pushCounter(&out, "semif_server_errors_total", "Responses with a 5xx status.", m.serverErrors.Load())
	pushCounter(&out, "semif_readouts_total", "Decisions produced.", m.readouts.Load())
	pushCounter(&out, "semif_readout_fallbacks_total",
		"Readouts that missed the backend's top-N and used /generative_scoring. This rate is measured to rise with load, which is why it is exported rather than hidden.",
		m.fallbacks.Load())
	pushCounter(&out, "semif_backend_timeouts_total", "Readouts that spent the whole budget without an answer.", m.backendTimeouts.Load())
	pushCounter(&out, "semif_backend_errors_total", "Readouts that failed without spending the budget.", m.backendErrors.Load())
	pushGauge(&out, "semif_inflight_requests", "Requests currently being handled.", uint64(max(m.inflight.Load(), 0)))

	out.WriteString("# HELP semif_readout_seconds Wall time of one backend readout.\n# TYPE semif_readout_seconds histogram\n")
	for index, bound := range readoutBuckets {
		out.WriteString(`semif_readout_seconds_bucket{le="` + formatFloat(bound) + `"} ` + formatUint(m.buckets[index].Load()) + "\n")
	}
	readouts := m.readouts.Load()
	out.WriteString(`semif_readout_seconds_bucket{le="+Inf"} ` + formatUint(readouts) + "\n")
	out.WriteString("semif_readout_seconds_sum " + formatMicros(m.readoutMicros.Load()) + "\n")
	out.WriteString("semif_readout_seconds_count " + formatUint(readouts) + "\n")
	return out.String()
}

func pushCounter(out *strings.Builder, name, help string, value uint64) {
	out.WriteString("# HELP " + name + " " + help + "\n# TYPE " + name + " counter\n" + name + " " + formatUint(value) + "\n")
}

func pushGauge(out *strings.Builder, name, help string, value uint64) {
	out.WriteString("# HELP " + name + " " + help + "\n# TYPE " + name + " gauge\n" + name + " " + formatUint(value) + "\n")
}

func formatUint(value uint64) string { return strconv.FormatUint(value, 10) }

// formatFloat never uses exponent notation: `0.005` and `1`, not `5e-03` and `1e+00`.
func formatFloat(value float64) string { return strconv.FormatFloat(value, 'f', -1, 64) }

func formatMicros(micros uint64) string { return formatFloat(float64(micros) / 1_000_000) }
