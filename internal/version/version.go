// Package version carries the build identity injected by the Makefile's -ldflags.
//
// The values are recorded in every response's semif block, so a number produced by one build
// can be attributed to that build. They default to explicit placeholders rather than empty
// strings: an unattributed artifact must say so, not look like a build with no revision.
package version

import "fmt"

// Injected at build time:
//
//	-X github.com/wweir/weigh/internal/version.Version=...
//	-X github.com/wweir/weigh/internal/version.Revision=...
//	-X github.com/wweir/weigh/internal/version.BuildDate=...
var (
	Version   = "dev"
	Revision  = "unknown"
	BuildDate = "unknown"
)

// String is the one-line identity logged at startup.
func String() string {
	return fmt.Sprintf("%s (revision %s, built %s)", Version, Revision, BuildDate)
}
