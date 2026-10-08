package media

import (
	"strconv"
	"strings"
	"testing"

	"github.com/wweir/weigh/internal/jsonx"
)

// limits is the enabled configuration the Rust tests used: a 1 KB per-image ceiling and two
// images allowed, so a size or count test does not need a megabyte of fixture.
func limits() Limits {
	return Limits{AllowMedia: true, MaxImages: 2, MaxMediaBytes: 1024}
}

func message(t *testing.T, raw string) map[string]any {
	t.Helper()
	decoded, err := jsonx.Decode([]byte(raw))
	if err != nil {
		t.Fatalf("test input is not valid JSON: %v", err)
	}
	object, ok := decoded.(map[string]any)
	if !ok {
		t.Fatalf("test input is not a JSON object: %s", raw)
	}
	return object
}

func read(t *testing.T, raw string, index int, limits Limits) (*Content, *Error) {
	t.Helper()
	return ReadContent(message(t, raw), index, limits)
}

func TestStringContentIsTextOnly(t *testing.T) {
	content, err := read(t, `{"role":"user","content":"hello"}`, 0, limits())
	if err != nil {
		t.Fatal(err)
	}
	if content.Text != "hello" || len(content.Media) != 0 {
		t.Fatalf("content = %+v", content)
	}
}

func TestDataURIIsDecodedAndFingerprinted(t *testing.T) {
	content, err := read(t, `{"role":"user","content":[
		{"type":"text","text":"what is this?"},
		{"type":"image_url","image_url":{"url":"`+ProbeImageDataURI+`"}}]}`, 0, limits())
	if err != nil {
		t.Fatal(err)
	}
	if content.Text != "what is this?" || len(content.Media) != 1 {
		t.Fatalf("content = %+v", content)
	}
	image := content.Media[0]
	if image.Mime != "image/png" {
		t.Errorf("mime = %q", image.Mime)
	}
	if image.Size == nil || *image.Size != 67 {
		t.Errorf("size = %v", image.Size)
	}
	if image.SHA256 == nil || *image.SHA256 != "ebf4f635a17d10d6eb46ba680b70142419aa3220f228001a036d311a22ee9d2a" {
		t.Errorf("sha256 = %v", image.SHA256)
	}
	if got := image.Provenance()["url"]; got != nil {
		t.Errorf("a locally hashed image must not restate its url: %v", got)
	}
}

func TestRemoteURLIsRefusedWithoutTheOptIn(t *testing.T) {
	_, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"https://example.invalid/x.png"}}]}`, 3, limits())
	refusal := err
	if refusal.Code != "media_remote_not_allowed" {
		t.Fatalf("code = %s (%s)", refusal.Code, refusal.Message)
	}
	if !strings.Contains(refusal.Message, "messages[3]") {
		t.Fatalf("message must name the message index: %s", refusal.Message)
	}
}

func TestRemoteURLCarriesNoByteHashWithTheOptIn(t *testing.T) {
	allowed := limits()
	allowed.AllowRemote = true
	content, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"https://example.invalid/x.png"}}]}`, 0, allowed)
	if err != nil {
		t.Fatal(err)
	}
	image := content.Media[0]
	if image.SHA256 != nil || image.Size != nil {
		t.Fatalf("a remote URL has no local fingerprint: %+v", image)
	}
	if got := image.Provenance()["url"]; got != "https://example.invalid/x.png" {
		t.Fatalf("provenance url = %v", got)
	}
}

func TestMediaWithoutTheMasterSwitchIsRefusedByName(t *testing.T) {
	_, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"`+ProbeImageDataURI+`"}}]}`, 1, DefaultLimits())
	if refusal := err; refusal.Code != "media_not_enabled" {
		t.Fatalf("code = %s", refusal.Code)
	}
}

func TestReferenceThatIsNeitherDataURINorHTTPIsRefused(t *testing.T) {
	cases := []struct{ reference, want string }{
		{"file:///etc/passwd", "filesystem path"},
		{"data:image/png,notbase64", "base64"},
		{"data:image/png;base64,", "zero bytes"},
	}
	for _, tc := range cases {
		_, refusal := read(t, `{"role":"user","content":[
			{"type":"image_url","image_url":{"url":"`+tc.reference+`"}}]}`, 0, limits())
		if refusal == nil || refusal.Code != "invalid_media" {
			t.Fatalf("%s: err = %v", tc.reference, refusal)
		}
		if !strings.Contains(refusal.Message, tc.want) {
			t.Errorf("%s: message = %q, want it to mention %q", tc.reference, refusal.Message, tc.want)
		}
	}
}

// MIME judgement belongs to the backend processor: this service records what the caller declared
// and forwards the reference unchanged.
func TestMimeTheBackendMayRejectIsStillPassedThrough(t *testing.T) {
	const reference = "data:application/pdf;base64,AAAA"
	content, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"`+reference+`"}}]}`, 0, limits())
	if err != nil {
		t.Fatal(err)
	}
	if content.Media[0].Mime != "application/pdf" || content.Media[0].Reference != reference {
		t.Fatalf("image = %+v", content.Media[0])
	}
}

func TestForwardedPartIsTheReferenceVerbatim(t *testing.T) {
	content, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"`+ProbeImageDataURI+`"}}]}`, 0, limits())
	if err != nil {
		t.Fatal(err)
	}
	image := content.Media[0]
	part := image.ContentPart()
	if image.Reference != ProbeImageDataURI {
		t.Fatalf("reference was rewritten: %q", image.Reference)
	}
	rendered, dumpErr := jsonx.Dump(part)
	if dumpErr != nil {
		t.Fatal(dumpErr)
	}
	if !strings.Contains(rendered, ProbeImageDataURI) {
		t.Fatalf("the forwarded part does not carry the reference verbatim: %s", rendered)
	}
}

func TestOversizedImageIsRefused(t *testing.T) {
	small := limits()
	small.MaxMediaBytes = 8
	_, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"`+ProbeImageDataURI+`"}}]}`, 0, small)
	if refusal := err; refusal.Code != "media_too_large" {
		t.Fatalf("code = %s (%s)", refusal.Code, refusal.Message)
	}
}

func TestMoreImagesThanAllowedIsRefused(t *testing.T) {
	one := limits()
	one.MaxImages = 1
	_, err := read(t, `{"role":"user","content":[
		{"type":"image_url","image_url":{"url":"`+ProbeImageDataURI+`"}},
		{"type":"image_url","image_url":{"url":"`+ProbeImageDataURI+`"}}]}`, 0, one)
	if refusal := err; refusal.Code != "too_many_images" {
		t.Fatalf("code = %s", refusal.Code)
	}
}

// Refused rather than dropped, at every shape: skipping a part would answer a question the
// caller did not ask, with nothing in the response saying so.
func TestUnreadablePartIsRefusedByIndexAndNamed(t *testing.T) {
	cases := []struct {
		name    string
		content string
		index   int
		want    string
	}{
		{"no type", `[{"text":"orphan"}]`, 2, `no "type"`},
		{"text is not a string", `[{"type":"text","text":42}]`, 0, "missing or not a string"},
		{"image_url extra key", `[{"type":"image_url","image_url":{"url":"` + ProbeImageDataURI + `","detail":"high"}}]`, 0, "detail"},
		{"unknown part", `[{"type":"video","video":{"url":"x"}}]`, 1, "video"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, refusal := read(t, `{"role":"user","content":`+tc.content+`}`, tc.index, limits())
			if refusal == nil || refusal.Code != "unsupported_content_part" {
				t.Fatalf("err = %v", refusal)
			}
			if !strings.Contains(refusal.Message, "messages["+strconv.Itoa(tc.index)+"]") || !strings.Contains(refusal.Message, tc.want) {
				t.Fatalf("message = %q, want index %d and %q", refusal.Message, tc.index, tc.want)
			}
		})
	}
}

func TestContentIsRequired(t *testing.T) {
	_, err := read(t, `{"role":"user"}`, 0, limits())
	if refusal := err; refusal.Code != "messages_not_semif_contract" {
		t.Fatalf("code = %s", refusal.Code)
	}
}

func TestTextPartsAreJoinedWithANewline(t *testing.T) {
	content, err := read(t, `{"role":"user","content":[
		{"type":"text","text":"line one"},
		{"type":"text","text":"line two"}]}`, 0, limits())
	if err != nil {
		t.Fatal(err)
	}
	if content.Text != "line one\nline two" {
		t.Fatalf("text = %q", content.Text)
	}
}
