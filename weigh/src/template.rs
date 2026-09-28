// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Chat-template registry: the wrappers that turn one system + user message into the
//! exact string the model was trained to see.
//!
//! ## Why a registry instead of a Jinja engine
//!
//! The obvious implementation is to read `tokenizer_config.json`'s `chat_template` and
//! render it with `minijinja`. That was already rejected for the gemma-4 template: it is
//! ~17 KB, references ~140 names, calls `.get(` 14 times, and minijinja does not implement
//! that method. Vendoring a template engine to serve the reachable subgraph of a handful
//! of families is not worth the blast radius.
//!
//! The served shape is always the same -- a plain system message plus a plain user message, no
//! tools, no images -- so each family's reachable subgraph is a fixed prefix, an assistant
//! turn opener, and two interpolations. Those are written out here and named.
//!
//! ## What is and is not recorded
//!
//! `prompt_version` (`direct-options-v1`) versions the *content*: the system text and the
//! user JSON. `prompt_template` versions the *rendering*. Only the pair identifies the
//! actual string sent to the server, so both are recorded in the artifact.
//!
//! ## Detection is not a guess
//!
//! `--template auto` reads `{model}/tokenizer_config.json` and matches known markers. A
//! file with a `chat_template` that matches nothing is an **error**, not a silent fall
//! back to gemma-4. An explicit `--template` that contradicts a detected family is also an
//! error, because rendering the wrong wrapper yields a plausible prompt the model was
//! never trained on -- and the boundary check only validates the last token, so it can
//! pass on a wrong wrapper.

use serde_json::Value;

/// The families this crate can render. Names are the CLI values.
pub const NAMES: &[&str] = &[
    "gemma4",
    "gemma3",
    "chatml",
    "chatml-thinking",
    "llama3",
    "llama2",
    "mistral",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChatTemplate {
    /// gemma-4 `<|turn>` format. The historical default, byte-identical to the string
    /// verified against `transformers` on 777 fixture rows.
    Gemma4,
    /// gemma-3 `<start_of_turn>` format.
    Gemma3,
    /// ChatML without a thinking-mode gate: Qwen2/2.5, Yi, and many others.
    Chatml,
    /// ChatML whose template gates the generation prompt on `enable_thinking`: Qwen3/3.5.
    ///
    /// The reference always passes `enable_thinking=False`, and the reachable effect on this
    /// shape is an empty thinking block after the assistant turn opener. Rendering plain
    /// ChatML here would prompt the model into a mode nobody asked for, and the bytes would
    /// not match the `prompt_sha256` a Python run records for the same row.
    ChatmlThinking,
    /// Llama-3.x `<|start_header_id|>` format.
    Llama3,
    /// Llama-2-chat `[INST] <<SYS>>` format.
    Llama2,
    /// Mistral-7B-Instruct v0.1/v0.2 `[INST]` format (system folded into the user turn).
    Mistral,
}

impl ChatTemplate {
    pub fn as_str(self) -> &'static str {
        match self {
            ChatTemplate::Gemma4 => "gemma4",
            ChatTemplate::Gemma3 => "gemma3",
            ChatTemplate::Chatml => "chatml",
            ChatTemplate::ChatmlThinking => "chatml-thinking",
            ChatTemplate::Llama3 => "llama3",
            ChatTemplate::Llama2 => "llama2",
            ChatTemplate::Mistral => "mistral",
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "gemma4" => Ok(ChatTemplate::Gemma4),
            "gemma3" => Ok(ChatTemplate::Gemma3),
            "chatml" => Ok(ChatTemplate::Chatml),
            "chatml-thinking" => Ok(ChatTemplate::ChatmlThinking),
            "llama3" => Ok(ChatTemplate::Llama3),
            "llama2" => Ok(ChatTemplate::Llama2),
            "mistral" => Ok(ChatTemplate::Mistral),
            other => Err(format!(
                "unknown template {:?}; expected auto or one of {}",
                other,
                NAMES.join(", ")
            )),
        }
    }

    /// Which family a `chat_template` string belongs to, by distinctive marker.
    ///
    /// `None` for both "a shape this crate does not know" and "a shape it knows it must not
    /// render"; `unrenderable_reason` tells the two apart for a caller that needs to say why.
    pub fn detect(chat_template: &str) -> Option<Self> {
        if unrenderable_reason(chat_template).is_some() {
            return None;
        }
        // Order matters only in that these markers are mutually exclusive in practice; the
        // gemma-4 marker is checked before gemma-3's because they are different strings.
        if chat_template.contains("<|turn>") {
            return Some(ChatTemplate::Gemma4);
        }
        if chat_template.contains("<start_of_turn>") {
            return Some(ChatTemplate::Gemma3);
        }
        if chat_template.contains("<|im_start|>") {
            // Qwen3/3.5 gate the generation prompt on `enable_thinking` and emit an empty
            // thinking block when it is false; plain ChatML (Qwen2/2.5) does neither.
            return Some(
                if chat_template.contains("enable_thinking") && chat_template.contains("<think>") {
                    ChatTemplate::ChatmlThinking
                } else {
                    ChatTemplate::Chatml
                },
            );
        }
        if chat_template.contains("<|start_header_id|>") {
            return Some(ChatTemplate::Llama3);
        }
        if chat_template.contains("<<SYS>>") {
            return Some(ChatTemplate::Llama2);
        }
        if chat_template.contains("[INST]") {
            // Only the v0.1/v0.2 bytes are rendered here; every other `[INST]` shape was
            // already refused above.
            return Some(ChatTemplate::Mistral);
        }
        None
    }

    /// Render the reachable subgraph for one system turn plus one user turn, ending at the
    /// assistant turn opener. Everything after that point is the single scored token.
    ///
    /// The gemma-4 branch is the historical construction verbatim (its template wraps both
    /// turns in `{{- ... | trim -}}`, hence the trims). The other inputs arrive with no
    /// surrounding whitespace, so their templates are rendered as written.
    pub fn render(self, system: &str, user: &str) -> String {
        let mut prompt = String::with_capacity(system.len() + user.len() + 256);
        match self {
            ChatTemplate::Gemma4 => {
                prompt.push_str("<bos><|turn>system\n");
                prompt.push_str(system.trim());
                prompt.push_str("<turn|>\n<|turn>user\n");
                prompt.push_str(user.trim());
                prompt.push_str("<turn|>\n<|turn>model\n");
            }
            ChatTemplate::Gemma3 => {
                prompt.push_str("<bos><start_of_turn>user\n");
                prompt.push_str(system);
                prompt.push_str("\n\n");
                prompt.push_str(user);
                prompt.push_str("<end_of_turn>\n<start_of_turn>model\n");
            }
            ChatTemplate::Chatml => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(system);
                prompt.push_str("<|im_end|>\n<|im_start|>user\n");
                prompt.push_str(user);
                prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
            }
            ChatTemplate::ChatmlThinking => {
                // The Chatml bytes plus the empty thinking block. Verified against
                // `Qwen/Qwen3-8B`'s own template, whose `add_generation_prompt` branch is
                // `{{- '<|im_start|>assistant\n' }}` followed by, when `enable_thinking` is
                // false, `{{- '<think>\n\n</think>\n\n' }}`.
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(system);
                prompt.push_str("<|im_end|>\n<|im_start|>user\n");
                prompt.push_str(user);
                prompt.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            }
            ChatTemplate::Llama3 => {
                prompt.push_str("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
                prompt.push_str(system);
                prompt.push_str("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n");
                prompt.push_str(user);
                prompt.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatTemplate::Llama2 => {
                prompt.push_str("<s>[INST] <<SYS>>\n");
                prompt.push_str(system);
                prompt.push_str("\n<</SYS>>\n\n");
                prompt.push_str(user);
                prompt.push_str(" [/INST]");
            }
            ChatTemplate::Mistral => {
                // Mistral-7B-Instruct v0.1/v0.2: after the `<s>` bos token the tag is
                // `' [INST] '`, so there is a space on both sides of `[INST]`.
                prompt.push_str("<s> [INST] ");
                prompt.push_str(system);
                prompt.push_str("\n\n");
                prompt.push_str(user);
                prompt.push_str(" [/INST]");
            }
        }
        prompt
    }
}

/// Why a `chat_template` this crate *recognises* is still not rendered.
///
/// Distinct from an unknown shape, and consulted by both [`ChatTemplate::detect`] (which
/// refuses either way) and [`resolve`] -- where it must also refuse the shape when the
/// operator named a family explicitly, since that family's bytes are exactly what is wrong.
fn unrenderable_reason(chat_template: &str) -> Option<&'static str> {
    // A template that raises for the system role cannot be served faithfully here -- the
    // service always sends one -- and gemma-2 does exactly that while sharing gemma-3's
    // `<start_of_turn>` marker. Refusing keeps `auto` from folding the system turn into the
    // user turn on a model whose own template refuses to.
    if chat_template.contains("System role not supported") {
        return Some("it raises for the system role (gemma-2), and this service always sends one");
    }
    // Mistral v0.1/v0.2 write the tag as `' [INST] '`/`' [/INST]'` -- a space on both sides;
    // v0.3 writes `"[INST] "`/`"[/INST]"`. Only the v0.1/v0.2 bytes are rendered here, so a
    // v0.3-shaped template is refused rather than rendered with the wrong spacing.
    if chat_template.contains("[INST]")
        && !(chat_template.contains(" [INST]") && chat_template.contains(" [/INST]"))
    {
        return Some("its [INST] spacing is Mistral v0.3's, and only v0.1/v0.2 bytes are rendered");
    }
    None
}

/// What the operator asked for on the command line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemplateChoice {
    Auto,
    Fixed(ChatTemplate),
}

impl TemplateChoice {
    pub fn parse(text: &str) -> Result<Self, String> {
        if text == "auto" {
            return Ok(TemplateChoice::Auto);
        }
        Ok(TemplateChoice::Fixed(ChatTemplate::parse(text)?))
    }
}

/// The template actually used, plus how it was decided. Both go into the artifact: a
/// defaulted gemma-4 and a detected gemma-4 are the same rendering but different evidence.
#[derive(Debug)]
pub struct Resolved {
    pub template: ChatTemplate,
    pub source: String,
}

/// Read `{model_dir}/tokenizer_config.json` and decide the template.
pub fn resolve(choice: TemplateChoice, model_dir: &str) -> Result<Resolved, String> {
    let path = std::path::Path::new(model_dir).join("tokenizer_config.json");
    let path_text = path.display().to_string();
    let declared: Option<Vec<String>> = match std::fs::read_to_string(&path) {
        Ok(text) => {
            let value: Value = serde_json::from_str(&text)
                .map_err(|error| format!("{} is not valid JSON: {}", path_text, error))?;
            let templates = extract_chat_templates(&value);
            if templates.is_empty() {
                // A *present* `chat_template` this build cannot read is not the same as an
                // absent one: falling back to the default wrapper would score under a
                // template the checkpoint never declared.
                if value.get("chat_template").is_some() {
                    return Err(format!(
                        "{} declares a `chat_template` that is neither a string nor an array of \
                         strings/objects with a `template` key; refusing to fall back to the \
                         default wrapper.",
                        path_text
                    ));
                }
                None
            } else {
                Some(templates)
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("Cannot read {}: {}", path_text, error)),
    };

    let detected = declared
        .as_ref()
        .and_then(|templates| templates.iter().find_map(|text| ChatTemplate::detect(text)));

    match choice {
        TemplateChoice::Fixed(template) => {
            if let Some(other) = detected {
                if other != template {
                    return Err(format!(
                        "--template {} was requested, but {} describes a {} chat template. \
                         Rendering the wrong wrapper produces a plausible prompt the model was \
                         not trained on, and the answer-boundary check only validates the last \
                         token, so it can pass anyway. Pass --template {} or fix --model.",
                        template.as_str(),
                        path_text,
                        other.as_str(),
                        other.as_str()
                    ));
                }
            } else if let Some(reason) = declared
                .as_ref()
                .and_then(|templates| templates.iter().find_map(|text| unrenderable_reason(text)))
            {
                // The operator named a family, but this file declares a shape that build
                // deliberately does not render -- and that shape shares the named family's
                // marker, so honouring the name would render exactly the bytes the refusal
                // exists to prevent.
                return Err(format!(
                    "--template {} was requested, but {} declares a chat_template this build \
                     will not render: {}. Point --model at a checkpoint whose template can be \
                     reproduced byte for byte.",
                    template.as_str(),
                    path_text,
                    reason
                ));
            }
            Ok(Resolved {
                template,
                source: "explicit".to_string(),
            })
        }
        TemplateChoice::Auto => match detected {
            Some(template) => Ok(Resolved {
                template,
                source: format!("detected:{}", path_text),
            }),
            None => {
                if let Some(templates) = declared {
                    let head: String = templates[0].chars().take(200).collect();
                    // Say *why* a recognised shape was refused, not just that it matched
                    // nothing: the fix differs (a different checkpoint vs a different flag).
                    let reason = templates
                        .iter()
                        .find_map(|text| unrenderable_reason(text))
                        .map(|reason| format!(": {}", reason))
                        .unwrap_or_default();
                    return Err(format!(
                        "{} declares a chat_template that matches no supported family ({}){}. \
                         Refusing to guess rather than render a wrapper the model was not trained \
                         on. Template head: {:?}",
                        path_text,
                        NAMES.join(", "),
                        reason,
                        head
                    ));
                }
                // No tokenizer_config.json, or one without a chat_template: keep the
                // historical default so existing gemma-4 runs stay byte-identical, and say
                // so in the source rather than implying it was detected.
                Ok(Resolved {
                    template: ChatTemplate::Gemma4,
                    source: "default:no-chat-template".to_string(),
                })
            }
        },
    }
}

fn extract_chat_templates(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    match value.get("chat_template") {
        Some(Value::String(text)) => out.push(text.clone()),
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(text) => out.push(text.clone()),
                    Value::Object(map) => {
                        if let Some(Value::String(text)) = map.get("template") {
                            out.push(text.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SYSTEM: &str = "Apply the criterion.";
    const USER: &str = r#"{"evidence": "x", "criterion": "y", "options": []}"#;

    #[test]
    fn gemma4_rendering_is_the_historical_construction() {
        let expected = format!(
            "<bos><|turn>system\n{SYSTEM}<turn|>\n<|turn>user\n{USER}<turn|>\n<|turn>model\n"
        );
        assert_eq!(ChatTemplate::Gemma4.render(SYSTEM, USER), expected);
    }

    #[test]
    fn every_family_opens_an_assistant_turn_after_the_user_turn() {
        let cases = [
            (ChatTemplate::Gemma4, "<|turn>model\n"),
            (ChatTemplate::Gemma3, "<start_of_turn>model\n"),
            (ChatTemplate::Chatml, "<|im_start|>assistant\n"),
            (ChatTemplate::ChatmlThinking, "<think>\n\n</think>\n\n"),
            (
                ChatTemplate::Llama3,
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
            ),
            (ChatTemplate::Llama2, " [/INST]"),
            (ChatTemplate::Mistral, " [/INST]"),
        ];
        for (template, suffix) in cases {
            let prompt = template.render(SYSTEM, USER);
            assert!(
                prompt.ends_with(suffix),
                "{}: {:?}",
                template.as_str(),
                prompt
            );
            assert!(
                prompt.contains(SYSTEM),
                "{} lost the system turn",
                template.as_str()
            );
            assert!(
                prompt.contains(USER),
                "{} lost the user turn",
                template.as_str()
            );
            assert!(
                prompt.find(SYSTEM).unwrap() < prompt.find(USER).unwrap(),
                "{} put the user turn first",
                template.as_str()
            );
        }
    }

    #[test]
    fn detection_is_marker_based_and_total_over_the_supported_families() {
        let cases = [
            ("{{ '<|turn>system' }}", ChatTemplate::Gemma4),
            ("{{ '<start_of_turn>user' }}", ChatTemplate::Gemma3),
            ("{{ '<|im_start|>system' }}", ChatTemplate::Chatml),
            (
                "{{ '<|im_start|>assistant\\n' }}{% if enable_thinking is false %}{{ \
                 '<think>\\n\\n</think>\\n\\n' }}{% endif %}",
                ChatTemplate::ChatmlThinking,
            ),
            (
                "{{ '<|start_header_id|>user<|end_header_id|>' }}",
                ChatTemplate::Llama3,
            ),
            ("{{ '<<SYS>>' }}", ChatTemplate::Llama2),
            (
                "{{ ' [INST] ' + system + ' [/INST]' }}",
                ChatTemplate::Mistral,
            ),
        ];
        for (template, expected) in cases {
            assert_eq!(
                ChatTemplate::detect(template),
                Some(expected),
                "{}",
                template
            );
        }
        assert_eq!(ChatTemplate::detect("{{ messages[0]['content'] }}"), None);
    }

    /// The exact bytes the real templates produce, for the families that were rendered
    /// wrongly before: both were verified against the models' own `tokenizer_config.json`.
    #[test]
    fn the_fixed_families_render_what_their_real_templates_produce() {
        // Qwen/Qwen3-8B, `add_generation_prompt` with `enable_thinking=False` (what
        // `direct.py` always passes).
        assert_eq!(
            ChatTemplate::ChatmlThinking.render("S", "U"),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        // Plain ChatML (Qwen2/2.5) must not gain that block.
        assert_eq!(
            ChatTemplate::Chatml.render("S", "U"),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        // mistralai/Mistral-7B-Instruct-v0.2: `' [INST] ' + system + '\n\n' + content +
        // ' [/INST]'`, i.e. a space after the `<s>` bos token.
        assert_eq!(
            ChatTemplate::Mistral.render("S", "U"),
            "<s> [INST] S\n\nU [/INST]"
        );
    }

    #[test]
    fn names_round_trip_through_parse() {
        for name in NAMES {
            let template = ChatTemplate::parse(name).unwrap();
            assert_eq!(template.as_str(), *name);
        }
        assert!(ChatTemplate::parse("gpt").is_err());
        assert_eq!(TemplateChoice::parse("auto").unwrap(), TemplateChoice::Auto);
        assert!(TemplateChoice::parse("nope").is_err());
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let directory =
            std::env::temp_dir().join(format!("semif-template-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// One table for the whole decision: what the file declares, what the operator asked for,
    /// and which of the two wins. Every row has the same shape -- a temp dir, at most one
    /// `tokenizer_config.json`, one outcome -- so the rows are data, not five copies of a test.
    #[test]
    fn resolve_matches_the_declared_file_against_the_operator_choice() {
        // The two shapes this crate recognises and still refuses: they share a family's marker
        // and render different bytes.
        let gemma2 =
            "{{ '<start_of_turn>user' }}{{ raise_exception('System role not supported') }}";
        let mistral_v3 = "{{ \"[INST] \" + content + \"[/INST]\" }}";
        let cases = [
            (
                "nothing declared",
                None,
                TemplateChoice::Auto,
                Ok((ChatTemplate::Gemma4, "default:no-chat-template")),
            ),
            // The array encoding some checkpoints ship, and the family it declares.
            (
                "declared, array encoding",
                Some(
                    json!({"chat_template": [{"name": "default", "template": "{{ '<|im_start|>system' }}"}]}),
                ),
                TemplateChoice::Auto,
                Ok((ChatTemplate::Chatml, "detected:")),
            ),
            // An explicit choice that agrees with the file is kept, and says it was explicit.
            (
                "explicit agreement",
                Some(json!({"chat_template": "{{ ' [INST] ' + x + ' [/INST]' }}"})),
                TemplateChoice::Fixed(ChatTemplate::Mistral),
                Ok((ChatTemplate::Mistral, "explicit")),
            ),
            (
                "nothing matches",
                Some(json!({"chat_template": "{{ weird }}"})),
                TemplateChoice::Auto,
                Err("matches no supported family"),
            ),
            (
                "present but unreadable",
                Some(json!({"chat_template": [{"name": "default"}]})),
                TemplateChoice::Auto,
                Err("neither a string nor an array"),
            ),
            (
                "explicit contradicts the file",
                Some(json!({"chat_template": "{{ '<|im_start|>system' }}"})),
                TemplateChoice::Fixed(ChatTemplate::Gemma4),
                Err("describes a chatml chat template"),
            ),
            (
                "gemma-2 refuses the system role",
                Some(json!({"chat_template": gemma2})),
                TemplateChoice::Auto,
                Err("system role"),
            ),
            // Naming the family it shares a marker with must not force it through.
            (
                "gemma-2, named anyway",
                Some(json!({"chat_template": gemma2})),
                TemplateChoice::Fixed(ChatTemplate::Gemma3),
                Err("system role"),
            ),
            (
                "mistral v0.3 spacing, named as v0.1/v0.2",
                Some(json!({"chat_template": mistral_v3})),
                TemplateChoice::Fixed(ChatTemplate::Mistral),
                Err("spacing"),
            ),
        ];
        for (index, (name, config, choice, expected)) in cases.iter().enumerate() {
            let directory = temp_dir(&format!("resolve-{}", index));
            if let Some(config) = config {
                std::fs::write(directory.join("tokenizer_config.json"), config.to_string())
                    .unwrap();
            }
            match (resolve(*choice, directory.to_str().unwrap()), expected) {
                (Ok(resolved), Ok((template, source))) => {
                    assert_eq!(resolved.template, *template, "{}", name);
                    assert!(
                        resolved.source.starts_with(source),
                        "{}: {}",
                        name,
                        resolved.source
                    );
                }
                (Err(error), Err(fragment)) => {
                    assert!(error.contains(fragment), "{}: {}", name, error)
                }
                (resolved, expected) => panic!(
                    "{}: resolved to {:?}, expected {:?}",
                    name,
                    resolved.map(|resolved| resolved.template),
                    expected
                ),
            }
        }
    }
}
