//! LLM-based spell checking via the Google AI Studio (Gemini) API.
//!
//! This is the app's only online feature, and it is strictly opt-in: the
//! frontend sends the user's API key + chosen flash model along with the
//! recognized page texts, and this command asks Gemini to proofread the OCR
//! output. The model is instructed to return **word-level pairs only**
//! (`wrong` → `correct`) per page — never rewritten text — so the frontend
//! can list the pairs with checkboxes and apply just the ones the user
//! accepts as a display-time projection (same non-destructive shape as the
//! offline Burmese spell-fix).
//!
//! Why the HTTP call lives here and not in the WebView: reqwest (rustls) is
//! already a dependency for language downloads, the frontend has no
//! `tauri-plugin-http` capability, and response parsing + hallucination
//! filtering benefit from serde. The key is passed per-call from the
//! frontend (stored in localStorage) and never persisted backend-side.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::json;

/// Endpoint base. v1beta carries the current flash models and structured
/// output; the model id is interpolated into the path (validated to contain
/// no `/` or whitespace below so it can't be abused as a path escape).
const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Model used by `llm_test_key`. A small, cheap model — the test only needs
/// to prove the key authenticates, so it shouldn't burn flash quota or
/// latency. Named here (not a parameter) so the test call stays stable even
/// as the spell-check model list evolves.
const TEST_MODEL: &str = "gemma-4-31b-it";

/// Whole-request timeout. A 30-page batch can take a while on flash models;
/// 120s leaves headroom for slow generations without hanging forever on a
/// stalled connection.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// One suggested word correction for a page. `wrong` is the exact substring
/// as it appears in the OCR text; `correct` is the model's proposed fix.
/// `line`, when present, is the 1-based line within the page the model says
/// the word sits on — fixes apply (and are validated) per line, so a short
/// Burmese substring can't ripple into other paragraphs of the same page.
/// `None` (model omitted it) falls back to page-wide matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmWordFix {
    #[serde(default)]
    pub line: Option<u32>,
    pub wrong: String,
    pub correct: String,
}

/// All suggested corrections for one page. `page` is 1-based, indexing into
/// the `pages` vec the frontend sent for this request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmPageFix {
    pub page: u32,
    pub fixes: Vec<LlmWordFix>,
}

/// One page's fully rewritten text (direct-fix mode). `lines` is the model's
/// corrected copy of the page, one entry per input line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmPageText {
    pub page: u32,
    pub lines: Vec<String>,
}

/// Proofread a batch of OCR'd page texts with Gemini and return the
/// wrong→correct word pairs the model proposes, per page.
///
/// The frontend drives batching (≤30 pages per call) so it can show
/// per-batch progress and stop between batches; this command handles exactly
/// one request. Async without `spawn_blocking` — the work is IO-bound, same
/// shape as `languages::download_language`. Every returned pair is filtered
/// against the page text (see `filter_fixes`) so hallucinated words never
/// reach the review UI.
#[tauri::command]
pub async fn llm_spell_check(
    api_key: String,
    model: String,
    pages: Vec<String>,
) -> Result<Vec<LlmPageFix>, String> {
    let t = Instant::now();
    let api_key = api_key.trim().to_string();
    let model = model.trim().to_string();
    if api_key.is_empty() {
        return Err("No API key configured — add your Google AI Studio key in Settings.".into());
    }
    if model.is_empty() {
        return Err("No AI model selected — pick one in Settings.".into());
    }
    if model.contains('/') || model.chars().any(char::is_whitespace) {
        return Err(format!("Invalid model id \"{model}\"."));
    }
    if pages.is_empty() {
        return Ok(Vec::new());
    }

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    // reqwest is built without the `json` feature (see Cargo.toml's minimal
    // feature set), so serialize manually and set the header by hand.
    let payload = serde_json::to_string(&build_request_body(&pages))
        .map_err(|e| format!("Failed to serialize request: {e}"))?;
    let body = send_generate(&client, &api_key, &model, payload).await?;

    let fixes = parse_response(&body, &pages)?;
    let total: usize = fixes.iter().map(|p| p.fixes.len()).sum();
    log::info!(
        "[llm] spell-check {} page{} on {model}: {:.0} ms, {} fix{}",
        pages.len(),
        if pages.len() == 1 { "" } else { "s" },
        t.elapsed().as_secs_f64() * 1000.0,
        total,
        if total == 1 { "" } else { "es" },
    );
    Ok(fixes)
}

/// POST a pre-serialized generateContent request and return the raw response
/// body on success. Shared by `llm_spell_check` and `llm_test_key`; maps
/// transport failures and non-2xx responses (with Gemini's `error.message`,
/// which covers invalid key / quota / bad model id) to user-facing strings.
async fn send_generate(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    payload: String,
) -> Result<String, String> {
    let url = format!("{API_BASE}/{model}:generateContent");
    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("Content-Type", "application/json")
        .body(payload)
        .send()
        .await
        .map_err(|e| format!("Gemini request failed: {e}"))?;

    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read Gemini response: {e}"))?;
    if !status.is_success() {
        let detail = extract_quota_error(&body)
            .or_else(|| extract_error_message(&body))
            .unwrap_or_else(|| truncate(&body, 200));
        return Err(format!("Gemini API error (HTTP {status}): {detail}"));
    }
    Ok(body)
}

/// Verify a Google AI Studio API key by making a minimal generateContent call
/// with the cheap [TEST_MODEL]. Driven by the "Test" button next to the key
/// field in Settings. Returns `Ok(())` when the key authenticates — the reply
/// content itself is irrelevant; an auth/quota problem surfaces as Gemini's
/// own error message.
#[tauri::command]
pub async fn llm_test_key(api_key: String) -> Result<(), String> {
    let t = Instant::now();
    let api_key = api_key.trim().to_string();
    if api_key.is_empty() {
        return Err("Enter an API key first.".into());
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
    let payload = serde_json::to_string(&json!({
        "contents": [{ "parts": [{ "text": "Reply with the single word: OK" }] }],
    }))
    .map_err(|e| format!("Failed to serialize request: {e}"))?;
    send_generate(&client, &api_key, TEST_MODEL, payload).await?;
    log::info!("[llm] key test on {TEST_MODEL}: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// Direct-fix counterpart of [`llm_spell_check`]: instead of wrong→correct
/// word pairs, the model returns each page's corrected text as an array of
/// lines. The frontend diffs the lines against the originals and still shows
/// changed lines for review — so this mode gains coverage (punctuation,
/// spacing, phrasing the word-pair mode can't express) at the cost of more
/// output tokens. Same batching + key handling as the word-pair path.
#[tauri::command]
pub async fn llm_rewrite_pages(
    api_key: String,
    model: String,
    pages: Vec<String>,
) -> Result<Vec<LlmPageText>, String> {
    let t = Instant::now();
    let api_key = api_key.trim().to_string();
    let model = model.trim().to_string();
    if api_key.is_empty() {
        return Err("No API key configured — add your Google AI Studio key in Settings.".into());
    }
    if model.is_empty() {
        return Err("No AI model selected — pick one in Settings.".into());
    }
    if model.contains('/') || model.chars().any(char::is_whitespace) {
        return Err(format!("Invalid model id \"{model}\"."));
    }
    if pages.is_empty() {
        return Ok(Vec::new());
    }

    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
    let payload = serde_json::to_string(&build_rewrite_request_body(&pages))
        .map_err(|e| format!("Failed to serialize request: {e}"))?;
    let body = send_generate(&client, &api_key, &model, payload).await?;

    let out = parse_rewrite_response(&body, &pages)?;
    let changed: usize = out.iter().map(|p| p.lines.len()).sum();
    log::info!(
        "[llm] rewrite {} page{} on {model}: {:.0} ms, {} line{} returned",
        pages.len(),
        if pages.len() == 1 { "" } else { "s" },
        t.elapsed().as_secs_f64() * 1000.0,
        changed,
        if changed == 1 { "" } else { "s" },
    );
    Ok(out)
}

/// Rewrite-mode request: the model must return the corrected text with the
/// SAME line structure — no merging/splitting/reordering — so the frontend
/// can diff line-by-line and let the user accept changes per line.
fn build_rewrite_request_body(pages: &[String]) -> serde_json::Value {
    let system = "You are an OCR proofreading assistant. The user gives you numbered pages of \
OCR output; every line of a page is prefixed with its 1-based line number, like \"12: text\". \
For each page, return the corrected text as an array of lines. Fix misspelled or misrecognized \
words, broken punctuation, and obvious OCR artifacts; keep the same language and script. \
STRICTLY keep the line structure: do not merge, split, reorder, add, or drop lines — the \
returned array must have exactly the same number of lines as the input page. Do not alter \
proper names, numbers, or already-correct words. Return ONLY a JSON array where each element \
is {\"page\": <page number>, \"lines\": [\"corrected line 1\", \"corrected line 2\", ...]}. \
No markdown fences, no commentary.";

    let mut prompt = String::from("Proofread and correct the following OCR output.\n");
    for (i, text) in pages.iter().enumerate() {
        prompt.push_str(&format!("\n### Page {}\n", i + 1));
        for (l, line) in text.split('\n').enumerate() {
            prompt.push_str(&format!("{}: {}\n", l + 1, line));
        }
    }

    json!({
        "systemInstruction": { "parts": [{ "text": system }] },
        "contents": [{ "parts": [{ "text": prompt }] }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": { "type": "integer" },
                        "lines": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["page", "lines"]
                }
            }
        }
    })
}

/// Parse a rewrite reply into per-page line arrays, dropping out-of-range
/// pages. Line-count mismatches against the original are NOT dropped here —
/// the frontend diffs per index and plainTextWithFix falls back to raw text
/// for missing trailing lines, so a mismatch degrades gracefully instead of
/// losing the page.
fn parse_rewrite_response(body: &str, pages: &[String]) -> Result<Vec<LlmPageText>, String> {
    let cleaned = extract_model_text(body)?;
    let parsed: Vec<LlmPageText> = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Could not parse Gemini's rewritten pages: {e}"))?;
    let mut out: Vec<LlmPageText> = parsed
        .into_iter()
        .filter(|p| p.page >= 1 && pages.get((p.page - 1) as usize).is_some())
        .collect();
    for page in &mut out {
        strip_echoed_line_numbers(&mut page.lines);
    }
    out.sort_by_key(|p| p.page);
    Ok(out)
}

/// If `line` starts with an input-style `N: ` number prefix (1–4 digits,
/// colon, optional single space), return the content after it.
fn line_no_prefix(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    let digits = i;
    if digits == 0 || digits > 4 || i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1; // colon
    if i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    // Safe cut: digits/colon/space are single-byte, so i is a char boundary.
    Some(&line[i..])
}

/// Models sometimes echo the prompt's `N: ` line-number prefixes in their
/// rewritten lines, which would make EVERY line "change" (identical lines
/// diff as a lone "1:" fragment). When ALL non-empty lines of a page carry
/// such a prefix, treat it as an echo and strip them; pages where only some
/// lines look numbered are left alone (could be genuine numbered content).
fn strip_echoed_line_numbers(lines: &mut [String]) {
    let all_numbered = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .all(|l| line_no_prefix(l).is_some());
    if !all_numbered {
        return;
    }
    for line in lines.iter_mut() {
        if let Some(rest) = line_no_prefix(line) {
            *line = rest.to_string();
        }
    }
}

/// Build the generateContent request body: a strict proofreading system
/// instruction, the numbered pages (each line prefixed with its 1-based
/// number) as user content, and a response schema pinning the reply to
/// `[{page, fixes:[{line, wrong, correct}]}]`. Temperature 0 keeps the model
/// from inventing variations; structured output means we never have to
/// scrape prose for the pairs. Line addressing exists so a fix for a short
/// substring (easy in unspaced Burmese) applies only to the line it was
/// flagged on — not to every paragraph of the page.
fn build_request_body(pages: &[String]) -> serde_json::Value {
    let system = "You are an OCR proofreading assistant. The user gives you numbered pages of \
OCR output; every line of a page is prefixed with its 1-based line number, like \"12: text\". \
For each page, identify words that are misspelled or misrecognized by OCR. Return ONLY a JSON \
array where each element is \
{\"page\": <page number>, \"fixes\": [{\"line\": <line number>, \"wrong\": \"...\", \"correct\": \"...\"}]}. \
Rules: \"line\" is the number of the line the wrong word appears on; copy \"wrong\" EXACTLY as \
it appears on that line; propose the most likely correct word in the same language and script; \
do not correct proper names, numbers, abbreviations, or unusual-but-correct words; keep changes \
minimal; the same misspelling on different lines needs one fix entry per line; omit pages with \
no errors; return an empty array if nothing is wrong. No markdown fences, no commentary.";

    let mut prompt = String::from("Proofread the following OCR output.\n");
    for (i, text) in pages.iter().enumerate() {
        // Delimiter-style page markers keep long batches unambiguous even
        // when a page's text itself contains lines that look like markers.
        prompt.push_str(&format!("\n### Page {}\n", i + 1));
        for (l, line) in text.split('\n').enumerate() {
            prompt.push_str(&format!("{}: {}\n", l + 1, line));
        }
    }

    json!({
        "systemInstruction": { "parts": [{ "text": system }] },
        "contents": [{ "parts": [{ "text": prompt }] }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": { "type": "integer" },
                        "fixes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "line": { "type": "integer" },
                                    "wrong": { "type": "string" },
                                    "correct": { "type": "string" }
                                },
                                "required": ["wrong", "correct"]
                            }
                        }
                    },
                    "required": ["page", "fixes"]
                }
            }
        }
    })
}

/// Pull the model's text out of a generateContent HTTP response body:
/// concatenates the candidate's text parts and strips a defensive markdown
/// fence (structured output is bare JSON, but a chatty model could wrap it).
/// Shared by the word-pair and rewrite paths.
fn extract_model_text(body: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| format!("Gemini returned invalid JSON: {e}"))?;
    let parts = value
        .pointer("/candidates/0/content/parts")
        .and_then(|p| p.as_array())
        .ok_or_else(|| {
            // Blocked/empty generations land here (no candidates, or a
            // candidate with no parts). The finishReason usually says why.
            let reason = value
                .pointer("/candidates/0/finishReason")
                .and_then(|r| r.as_str())
                .unwrap_or("no content returned");
            format!("Gemini returned no content ({reason}).")
        })?;
    let text: String = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect();
    if text.trim().is_empty() {
        return Err("Gemini returned an empty response.".into());
    }
    let cleaned = text.trim().trim_start_matches("```json").trim_start_matches("```");
    Ok(cleaned.trim_end_matches("```").trim().to_string())
}

/// Pull the model's text out of a generateContent HTTP response body, then
/// parse + filter it into per-page fixes.
fn parse_response(body: &str, pages: &[String]) -> Result<Vec<LlmPageFix>, String> {
    let cleaned = extract_model_text(body)?;
    let parsed: Vec<LlmPageFix> = serde_json::from_str(&cleaned)
        .map_err(|e| format!("Could not parse Gemini's corrections: {e}"))?;
    Ok(filter_fixes(parsed, pages))
}

/// Drop suggestions that can't be safely applied: out-of-range pages, empty
/// or identical pairs, duplicates, and — most importantly — pairs whose
/// `wrong` doesn't occur where the model says it does. With a `line`, the
/// word must occur on exactly that line of that page (this is the guard
/// that keeps a short Burmese substring fix from leaking into other
/// paragraphs of the same page); without one, anywhere on the page
/// suffices. Case-sensitive substring match: the pairs are meant to be
/// copied verbatim, and a case-insensitive match could silently alter a
/// different word.
fn filter_fixes(fixes: Vec<LlmPageFix>, pages: &[String]) -> Vec<LlmPageFix> {
    let mut out: Vec<LlmPageFix> = Vec::new();
    for mut page_fix in fixes {
        let Some(text) = page_fix
            .page
            .checked_sub(1)
            .and_then(|i| pages.get(i as usize))
        else {
            continue;
        };
        let lines: Vec<&str> = text.split('\n').collect();
        page_fix.fixes.retain(|f| {
            if f.wrong.is_empty() || f.wrong == f.correct {
                return false;
            }
            match f.line {
                Some(n) => lines
                    .get(n.checked_sub(1).map_or(usize::MAX, |i| i as usize))
                    .is_some_and(|l| l.contains(f.wrong.as_str())),
                None => text.contains(f.wrong.as_str()),
            }
        });
        // Compare line too: the same wrong word on two lines is two distinct
        // fixes, not a duplicate.
        page_fix
            .fixes
            .dedup_by(|a, b| a.wrong == b.wrong && a.correct == b.correct && a.line == b.line);
        if !page_fix.fixes.is_empty() {
            out.push(page_fix);
        }
    }
    out.sort_by_key(|p| p.page);
    out
}

/// Extract `error.message` from a Gemini error body, if present.
fn extract_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .pointer("/error/message")?
        .as_str()
        .map(str::to_string)
}

/// Turn a quota/rate-limit error body (HTTP 429 / RESOURCE_EXHAUSTED) into a
/// user-facing message. Google AI Studio's free tier caps requests per day
/// per model; when the daily cap is hit the error carries QuotaFailure
/// details (quota id like `GenerateRequestsPerDayPerProjectPerModel` + the
/// limit value) and often a RetryInfo delay. Daily caps only reset at
/// midnight US Pacific, so the actionable advice is "try again tomorrow" —
/// not a retry in seconds. Returns None for non-quota errors (handled by
/// [`extract_error_message`] instead).
fn extract_quota_error(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let err = value.get("error")?;
    let is_quota = err.get("status").and_then(|s| s.as_str()) == Some("RESOURCE_EXHAUSTED")
        || err.get("code").and_then(|c| c.as_i64()) == Some(429);
    if !is_quota {
        return None;
    }

    let mut quota_id: Option<String> = None;
    let mut quota_limit: Option<String> = None;
    let mut retry_delay: Option<String> = None;
    if let Some(details) = err.get("details").and_then(|d| d.as_array()) {
        for detail in details {
            let typ = detail.get("@type").and_then(|t| t.as_str()).unwrap_or("");
            if typ.ends_with("QuotaFailure") {
                if let Some(violations) = detail.get("violations").and_then(|v| v.as_array()) {
                    for viol in violations {
                        quota_id = quota_id
                            .or_else(|| viol.get("quotaId").and_then(|q| q.as_str()).map(str::to_string));
                        quota_limit = quota_limit
                            .or_else(|| viol.get("quotaValue").and_then(|q| q.as_str()).map(str::to_string));
                    }
                }
            } else if typ.ends_with("RetryInfo") {
                retry_delay = retry_delay
                    .or_else(|| detail.get("retryDelay").and_then(|r| r.as_str()).map(str::to_string));
            }
        }
    }

    let limit_note = quota_limit
        .map(|l| format!(" (limit: {l} requests/day)"))
        .unwrap_or_default();
    let quota_id = quota_id.unwrap_or_default();
    if quota_id.contains("PerDay") || quota_id.contains("Daily") {
        return Some(format!(
            "Google AI Studio free-tier daily limit reached{limit_note}. \
The quota resets tomorrow (midnight US Pacific) — finish reviewing the pages \
already checked, then run the rest tomorrow."
        ));
    }
    match retry_delay {
        // Per-minute bursts and other short-term quotas: the delay is real.
        Some(delay) => Some(format!(
            "Gemini rate limit hit{limit_note} — retry in {delay}. If this keeps \
happening, the free-tier daily cap may be exhausted (resets tomorrow, midnight \
US Pacific)."
        )),
        None => Some(format!(
            "Gemini quota exceeded{limit_note}. If this is the free-tier daily \
cap, it resets tomorrow (midnight US Pacific)."
        )),
    }
}

/// Truncate a string for inclusion in an error message, cutting on a char
/// boundary (pages of OCR text can contain multi-byte scripts).
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(wrong: &str, correct: &str) -> LlmWordFix {
        LlmWordFix { line: None, wrong: wrong.into(), correct: correct.into() }
    }

    /// A realistic generateContent success body: one candidate, JSON in a
    /// single text part (as structured output produces).
    fn response_body(model_json: &str) -> String {
        let escaped = serde_json::json!(model_json).to_string();
        format!(
            r#"{{"candidates":[{{"content":{{"parts":[{{"text":{escaped}}}]}},"finishReason":"STOP"}}]}}"#
        )
    }

    #[test]
    fn parses_and_filters_model_reply() {
        let pages = vec![
            "recogntion is hard\ngood line here".to_string(), // page 1
            "all good here".to_string(),                      // page 2: nothing wrong
        ];
        let model = r#"[
            {"page":1,"fixes":[
                {"line":1,"wrong":"recogntion","correct":"recognition"},
                {"line":2,"wrong":"recogntion","correct":"recognition"},
                {"line":99,"wrong":"good","correct":"godd"},
                {"wrong":"hard","correct":"difficult"}
            ]},
            {"page":2,"fixes":[{"line":1,"wrong":"godd","correct":"good"}]},
            {"page":9,"fixes":[{"wrong":"x","correct":"y"}]},
            {"page":1,"fixes":[{"line":1,"wrong":"is","correct":"is"},{"line":1,"wrong":"","correct":"z"}]}
        ]"#;
        let out = parse_response(&response_body(model), &pages).unwrap();
        // Page 1 keeps only: the line-addressed fix (word really on line 1);
        // the same word claimed for line 2 is dropped (not there), line 99
        // is out of range, and the line-less "hard" fix survives page-wide
        // validation. Page 2's pair is dropped (word not present), page 9
        // is out of range, and page 1's identical/empty pairs are filtered.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].page, 1);
        assert_eq!(out[0].fixes.len(), 2);
        assert_eq!(out[0].fixes[0].line, Some(1));
        assert_eq!(out[0].fixes[0].wrong, "recogntion");
        assert_eq!(out[0].fixes[1].wrong, "hard");
        assert_eq!(out[0].fixes[1].line, None);
    }

    #[test]
    fn same_word_on_different_lines_gets_separate_fixes() {
        // The point of line addressing: "တို" occurs on both lines, but the
        // model flags only line 2's occurrence — the line-1 word must not
        // be touched, and both entries survive dedup (they differ in line).
        let pages = vec!["တို ပထမ\nတို ဒုတိယ".to_string()];
        let model = r#"[
            {"page":1,"fixes":[
                {"line":2,"wrong":"တို","correct":"တို့"}
            ]}
        ]"#;
        let out = parse_response(&response_body(model), &pages).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fixes.len(), 1);
        assert_eq!(out[0].fixes[0].line, Some(2));
    }

    #[test]
    fn empty_model_array_yields_no_fixes() {
        let out = parse_response(&response_body("[]"), &["text".into()]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn strips_defensive_markdown_fence() {
        let fenced = "```json\n[{\"page\":1,\"fixes\":[{\"wrong\":\"teh\",\"correct\":\"the\"}]}]\n```";
        let pages = vec!["teh cat".to_string()];
        let out = parse_response(&response_body(fenced), &pages).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fixes[0].correct, "the");
    }

    #[test]
    fn no_candidates_is_an_error() {
        let err = parse_response(r#"{"candidates":[]}"#, &["x".into()]).unwrap_err();
        assert!(err.contains("no content returned"), "got: {err}");
    }

    #[test]
    fn error_message_extraction() {
        let body = r#"{"error":{"code":400,"message":"API key not valid"}}"#;
        assert_eq!(extract_error_message(body).as_deref(), Some("API key not valid"));
        assert_eq!(extract_error_message("not json"), None);
    }

    /// A realistic free-tier daily-cap 429: RESOURCE_EXHAUSTED with a
    /// QuotaFailure (per-day quota id + limit) and a RetryInfo delay.
    #[test]
    fn daily_quota_error_reads_limit_and_says_resets_tomorrow() {
        let body = r#"{
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel",
                                "quotaValue": "250"
                            }
                        ]
                    },
                    { "@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s" }
                ]
            }
        }"#;
        let msg = extract_quota_error(body).unwrap();
        assert!(msg.contains("daily limit"), "got: {msg}");
        assert!(msg.contains("limit: 250 requests/day"), "got: {msg}");
        assert!(msg.contains("tomorrow"), "got: {msg}");
        // A daily cap must NOT advise a seconds-scale retry.
        assert!(!msg.contains("retry in 36s"), "got: {msg}");
    }

    /// Short-term (per-minute) quota: keep the API's retry delay.
    #[test]
    fn short_term_quota_error_keeps_retry_delay() {
        let body = r#"{
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [ { "quotaId": "GenerateRequestsPerMinutePerProjectPerModel" } ]
                    },
                    { "@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s" }
                ]
            }
        }"#;
        let msg = extract_quota_error(body).unwrap();
        assert!(msg.contains("retry in 12s"), "got: {msg}");
    }

    /// Non-quota errors are left to the plain message extractor.
    #[test]
    fn quota_extraction_ignores_other_errors() {
        assert!(extract_quota_error(r#"{"error":{"code":400,"message":"bad"}}"#).is_none());
        assert!(extract_quota_error("not json").is_none());
    }

    #[test]
    fn rewrite_response_parses_and_drops_out_of_range_pages() {
        let pages = vec!["အတာ တစ်".to_string(), "second page".to_string()];
        let model = r#"[
            {"page":2,"lines":["second page"]},
            {"page":1,"lines":["အတာ တစ်", " "]},
            {"page":9,"lines":["bogus"]}
        ]"#;
        let out = parse_rewrite_response(&response_body(model), &pages).unwrap();
        // Out-of-range page dropped; in-range pages sorted by page number.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].page, 1);
        assert_eq!(out[0].lines, vec!["အတာ တစ်", " "]);
        assert_eq!(out[1].page, 2);
    }

    #[test]
    fn rewrite_request_pins_line_structure() {
        let body = build_rewrite_request_body(&["a\nb".into()]);
        let prompt = body.pointer("/contents/0/parts/0/text").unwrap().as_str().unwrap();
        assert!(prompt.contains("### Page 1\n1: a\n2: b"));
        let system = body.pointer("/systemInstruction/parts/0/text").unwrap().as_str().unwrap();
        assert!(system.contains("same number of lines"));
        // Schema returns per-page line arrays.
        assert_eq!(
            body.pointer("/generationConfig/responseSchema/items/properties/lines/type")
                .unwrap(),
            "array"
        );
    }

    #[test]
    fn rewrite_strips_echoed_line_numbers() {
        // The model echoed the prompt's "N: " prefixes — every line would
        // otherwise "change" (identical lines diff as a lone prefix).
        let pages = vec!["cat".to_string(), "dog".to_string()];
        let model = r#"[
            {"page":1,"lines":["1: cat","2: dog"]},
            {"page":2,"lines":["1: dogs"]}
        ]"#;
        let out = parse_rewrite_response(&response_body(model), &pages).unwrap();
        assert_eq!(out[0].lines, vec!["cat", "dog"]);
        assert_eq!(out[1].lines, vec!["dogs"]);
    }

    #[test]
    fn rewrite_leaves_partly_numbered_lines_alone() {
        // Only one line looks numbered → could be genuine content; no strip.
        let pages = vec!["1: real list item\nplain line".to_string()];
        let model = r#"[{"page":1,"lines":["1: real list item","plain line"]}]"#;
        let out = parse_rewrite_response(&response_body(model), &pages).unwrap();
        assert_eq!(out[0].lines, vec!["1: real list item", "plain line"]);
    }

    #[test]
    fn request_body_is_wellformed() {
        let body = build_request_body(&["hello\nworld".into(), "solo".into()]);
        // Prompt carries page markers AND per-line numbers.
        let prompt = body.pointer("/contents/0/parts/0/text").unwrap().as_str().unwrap();
        assert!(prompt.contains("### Page 1\n1: hello\n2: world"));
        assert!(prompt.contains("### Page 2\n1: solo"));
        // System instruction + JSON response schema are wired up.
        assert!(body.pointer("/systemInstruction/parts/0/text").is_some());
        assert_eq!(
            body.pointer("/generationConfig/responseMimeType").unwrap(),
            "application/json"
        );
        assert!(body.pointer("/generationConfig/responseSchema").is_some());
    }
}
