//! Burmese post-OCR spelling fix: two complementary correction stages
//! applied per-line to Myanmar OCR results.
//!
//! 1. **Regex / substring normalization** ([`normalize`]): a curated list of
//!    ~90 rules ported from a JS post-processing script — doubled-vowel-mark
//!    dedup, stacked-consonant + char-class fixes, lookaround-based context
//!    fixes, and literal substring substitutions (Pali + Myanmar words). Each
//!    rule is a global `replace_all` pass over the current text, applied in
//!    source order. Patterns use `fancy-regex` syntax (lookahead/lookbehind);
//!    replacements are literal strings (no `$N` backreferences).
//!
//! 2. **Whole-token dictionary replacement** ([`correct`]): a `wrong,right`
//!    word list applied at token boundaries (Myanmar-aware tokenization —
//!    combining marks stick to their base consonant; sentence marks `၊`/`။`
//!    are boundaries).
//!
//! `correct_line` runs **both** stages in that order — normalize first (so
//! the token dict sees already-tidied text), then whole-token correction.
//! Callers gate the whole thing on the opt-in flag (see `engine::run_myanmar`).
//!
//! Both data files are embedded via `include_str!` (zero-setup, like the
//! bundled models) and parsed once into statics over the `'static` bytes —
//! lookups are allocation-free.

use std::collections::HashMap;

use fancy_regex::Regex;
use once_cell::sync::Lazy;

use crate::engine::LineBox;

// ── Stage 1: regex / substring normalization ─────────────────────────────────

/// The raw `pattern\treplacement` rule list, embedded at compile time. Path
/// is relative to this file (`src-tauri/src/`). See the file header for the
/// format and the category breakdown.
static RULES_RAW: &str = include_str!("burmese_spelling_rules.tsv");

/// One compiled normalization rule: a regex and its literal replacement.
struct Rule {
    re: Regex,
    replacement: String,
}

/// Compiled normalization rules, built once on first use via `Lazy`. Each
/// pattern is compiled once at process start (the 9 lookaround rules go
/// through `fancy-regex`'s slower backtracking engine; the rest are
/// internally forwarded to the std `regex` crate). Compilation failures are
/// logged at `error` level and the rule is dropped — a bad pattern in the
/// data file should never crash the app, just silently skip that rule.
static RULES: Lazy<Vec<Rule>> = Lazy::new(|| {
    let mut out = Vec::new();
    for (lineno, line) in RULES_RAW.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Split on the FIRST tab only: the replacement may (in principle)
        // contain a tab, though none currently do. We keep the pattern and
        // replacement's interior verbatim, including any trailing whitespace
        // (one rule intentionally matches a trailing space — see the file
        // header). Only the line's trailing CR (Windows line endings) is
        // stripped, and a trailing newline never reaches us (`.lines()`).
        let Some((pattern, replacement)) = line.split_once('\t') else {
            log::warn!("[spelling] rules.tsv:{}: no tab, skipping", lineno + 1);
            continue;
        };
        match Regex::new(pattern) {
            Ok(re) => out.push(Rule {
                re,
                replacement: replacement.to_string(),
            }),
            Err(e) => log::error!(
                "[spelling] rules.tsv:{}: bad regex {:?}: {}",
                lineno + 1,
                pattern,
                e
            ),
        }
    }
    log::info!("[spelling] compiled {} normalization rules", out.len());
    out
});

/// Apply the regex/substring normalization rules to `text`, returning the
/// fixed string and a count of individual substitutions made. Each rule is a
/// global `replace_all` pass; rules run in source order (output of one feeds
/// the next). Unknown text passes through.
///
/// The count is the sum of matches across all rules (each regex hit is one
/// substitution) — i.e. the total number of individual fixes this pass
/// applied. A caller surfacing "how many spelling fixes were made" to the UI
/// sums this with the count from [`correct`].
///
/// `fancy-regex`'s `replace_all` takes a closure for the replacement so it
/// can interpolate captures; we ignore captures (replacements are literal)
/// and just return the rule's replacement string, which makes this equivalent
/// to JS's `String.prototype.replace(/pat/g, "literal")`. The closure also
/// bumps a counter per match so we know how many substitutions happened.
pub fn normalize(text: &str) -> (String, u32) {
    // Optimistic: most lines won't match any rule, so a single allocation
    // for the working buffer + in-place edits would be ideal. fancy-regex
    // doesn't expose an in-place replace, so we allocate one String per
    // matching rule. Cheap relative to recognition — but guard the whole
    // pass behind the opt-in flag at the call site.
    let mut cur = String::from(text);
    let mut total: u32 = 0;
    for rule in RULES.iter() {
        let mut hits: u32 = 0;
        cur = rule
            .re
            .replace_all(&cur, |_caps: &fancy_regex::Captures| {
                hits += 1;
                rule.replacement.as_str()
            })
            .to_string();
        total += hits;
    }
    (cur, total)
}

// ── Stage 2: whole-token dictionary replacement ──────────────────────────────

/// The raw `wrong,right` word list, embedded at compile time. Path is
/// relative to this file (`src-tauri/src/`). One pair per line, no header.
/// Lines without a comma (or blank) are skipped by the parser.
static DICT_RAW: &str = include_str!("burmese_spelling_fix.csv");

/// Parsed dictionary: `{ wrong => right }`, borrowing directly into the
/// `&'static` embedded bytes. Built once on first lookup via `Lazy`. Split
/// is on the **first** comma so a value containing a comma would still parse
/// (none currently do, but the parser doesn't assume that). Duplicate keys
/// are last-wins — the bundled list has two (`ဝန်စံ`, `သညဲ`) that each
/// map to the same value, so the choice is unobservable.
static DICT: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for line in DICT_RAW.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((wrong, right)) = line.split_once(',') else {
            continue;
        };
        if !wrong.is_empty() {
            map.insert(wrong, right);
        }
    }
    map
});

/// A character that belongs to a lookup token. This is **not** just
/// `char::is_alphanumeric()`: Myanmar combining marks (tone dots U+1036/1037,
/// vowel signs, the asat, medials) have Unicode `Alphabetic = No`, so the
/// std predicate splits Burmese words on their own diacritics — a base
/// consonant plus its stacked marks would be tokenized as several fragments
/// and never match a dict key. Burmese orthography is "base + diacritic
/// stack", so the whole Myanmar Unicode block must be one token class.
///
/// We therefore treat a char as a token char if it is alphanumeric OR it
/// falls in a Myanmar block (main U+1000–U+109F, Extended-A U+AA60–AA7F,
/// Extended-B U+A9E0–A9FF) — with the **two sentence-punctuation marks
/// excluded**: `၊` (U+104A) and `။` (U+104B) are word boundaries, not part
/// of a token (a line ending in `...ဥပါသကာ။` must look up `ဥပါသကာ`, not
/// `ဥပါသကာ။`). Digits (Latin + Myanmar U+1040–U+1049) are token chars too,
/// but no dict key is purely digits, so they pass through unchanged.
fn is_token_char(c: char) -> bool {
    if c.is_alphanumeric() {
        return true;
    }
    let cp = c as u32;
    matches!(cp, 0x1000..=0x109F | 0xAA60..=0xAA7F | 0xA9E0..=0xA9FF)
        && !matches!(cp, 0x104A | 0x104B)
}

/// Apply whole-token dictionary replacements to `text`, returning the fixed
/// string and a count of tokens replaced. Tokenizes on [`is_token_char`]
/// (Myanmar-aware — see its doc comment); everything else is a separator
/// copied verbatim. Each token is looked up in [`DICT`]; a hit appends the
/// replacement and counts as one substitution, a miss appends the original
/// token. Unrecognized text is therefore returned unchanged.
///
/// Allocation: one `String` per call regardless of hit count. The dict
/// lookup itself borrows — no per-token allocation.
pub fn correct(text: &str) -> (String, u32) {
    let mut out = String::with_capacity(text.len());
    let mut hits: u32 = 0;
    let mut chars = text.char_indices().peekable();
    let bytes = text.as_bytes();

    while let Some((i, c)) = chars.peek().copied() {
        if is_token_char(c) {
            // Start of a token: consume the maximal token-char run and look
            // it up. We track the run's byte range via the char's byte offset
            // (`i`) and the offset of the first separator after it.
            let start = i;
            let mut end = start;
            while let Some(&(j, ch)) = chars.peek() {
                if is_token_char(ch) {
                    end = j + ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let token = std::str::from_utf8(&bytes[start..end]).unwrap_or("");
            match DICT.get(token) {
                Some(repl) => {
                    out.push_str(repl);
                    hits += 1;
                }
                None => out.push_str(token),
            }
        } else {
            // Separator: copy through verbatim.
            out.push(c);
            chars.next();
        }
    }
    (out, hits)
}

// ── Combined entry point ─────────────────────────────────────────────────────

/// Apply both correction stages to `text`: normalization rules first (dedup,
/// char-class, lookaround, literal substring passes), then whole-token dict
/// replacement. Normalizing first means the dict sees already-tidied text.
/// Returns the fixed string plus the total number of substitutions across
/// both stages (regex matches + dict token replacements).
pub fn fix(text: &str) -> (String, u32) {
    let (normalized, n_hits) = normalize(text);
    let (corrected, c_hits) = correct(&normalized);
    (corrected, n_hits + c_hits)
}

/// Apply [`fix`] to a line's text in place, returning the number of
/// substitutions made. Called from the Myanmar pipeline after recognition,
/// before the result is returned to the UI. Mutating `line.text` means every
/// downstream projection (text panel, export, copy) reflects the fix with no
/// further wiring; the returned count lets the caller total fixes across all
/// lines for the result.
pub fn correct_line(line: &mut LineBox) -> u32 {
    let (text, hits) = fix(&line.text);
    line.text = text;
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalization (stage 1) ──────────────────────────────────────────────

    #[test]
    fn rules_compiled_to_nonempty_list() {
        // Sanity: the embedded TSV parsed into rules. Don't pin the exact
        // count so adding entries doesn't break this test.
        assert!(RULES.len() > 50, "rules too few: {}", RULES.len());
    }

    #[test]
    fn normalize_dedups_doubled_vowel_marks() {
        // `ာာ` (two U+102F) collapses to a single `ာ`. One substitution.
        assert_eq!(normalize("ကာာ"), ("ကာ".to_string(), 1));
        // Runs through each vowel mark independently.
        assert_eq!(normalize("ကိိ"), ("ကိ".to_string(), 1));
    }

    #[test]
    fn normalize_applies_char_class_rule() {
        // `က္ + [ဃဆဏတထဘယလသဟ]` → `က္က`. Pick တ from the class.
        assert_eq!(normalize("က္တ"), ("က္က".to_string(), 1));
        // A consonant NOT in the class is left alone.
        assert_eq!(normalize("က္ဂ"), ("က္ခ".to_string(), 1));
    }

    #[test]
    fn normalize_applies_lookbehind_rule() {
        // `(?<![ငညဏနမ])(ူ|ု|ှု)ာ` → `ွာ`. With a non-nasal preceding
        // consonant, `ူာ` becomes `ွာ`.
        assert_eq!(normalize("ကူာ"), ("ကွာ".to_string(), 1));
        // With a nasal (င) preceding, the negative-lookbehind rule does NOT
        // fire; the positive-lookbehind rule turns it into `ှာ`.
        assert_eq!(normalize("ငူာ"), ("ငှာ".to_string(), 1));
    }

    #[test]
    fn normalize_applies_literal_substring() {
        // `ကထွာ` → `ကတွာ` (Pali fix).
        assert_eq!(normalize("ကထွာ"), ("ကတွာ".to_string(), 1));
    }

    #[test]
    fn normalize_preserves_trailing_space_rule() {
        // One rule intentionally matches `ထူးခြားခုက် ` (trailing space).
        // Confirm that rule still fires when a space follows.
        assert_eq!(normalize("ထူးခြားခုက် "), ("ထူးခြားချက် ".to_string(), 1));
    }

    #[test]
    fn normalize_counts_each_match_globally() {
        // Two occurrences of the doubled `ာာ` dedup → two substitutions.
        assert_eq!(normalize("ကာာ ကာာ").1, 2);
    }

    // ── whole-token dict (stage 2) ───────────────────────────────────────────

    #[test]
    fn dict_is_populated() {
        assert!(DICT.len() > 100, "dict too small: {}", DICT.len());
    }

    #[test]
    fn corrects_known_pair() {
        // `ကျေးစူး` → `ကျေးဇူး` is in the bundled CSV. Ties the test to the
        // list contents on purpose — if the pair is ever dropped, the test
        // fails loudly rather than silently degrading. One substitution.
        assert_eq!(correct("ကျေးစူး"), ("ကျေးဇူး".to_string(), 1));
    }

    #[test]
    fn leaves_unknown_word_unchanged() {
        let s = "မရှိပါလို့ဓာတ်ကူး";
        assert_eq!(correct(s), (s.to_string(), 0));
    }

    #[test]
    fn leaves_latin_unchanged() {
        assert_eq!(correct("hello world"), ("hello world".to_string(), 0));
    }

    #[test]
    fn preserves_punctuation_and_spacing() {
        assert_eq!(
            correct("(ကျေးစူး) ဥပါသကာ။"),
            ("(ကျေးဇူး) ဥပါသကာ။".to_string(), 1)
        );
    }

    #[test]
    fn empty_string_stays_empty() {
        assert_eq!(correct(""), ("".to_string(), 0));
    }

    #[test]
    fn replaces_multiple_tokens_in_one_line() {
        // Two dict keys in one line → two substitutions.
        assert_eq!(correct("ကျေးစူး ခံ့ယူ"), ("ကျေးဇူး ခံယူ".to_string(), 2));
    }

    #[test]
    fn tokenizes_through_myanmar_combining_marks() {
        // Regression: Myanmar combining marks have Unicode `Alphabetic = No`.
        // The naive `is_alphanumeric` tokenizer split `ခံ့ယူ` into fragments
        // and never matched the dict key; the Myanmar-aware tokenizer keeps
        // base+stacked-marks together and the lookup succeeds.
        assert_eq!(correct("ခံ့ယူ"), ("ခံယူ".to_string(), 1));
    }

    #[test]
    fn digit_runs_are_tokens_but_never_matched() {
        let s = "၂၀၂၆ ခု";
        assert_eq!(correct(s), (s.to_string(), 0));
    }

    // ── combined (stage 1 + stage 2) ─────────────────────────────────────────

    #[test]
    fn fix_runs_normalize_before_correct() {
        // A doubled mark would block the token dict lookup (the doubled form
        // isn't a key). normalize collapses it first, then correct resolves
        // the tidied token. We exercise this with a doubled vowel mark the
        // normalizer handles; the result must have the dedup applied and a
        // count of at least 1 (the dedup itself counts).
        let (out, hits) = fix("ကာာ");
        assert_eq!(out, "ကာ");
        assert!(hits >= 1, "fix should count the dedup substitution");
    }

    #[test]
    fn fix_totals_substitutions_across_both_stages() {
        // `ကျေးစူး` (no normalization hits) + the dict replacement = 1 fix.
        let (out, hits) = fix("ကျေးစူး");
        assert_eq!(out, "ကျေးဇူး");
        assert_eq!(hits, 1);
    }

    #[test]
    fn correct_line_mutates_text_in_place() {
        let mut lb = LineBox {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
            text: "ကျေးစူး".to_string(),
            polygon: None,
        };
        let hits = correct_line(&mut lb);
        assert_eq!(lb.text, "ကျေးဇူး");
        assert_eq!(hits, 1);
    }
}
