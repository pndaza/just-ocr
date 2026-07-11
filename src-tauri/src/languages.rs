//! User-installed language model management.
//!
//! Tesseract traineddata files (e.g. `fra.traineddata`) are stored in the app's
//! local data directory under `tessdata/`. Languages bundled into the binary via
//! the `embed-tessdata` feature (eng, tur) are always available; user-installed
//! languages extend that set. OCR resolves a language by trying the embedded set
//! first, then falling back to the on-disk tessdata.

use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tesseract_rs::{embedded_languages, get_embedded_tessdata};

/// Description of a language the user can install or has installed.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
    pub source: &'static str, // "embedded" | "installed" | "available"
}

/// Traineddata sources on GitHub. `fast` is the default; `best` and `standard`
/// trade speed for accuracy.
const TESSDATA_REPOS: &[(&str, &str)] = &[
    ("fast", "tesseract-ocr/tessdata_fast"),
    ("standard", "tesseract-ocr/tessdata"),
    ("best", "tesseract-ocr/tessdata_best"),
];

/// A curated subset of the ~100 tessdata languages, mapping the code to a
/// readable name. Codes not listed here still work if the user types them.
const KNOWN_LANGUAGES: &[(&str, &str)] = &[
    ("afr", "Afrikaans"),
    ("amh", "Amharic"),
    ("ara", "Arabic"),
    ("asm", "Assamese"),
    ("aze", "Azerbaijani"),
    ("aze_cyrl", "Azerbaijani (Cyrillic)"),
    ("bel", "Belarusian"),
    ("ben", "Bengali"),
    ("bod", "Tibetan"),
    ("bos", "Bosnian"),
    ("bre", "Breton"),
    ("bul", "Bulgarian"),
    ("cat", "Catalan"),
    ("ceb", "Cebuano"),
    ("ces", "Czech"),
    ("chi_sim", "Chinese (Simplified)"),
    ("chi_tra", "Chinese (Traditional)"),
    ("chr", "Cherokee"),
    ("cos", "Corsican"),
    ("cym", "Welsh"),
    ("dan", "Danish"),
    ("deu", "German"),
    ("div", "Dhivehi"),
    ("dzo", "Dzongkha"),
    ("ell", "Greek"),
    ("eng", "English"),
    ("enm", "Middle English"),
    ("epo", "Esperanto"),
    ("equ", "Math / equation"),
    ("est", "Estonian"),
    ("eus", "Basque"),
    ("fao", "Faroese"),
    ("fas", "Persian"),
    ("fil", "Filipino"),
    ("fin", "Finnish"),
    ("fra", "French"),
    ("frk", "Frankish"),
    ("frm", "Middle French"),
    ("fry", "Western Frisian"),
    ("gla", "Scottish Gaelic"),
    ("gle", "Irish"),
    ("glg", "Galician"),
    ("grc", "Ancient Greek"),
    ("guj", "Gujarati"),
    ("hat", "Haitian Creole"),
    ("heb", "Hebrew"),
    ("hin", "Hindi"),
    ("hrv", "Croatian"),
    ("hun", "Hungarian"),
    ("hye", "Armenian"),
    ("iku", "Inuktitut"),
    ("ind", "Indonesian"),
    ("isl", "Icelandic"),
    ("ita", "Italian"),
    ("ita_old", "Italian (old)"),
    ("jav", "Javanese"),
    ("jpn", "Japanese"),
    ("kan", "Kannada"),
    ("kat", "Georgian"),
    ("kat_old", "Georgian (old)"),
    ("kaz", "Kazakh"),
    ("khm", "Khmer"),
    ("kir", "Kyrgyz"),
    ("kmr", "Kurmanji"),
    ("kor", "Korean"),
    ("kor_vert", "Korean (vertical)"),
    ("lao", "Lao"),
    ("lat", "Latin"),
    ("lav", "Latvian"),
    ("lit", "Lithuanian"),
    ("ltz", "Luxembourgish"),
    ("mal", "Malayalam"),
    ("mar", "Marathi"),
    ("mkd", "Macedonian"),
    ("mlt", "Maltese"),
    ("mon", "Mongolian"),
    ("mri", "Maori"),
    ("msa", "Malay"),
    ("mya", "Burmese"),
    ("nep", "Nepali"),
    ("nld", "Dutch"),
    ("nor", "Norwegian"),
    ("oci", "Occitan"),
    ("ori", "Odia"),
    ("osd", "Orientation & script"),
    ("pan", "Punjabi"),
    ("pol", "Polish"),
    ("por", "Portuguese"),
    ("pus", "Pashto"),
    ("que", "Quechua"),
    ("ron", "Romanian"),
    ("rus", "Russian"),
    ("san", "Sanskrit"),
    ("sin", "Sinhala"),
    ("slk", "Slovak"),
    ("slv", "Slovenian"),
    ("snd", "Sindhi"),
    ("spa", "Spanish"),
    ("spa_old", "Spanish (old)"),
    ("sqi", "Albanian"),
    ("srp", "Serbian"),
    ("srp_latn", "Serbian (Latin)"),
    ("sun", "Sundanese"),
    ("swa", "Swahili"),
    ("swe", "Swedish"),
    ("syr", "Syriac"),
    ("tam", "Tamil"),
    ("tat", "Tatar"),
    ("tel", "Telugu"),
    ("tgk", "Tajik"),
    ("tha", "Thai"),
    ("tir", "Tigrinya"),
    ("ton", "Tongan"),
    ("tur", "Turkish"),
    ("uig", "Uyghur"),
    ("ukr", "Ukrainian"),
    ("urd", "Urdu"),
    ("uzb", "Uzbek"),
    ("uzb_cyrl", "Uzbek (Cyrillic)"),
    ("vie", "Vietnamese"),
    ("yid", "Yiddish"),
    ("yor", "Yoruba"),
];

/// Resolve the on-disk directory that holds user-installed `*.traineddata`.
pub fn tessdata_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_local_data_dir()
        .expect("app local data dir")
        .join("tessdata")
}

fn installed_codes(app: &AppHandle) -> Vec<String> {
    let dir = tessdata_dir(app);
    let mut codes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(code) = name.strip_suffix(".traineddata") {
                    codes.push(code.to_string());
                }
            }
        }
    }
    codes.sort();
    codes
}

fn pretty_name(code: &str) -> String {
    KNOWN_LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, n)| n.to_string())
        .unwrap_or_else(|| code.to_string())
}

/// List every available language: embedded ones, plus any installed on disk.
/// Duplicates (an installed copy of an embedded language) collapse to "embedded".
#[tauri::command]
pub fn list_languages(app: AppHandle) -> Vec<LanguageInfo> {
    let mut out: Vec<LanguageInfo> = embedded_languages()
        .into_iter()
        .map(|c| LanguageInfo {
            code: c.to_string(),
            name: pretty_name(c),
            source: "embedded",
        })
        .collect();

    let embedded: Vec<String> = embedded_languages().into_iter().map(String::from).collect();
    for code in installed_codes(&app) {
        if embedded.contains(&code) {
            continue;
        }
        out.push(LanguageInfo {
            code: code.clone(),
            name: pretty_name(&code),
            source: "installed",
        });
    }

    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

/// The catalog of languages available to download (all known codes, excluding
/// ones already embedded or installed).
#[tauri::command]
pub fn downloadable_languages(app: AppHandle) -> Vec<LanguageInfo> {
    let have: Vec<String> = list_languages(app)
        .into_iter()
        .map(|l| l.code)
        .collect();
    KNOWN_LANGUAGES
        .iter()
        .filter(|(c, _)| !have.contains(&(*c).to_string()))
        .map(|(c, n)| LanguageInfo {
            code: c.to_string(),
            name: n.to_string(),
            source: "available",
        })
        .collect()
}

/// Progress emitted to the frontend during a download.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub language: String,
    pub downloaded: u64,
    pub total: u64,
}

/// Download a language model from the tesseract-ocr GitHub repos.
///
/// `variant` is one of "fast" (default), "standard", "best".
#[tauri::command]
pub async fn download_language(
    app: AppHandle,
    language: String,
    variant: Option<String>,
) -> Result<(), String> {
    let variant = variant.unwrap_or_else(|| "fast".to_string());
    let repo = TESSDATA_REPOS
        .iter()
        .find(|(k, _)| *k == variant)
        .map(|(_, r)| *r)
        .ok_or_else(|| format!("Unknown variant '{variant}'"))?;

    let url = format!(
        "https://raw.githubusercontent.com/{repo}/main/{language}.traineddata"
    );

    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Language '{language}' not found in {variant} repository (HTTP {})",
            resp.status()
        ));
    }

    let total = resp.content_length().unwrap_or(0);
    let event_name = format!("lang-download://{}", language);

    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::with_capacity(total as usize);
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream
        .next()
        .await
        .transpose()
        .map_err(|e| format!("Download stream error: {e}"))?
    {
        buf.extend_from_slice(&chunk);
        downloaded += chunk.len() as u64;
        let _ = app.emit(
            &event_name,
            DownloadProgress {
                language: language.clone(),
                downloaded,
                total,
            },
        );
    }

    save_traineddata(&app, &language, &buf)?;
    Ok(())
}

/// Install a language from a local `.traineddata` file chosen by the user.
/// `source_path` is an absolute path on disk.
#[tauri::command]
pub fn install_local_language(
    app: AppHandle,
    source_path: String,
) -> Result<LanguageInfo, String> {
    let path = Path::new(&source_path);
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Invalid file name".to_string())?;

    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;
    save_traineddata(&app, filename, &bytes)?;
    Ok(LanguageInfo {
        code: filename.to_string(),
        name: pretty_name(filename),
        source: "installed",
    })
}

fn save_traineddata(app: &AppHandle, code: &str, bytes: &[u8]) -> Result<(), String> {
    let dir = tessdata_dir(app);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create tessdata dir: {e}"))?;
    let dest = dir.join(format!("{code}.traineddata"));
    std::fs::write(&dest, bytes).map_err(|e| format!("Failed to write tessdata: {e}"))?;
    Ok(())
}

/// Delete a user-installed language. Embedded languages cannot be removed.
#[tauri::command]
pub fn delete_language(app: AppHandle, code: String) -> Result<(), String> {
    if embedded_languages().iter().any(|c| *c == code.as_str()) {
        return Err(format!(
            "'{}' is bundled into the app and cannot be removed.",
            code
        ));
    }
    let dest = tessdata_dir(&app).join(format!("{code}.traineddata"));
    if !dest.exists() {
        return Err(format!("'{code}' is not installed."));
    }
    std::fs::remove_file(&dest).map_err(|e| format!("Failed to delete: {e}"))?;
    Ok(())
}

/// Load the traineddata bytes for a language, embedded first then on-disk.
/// Returns `(bytes, was_embedded)`.
pub fn resolve_tessdata(app: &AppHandle, language: &str) -> Result<(Vec<u8>, bool), String> {
    if let Some(bytes) = get_embedded_tessdata(language) {
        return Ok((bytes.to_vec(), true));
    }
    let path = tessdata_dir(app).join(format!("{language}.traineddata"));
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("Language '{language}' not available: {e}"))?;
    Ok((bytes, false))
}
