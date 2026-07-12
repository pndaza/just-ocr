/// Formats the display name for a page rendered from a PDF.
///
/// `pdf_name` is the original file name (e.g. `"scan_3.pdf"`); `page` is the
/// 1-based page index. Returns `"<stem> · p<n>"`, e.g. `"scan_3 · p4"`.
pub(crate) fn page_name(pdf_name: &str, page: usize) -> String {
    let stem = std::path::Path::new(pdf_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(pdf_name);
    format!("{stem} · p{page}")
}

#[cfg(test)]
mod tests {
    use super::page_name;

    #[test]
    fn strips_pdf_extension_and_appends_page() {
        assert_eq!(page_name("scan_3.pdf", 4), "scan_3 · p4");
    }

    #[test]
    fn handles_name_without_extension() {
        assert_eq!(page_name("plain", 1), "plain · p1");
    }

    #[test]
    fn handles_path_like_name() {
        assert_eq!(page_name("/tmp/foo/report.pdf", 12), "report · p12");
    }
}

use std::fs;

/// Render every page of `pdf_bytes` to a PNG at `dpi`, returned in page order.
/// Uses a temp directory: pdq's API writes files, not in-memory buffers.
pub(crate) fn render_pages(pdf_bytes: &[u8], dpi: f32) -> Result<Vec<Vec<u8>>, String> {
    // Unique temp dir per call so concurrent renders can't collide.
    let tmp_dir = std::env::temp_dir().join(format!(
        "just-ocr-pdf-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let result = render_pages_into(pdf_bytes, dpi, &tmp_dir);

    // Best-effort cleanup regardless of success/failure.
    let _ = fs::remove_dir_all(&tmp_dir);
    result
}

fn render_pages_into(
    pdf_bytes: &[u8],
    dpi: f32,
    out_dir: &std::path::Path,
) -> Result<Vec<Vec<u8>>, String> {
    let pdf_path = out_dir.join("input.pdf");
    fs::write(&pdf_path, pdf_bytes).map_err(|e| format!("Failed to write temp PDF: {e}"))?;

    let opts = pdq::render::RenderOptions {
        dpi,
        pages: None, // render all pages
    };
    // pdq's output pattern uses a literal `%d` placeholder (not Rust format
    // syntax) for the 1-based page index, zero-padded to the width of the
    // total page count. That padding is what makes the lexicographic sort
    // below match true page order — don't switch to an unpadded pattern.
    let pattern = out_dir.join("page_%d.png");
    let pattern_str = pattern.to_string_lossy().into_owned();
    pdq::render::render_pages(&pdf_path, &pattern_str, &opts)
        .map_err(|e| format!("PDF render failed: {e}"))?;

    // Ascending filename order == page order (relies on pdq's zero-padding).
    let mut files: Vec<_> = fs::read_dir(out_dir)
        .map_err(|e| format!("Failed to read temp dir: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("page_") && n.ends_with(".png"))
                .unwrap_or(false)
        })
        .collect();
    files.sort_by_key(|e| e.file_name());
    files
        .into_iter()
        .map(|e| fs::read(e.path()).map_err(|err| format!("Failed to read PNG: {err}")))
        .collect()
}

#[cfg(test)]
mod render_tests {
    use super::render_pages;
    use std::path::PathBuf;

    fn fixture() -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.pdf");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn renders_at_least_one_nonempty_png() {
        let Some(path) = fixture() else {
            eprintln!("skip: tests/fixtures/tiny.pdf not present");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");
        let pages = render_pages(&bytes, 150.0).expect("render succeeds");
        assert!(!pages.is_empty(), "expected at least one rendered page");
        for png in &pages {
            assert!(!png.is_empty(), "rendered page PNG must be non-empty");
            // PNG magic bytes.
            assert_eq!(
                &png[0..8],
                &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
            );
        }
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = render_pages(b"not a pdf", 150.0).unwrap_err();
        // The wrap prefix from render_pages_into; the trailing detail comes
        // from pdq. Match case-insensitively so a capitalization tweak in the
        // prefix doesn't silently break this test.
        assert!(
            err.to_lowercase().contains("render failed"),
            "expected an error mentioning render failure, got: {err}"
        );
    }
}
