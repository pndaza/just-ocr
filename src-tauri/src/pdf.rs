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
