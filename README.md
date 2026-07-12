# just-ocr

Offline Tesseract OCR GUI built with Tauri 2 + Svelte, using the
[`tesseract-rs`](https://github.com/cafercangundogdu/tesseract-rs) Rust binding.

Tesseract 5.5.2 and Leptonica are compiled from source and statically linked by
the crate, and English/Turkish traineddata are embedded into the binary, so the
resulting app is fully standalone — no system tesseract install required.

## Requirements

- Rust 1.88+, Node 18+
- C++17 compiler (clang/gcc/MSVC) and **CMake** — needed for the one-time
  Tesseract + Leptonica build (cached under `~/Library/Application Support/tesseract-rs` on macOS)
- Internet access on first build (the crate downloads tesseract/leptonica sources)

## Develop

```sh
npm install
cargo tauri dev
```

The first build takes several minutes because it compiles Tesseract. Subsequent
builds reuse the cache and are fast.

## Build a distributable app

```sh
cargo tauri build
```

## Usage

1. Drag an image (PNG/JPG/BMP/WebP) or a PDF onto the dropzone, or click to
   browse. PDFs become one job per page. The **PDF** selector in the toolbar
   chooses how: **Extract** (default, fast) pulls the embedded scan at its
   native resolution; **Render** rasterizes each page at 1500 px height for
   vector or mixed-content PDFs. Set the mode before dropping a PDF.
2. Pick a language and page-segmentation mode (and optionally a char whitelist).
3. Click **Run OCR** — recognized text, mean confidence, and elapsed time appear
   in the result panel. Use **Copy** to copy the text.

## Adding more languages

The `embed-tessdata` feature bundles `eng` and `tur` by default. To embed
others at build time:

```sh
TESSERACT_EMBED_LANGUAGES=eng,fra,deu cargo tauri build
```
