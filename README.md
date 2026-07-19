# Just OCR

A simple, fully offline OCR app. No internet connection needed, no system
installs required — just download and run.

![Just OCR recognizing Burmese with Kraken](screenshots/just-ocr-app-ss.webp)

## What it does

- Extracts text from images and PDFs
- Supports Latin scripts (English, French, etc.)
- Supports Burmese (and other complex scripts)

## How it works

- Latin text is recognized with [Tesseract](https://github.com/tesseract-ocr/tesseract)
- Burmese and similar scripts are handled by a built-in [Kraken](https://kraken.re) engine
  (vendored [candle](https://github.com/huggingface/candle) port)
- All models are bundled inside the app — nothing extra to install

For Burmese, Kraken handles layout segmentation (Tesseract's is poor for the
script) and you can pick either engine for recognition. For everything else,
Tesseract does both layout and recognition.

## Download

Prebuilt binaries are on the [releases page](https://github.com/pndaza/just-ocr/releases):
`.dmg` for macOS, `.msi`/`.exe` for Windows, `.deb`/`.AppImage` for Linux.

## Develop

Requires Rust 1.88+, Node 18+, a C++17 compiler, and CMake.

```sh
npm install
cargo tauri dev
```

The first build takes several minutes (it compiles Tesseract + the candle
neural-network crates from source). Subsequent builds reuse the cache.

The Kraken models in `kraken-models/` are tracked via Git LFS — run
`git lfs install` once per machine, then clone normally.

## Build a distributable app

```sh
cargo tauri build
```

## Release

Push a tag matching `v*` to trigger the release workflow, which builds all
platforms and attaches the bundles to a draft GitHub Release:

```sh
git tag v0.1.0
git push origin v0.1.0
```
