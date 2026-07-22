# AGENTS.md

Workspace instructions for ZCode agents working in `just-ocr`.

## What this is

**Just OCR** — a fully offline OCR desktop app (Tauri v2). No internet, no
system installs; download and run. Extracts text from images and PDFs.

- Latin scripts → **Tesseract** (embedded via `tesseract-rs` `embed-tessdata`)
- Burmese / complex scripts → **Kraken** (a vendored candle port in
  `src-tauri/kraken-engine`, does layout segmentation; recognition can be
  Kraken or Tesseract per-line)
- All models bundled in the binary via `include_bytes!` — zero setup.

Stack: **Rust** backend (`src-tauri/`) + **Svelte 5 + TypeScript** frontend
(`src/`), built with **Vite**, glued by **Tauri v2**.

## Key directories

```
src/                      Svelte 5 frontend (UI, IPC wrappers)
  lib/ocr.ts              ← all Tauri `invoke()` calls live here
  lib/result.ts           OcrResult/LineBox types + plaintext projection
  theme.ts                light/dark, persisted in localStorage
src-tauri/                Rust backend (the app crate `just_ocr_lib`)
  src/lib.rs              Tauri commands + entry point (run())
  src/engine.rs           OCR dispatcher: Kraken seg → recognizer
  src/languages.rs        language install/download/resolve
  src/pdf.rs              PDF → PNG (extract or render mode)
  src/tesseract_page.rs   non-Myanmar full-page Tesseract path
  src/tesseract_line.rs   per-line Tesseract (Myanmar + Kraken seg)
  kraken-engine/          vendored Kraken OCR engine (separate crate)
kraken-models/            Burmese .safetensors (Git LFS — see below)
.github/workflows/release.yml   tag-triggered multi-platform release CI
docs/                     design notes + superpowers plans/specs
```

## Commands

```sh
npm install               # first-time frontend deps
cargo tauri dev           # dev app (FIRST BUILD IS SLOW — compiles Tesseract
                          #   + candle NN crates from source, several minutes)
cargo tauri build         # distributable bundle (.dmg/.msi/.deb/AppImage)

npm run dev               # frontend-only dev server (no Tauri)
npm run build             # vite build (frontend only)
npm test                  # frontend tests (vitest run)
npm run test -- --watch   # vitest watch

# Backend (run inside src-tauri/):
cargo test                # Rust unit/integration tests
cargo test -- --nocapture # with timing/log output
cargo run --example smoke_kraken   # kraken smoke test on a fixture
cargo run --example bench_kraken   # kraken benchmark
```

There is no separate lint script configured. TypeScript uses `strict` +
`checkJs`; type errors surface via `vite build`.

## Architecture rules that matter for edits

**IPC boundary.** Frontend → backend goes through Tauri `invoke()` commands
declared in `src/lib.rs` `run()`'s `invoke_handler`. Every new Rust command
must be added there **and** wrapped in `src/lib/ocr.ts`. Conventions:

- Rust structs crossing IPC use `#[serde(rename_all = "camelCase")]`; the TS
  side mirrors the field names exactly. Keep both in sync.
- `Vec<u8>` is serialized by Tauri as a JSON number array — expensive. For
  large binary (e.g. PDF page PNGs) write to a temp file and return only the
  `path`; the frontend reads it on demand (see `render_pdf` / `ReadFile`).
- Events (progress) use `app.emit("event-name", payload)` + frontend
  `listen()`. See `pdf-progress` and `lang-download://{code}`.

**OCR pipeline dispatch** (`engine.rs::run_ocr`) is language-driven:
- `language == "mya"` → Kraken segmentation (always, regardless of recognizer
  choice) → per-line recognition by `engine` ("kraken" | "tesseract").
- any other language → full-page Tesseract with the user's `psm`.

**Threading.** Heavy OCR/PDF work runs on `spawn_blocking` (never block the UI
thread). Kraken recognition is `Send + Sync` → parallelized over rayon.
**libtesseract is NOT thread-safe across concurrent calls** → the Tesseract
recognizer path stays serial. Do not parallelize Tesseract calls.

**Models.** Bundled Kraken models (`BUNDLED_SEG`/`BUNDLED_REC`) are embedded
via `include_bytes!` with paths relative to `src-tauri/src/`. A user can
override by placing both `.safetensors` in the app-local-data `kraken-models/`
dir (partial overrides are ignored). `resolve_override_models` checks both
files exist. Kraken engine is lazy-loaded once process-wide via `OnceCell`.

**Temp files.** PDF page PNGs go to `just-ocr-<pid>-<seq>/pNNN.png` under the
system temp dir. PID-namespacing lets `sweep_stale_temp_dirs()` (startup) and
`remove_session_temp_dirs()` (shutdown) reclaim dirs without touching other
instances. Don't change the `just-ocr-<pid>-<seq>` naming — `just_ocr_temp_pid`
parses it.

## Critical gotchas

1. **kraken-engine is deliberately NOT a workspace member.** See the long
   comment in `src-tauri/Cargo.toml`. `[profile.dev.package."*"] opt-level=3`
   only optimizes non-workspace deps; keeping kraken-engine outside the
   workspace makes dev-build OCR ~10× faster. Do not "fix" this by adding it
   to a workspace.

2. **Git LFS for models.** `kraken-models/*.safetensors` are LFS-tracked
   (`.gitattributes`). Run `git lfs install` once per machine, then clone
   normally. A plain clone without LFS leaves pointer stubs and the Rust
   `include_bytes!` build fails. CI has an explicit `git lfs pull` step.

3. **macOS build needs the patched tesseract-rs fork.** `[patch.crates-io]`
   in `Cargo.toml` pins `tesseract-rs` to `pndaza/tesseract-rs` tag
   `v0.3.0-macos-fix` (adds `-mmacosx-version-min=10.15` to CMAKE_CXX_FLAGS
   so Xcode 26+ SDKs accept tesseract 5.x's `std::filesystem`). The release CI
   additionally sets `CFLAGS`/`CXXFLAGS` (not `MACOSX_DEPLOYMENT_TARGET`) for
   the `cc` crate — see the comment block in `release.yml`.

4. **`TESSERACT_EMBED_LANGUAGES=eng`** (`src-tauri/.cargo/config.toml`)
   limits which traineddata `tesseract-rs` compiles in. Burmese (`mya`) is
   shipped separately via `include_bytes!("mya.traineddata")` in
   `languages.rs`, so it is intentionally NOT in that env var.

## Conventions

- **Doc comments are heavy and load-bearing** throughout the Rust backend —
  they explain *why*, timing implications, and thread-safety. Match that
  density when editing. Many files open with a `//!` module overview.
- **Per-stage timing logs** via `log::info!("[ocr] ...: {:.0} ms")`. Default
  log level is `info` (set in `run()`); bump with `RUST_LOG=debug` (per-line
  recog) or `RUST_LOG=trace` (kraken internals). Add timing logs to new
  pipeline stages.
- **Frontend persistence** keys are prefixed `just-ocr:` (theme, language,
  engine) in localStorage; wrap in try/catch (private mode).
- TS is strict + `checkJs`; no separate lint step — `vite build` is the type
  gate. Svelte 5 (runes) components live in `src/lib/`.

## Release

Push a tag matching `v*` (e.g. `v0.1.0`) to trigger `.github/workflows/release.yml`,
which builds macOS (aarch64 + x86_64), Linux, and Windows and attaches bundles
to a **draft** GitHub Release (review then publish manually). The app version
source of truth is `src-tauri/tauri.conf.json` `version`.

## Docs worth reading before sensitive changes

- `docs/superpowers/specs/2026-07-18-kraken-engine-design.md` — Kraken engine
  integration design.
- `docs/superpowers/plans/2026-07-12-pdf-support.md` — PDF support plan.
- `docs/notes/2026-07-19-polygon-overlay.md` — line-box overlay notes.
