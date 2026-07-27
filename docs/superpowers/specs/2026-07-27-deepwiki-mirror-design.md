# DeepWiki-Style Documentation Mirror — Design

**Status:** Approved
**Date:** 2026-07-27
**Author:** brainstorming session (agent + maintainer)

## Summary

Add a hand-curated, DeepWiki-style documentation wiki to `just-ocr` as a set of
version-controlled Markdown files under `docs/wiki/`. The wiki is
**human-focused** (agents continue to use `AGENTS.md`) and explains the
codebase's structure, pipeline, engines, build, and PDF support with prose plus
Mermaid diagrams. This is a **local mirror** — no MCP server, no live service,
no auto-generation. It is a one-time snapshot that humans refresh by hand when
the referenced code changes.

The motivation: DeepWiki (the service) auto-generates navigable, diagram-rich
documentation for public GitHub repos. We want that *feel* — multi-page,
organized, visual — but kept entirely in-repo so it is readable offline, on
GitHub, and survives independently of any external service.

## Goals & Non-Goals

### Goals

- **G1.** A navigable, multi-page wiki that a new contributor can read
  top-to-bottom to understand how `just-ocr` works.
- **G2.** Mermaid diagrams on every page (8 pages, 9 diagrams total —
  Architecture and Engines each carry two).
- **G3.** Cross-linked structure: Home index + per-page nav + Related footers.
  No orphan pages.
- **G4.** Grounded in the real code — every claim traceable to a
  `file_path:line` reference or a quoted signature.
- **G5.** Zero new build tooling, scripts, dependencies, or CI. Pure docs.

### Non-Goals

- **N1.** Not an auto-syncing system. Refresh is manual; drift is mitigated by a
  per-page source-of-truth banner and a `Last reviewed` date.
- **N2.** Not a replacement for `AGENTS.md`. Agents ignore the wiki; the wiki
  is the human mirror of the same conventions.
- **N3.** Not an exhaustive API reference. Function signatures live in the
  source; the wiki explains purpose, responsibility, and *why*.
- **N4.** Not the DeepWiki MCP integration. We explicitly chose the local-only
  scope over the live MCP server option.
- **N5.** No roadmap, changelog, or contribution guide (none exist; we don't
  invent them).

## Deliverables

```
docs/wiki/
├── Home.md                     index: 30-sec tour + nav grid
├── Architecture.md             layers, IPC contract, threading
├── OCR-Pipeline.md             run_ocr decision tree (the centerpiece)
├── Engines.md                  Tesseract / Kraken / PP-OCR deep dive
├── Backend-Reference.md        Rust modules + Tauri command catalog
├── Frontend-Reference.md       Svelte components + ocr.ts wrappers
├── PDF-Support.md              extract vs render, image modes, temp files
└── Building-and-Releasing.md   dev setup, opt-level trick, LFS, CI matrix
```

Plus a **single-line edit** to `README.md` adding a 📚 wiki link near the top.

Single commit on `main`:
`docs(wiki): add hand-curated DeepWiki-style documentation`.

## Page Specifications

Each page targets **800–2000 words** — enough to explain *why*, short enough to
read in one sitting. Pages are written in dependency order so cross-links always
resolve to existing files:
`Home → Architecture → OCR-Pipeline → Engines → Backend → Frontend → PDF → Building`.

### 1. `Home.md`

**Role.** Landing page. 30-second "what is just-ocr," the three-engine
one-liner, screenshot reference (link into `screenshots/`), and a navigation
grid to the other seven pages. States up front: this wiki is for humans; agents
see [`AGENTS.md`](../../AGENTS.md).

**Diagram (1):** *system map* — high-level box diagram:
Frontend (Svelte 5) ↔ Tauri IPC ↔ Rust backend ↔ engines (Tesseract/Kraken/PP-OCR)
↔ bundled models.

### 2. `Architecture.md`

**Role.** The layered view. Process topology (UI thread vs `spawn_blocking` vs
rayon pool), the IPC contract conventions
(`#[serde(rename_all="camelCase")]`, the `Vec<u8>`-as-JSON-array cost and the
temp-file workaround for large binary), the two event channels (`pdf-progress`,
`lang-download://{code}`), and how frontend state persists
(`just-ocr:` localStorage keys; theme no-flash inline script in `index.html`).

**Diagrams (2):**
- *Layered architecture* — Frontend → Tauri IPC bridge → Rust commands → engine
  dispatch → vendored crates → bundled models.
- *Threading model* — UI thread / `spawn_blocking` pool / rayon pool, with what
  runs where. Calls out the **libtesseract is not thread-safe → Tesseract recog
  stays serial** invariant.

### 3. `OCR-Pipeline.md` (the centerpiece)

**Role.** Walks `engine.rs::run_ocr` end to end: decode → language switch →
(Myanmar: segmenter choice → per-line recognizer axis) vs (any other language:
full-page Tesseract). Explains:

- The 2×2 segmenter/recognizer matrix on the Myanmar path:
  `{ppocr, kraken} × {tesseract, kraken}`.
- Why **PP-OCR is the default segmenter** (~25 ms/page) and **Kraken is the
  default recognizer** on the not-Myanmar→Myanmar transition (enforced in
  `Toolbar.svelte`).
- Dewarp + masked-bbox fallback in Kraken recog; quad→polygon+synth-baseline in
  PP-OCR seg.
- Confidence semantics (Kraken reports none → `conf = -1`).
- Field-presence rules: `segmentation_ms`/`recognition_ms`/`LineBox.polygon` are
  `Some` only on the Myanmar path, `None` (and serde-skipped for `polygon`) on
  full-page Tesseract.

**Diagram (1):** *full pipeline flowchart* — the decision tree from `run_ocr`
entry to `OcrResult`, annotated with the `[ocr] ... N ms` timing logs and the
parallel (Kraken, rayon) vs serial (Tesseract) branch.

**Links:** the Kraken engine design spec
(`docs/superpowers/specs/2026-07-18-kraken-engine-design.md`) and PP-OCR
segmentation design spec
(`docs/superpowers/specs/2026-07-27-ppocr-segmentation-design.md`) as
"design rationale" references.

### 4. `Engines.md`

**Role.** Three-engine deep dive. Per engine: what it is, where the code lives,
how it's loaded (bundled `include_bytes!` vs override path), thread-safety, and
trade-offs.

- **Tesseract** — Latin full-page; serial (libtesseract); embedded via
  `tesseract-rs` `embed-tessdata` feature gated by
  `TESSERACT_EMBED_LANGUAGES=eng`; Burmese shipped separately via
  `include_bytes!("mya.traineddata")` in `languages.rs`.
- **Kraken** — vendored candle NN (`kraken-engine/`); seg + recog; rayon
  parallel (`Send + Sync`); `accelerate` feature on macOS/aarch64 (Apple
  vDSP/BNNS, ~1.15× matmul); override requires **both** files.
- **PP-OCR** — vendored PP-OCRv6 tiny-det DBNet (`ppocr-engine/`); quad →
  `close_polygon` + `synth_midline(8)`; override is **single-file,
  all-or-nothing**.

Ends with the **model override rules table**: Kraken = both required,
PP-OCR = all-or-nothing, languages = embedded → bundled → on-disk.

**Diagrams (2):**
- *Engine decision matrix* — the 2×2 grid with valid combos and characteristics.
- *Model loading & override flow* — `OnceCell<Arc<...>>` lazy load per engine,
  bundled vs override resolution.

### 5. `Backend-Reference.md`

**Role.** One section per Rust file in `src-tauri/src/` (`lib.rs`, `engine.rs`,
`languages.rs`, `pdf.rs`, `segmentation.rs`, `segmenter_adapters.rs`,
`tesseract_page.rs`, `tesseract_line.rs`) — module responsibility, key public
types, and the non-obvious bits. Then a complete **Tauri command table**: all 9
commands with signature, defined-in file, purpose, and which events they emit
(`pdf-progress`, `lang-download://{code}`). Finally the temp-dir lifecycle
(`just-ocr-<pid>-<seq>` naming, `sweep_stale_temp_dirs` at startup,
`remove_session_temp_dirs` at shutdown, and why the PID namespace matters).

**Diagram (1):** *module dependency graph* — which `src/*.rs` imports what,
showing `engine.rs` as the hub and the `Segmenter` trait abstraction over the
two adapters.

### 6. `Frontend-Reference.md`

**Role.** Component tree (`App.svelte` → `Toolbar` / `Preview` / `Output` /
`Thumbnail` / `Settings` / `LanguageManager` / `PdfModeDialog`), each with a
one-line purpose and the non-obvious tricks:

- **Preview** renders the source image + OCR overlay in **one shared SVG**
  (image and boxes share a single coordinate system — no JS measurement drift).
- **Thumbnail** uses **virtual scrolling** (visible rows + overscan only).
- **Toolbar** enforces language-driven UI rules (Myanmar defaults engine to
  Kraken on the not-my→mya transition; hides whitelist for Kraken recog; hides
  the engine selector entirely for non-Myanmar).

Then `src/lib/ocr.ts` IPC wrappers grouped by concern (jobs, languages,
persisted settings, PDF), and the `just-ocr:` localStorage pattern with
`try/catch` for private mode.

**Diagram (1):** *component tree* — parent/child wiring and which component
owns which slice of state.

### 7. `PDF-Support.md`

**Role.** The extract-vs-render duality (lopdf extracts the largest embedded
image XObject per page, default; hayro rasterizes at `PDF_RENDER_HEIGHT=1500`
in render mode). Image modes (Color / Gray / Bw-with-Otsu). The decode filter
chain (`decode_image` dispatches DCT/JPX/JBIG2/CCITT/Flate/RunLength/PNG
predictor). Per-page temp PNGs at
`$TMPDIR/just-ocr-<pid>-<seq>/pNNN.png` and why they're path-returned (not
bytes) — the IPC cost rule. The `pdf-progress` event channel. The
PID-namespaced cleanup invariants (don't change the naming;
`just_ocr_temp_pid` parses it).

**Diagram (1):** *PDF → OCR flow* — drop → `PdfModeDialog` → `render_pdf` →
temp PNGs → `Preview` lazy-loads via `ensureThumb` → OCR.

### 8. `Building-and-Releasing.md`

**Role.** Dev prerequisites (Rust 1.88+, Node 18+, C++17 compiler, CMake,
`git lfs install` once per machine), the `cargo tauri dev` / `cargo tauri build`
/ `npm test` commands. Then the **load-bearing gotchas**:

- **The opt-level trick** — `[profile.dev.package."*"] opt-level = 3` only
  optimizes non-workspace deps; `kraken-engine` and `ppocr-engine` are
  deliberately kept as path dependencies *outside* any `[workspace]` table so
  dev-build NN inference + hayro rasterization stay fast (~10–15×; without it
  Kraken OCR is ~30 s/image and PDF render mode is unusable). "Do not fix by
  adding a `[workspace]` table."
- **The patched `tesseract-rs` fork** — `[patch.crates-io]` pins to
  `pndaza/tesseract-rs` tag `v0.3.0-macos-fix`, which adds
  `-mmacosx-version-min=10.15` to `CMAKE_CXX_FLAGS` so Xcode 26+ SDKs accept
  tesseract 5.x's `std::filesystem`.
- **`TESSERACT_EMBED_LANGUAGES=eng`** in `src-tauri/.cargo/config.toml` — limits
  which traineddata `tesseract-rs` compiles in; Burmese is intentionally
  excluded and shipped separately via `include_bytes!`.
- **Git LFS** — `kraken-models/*.safetensors` and `ppocr-models/*.safetensors`
  are LFS-tracked; a plain clone leaves pointer stubs and `include_bytes!`
  fails; CI has an explicit `git lfs pull` retry step.

Then the release flow: push a `v*` tag → `.github/workflows/release.yml` →
draft GitHub Release (manual publish). The CI matrix.

**Diagram (1):** *CI matrix* — 4 lanes (macOS aarch64 cross-compiled,
macOS x86_64 native, Ubuntu 22.04, Windows) with per-lane setup, the
`git lfs pull` retry step, and the `CFLAGS`/`CXXFLAGS` flag story explained.

## Conventions

### Voice & density

Matches the repo's existing style (see the heavy `//!` module headers and
`AGENTS.md`): dense, *why*-heavy prose with load-bearing explanations. No
marketing tone. Quote real signatures/types only when they clarify; don't
reproduce whole functions — point to source.

### Code references

Two forms:
- **Inline symbol:** `` `engine.rs::run_ocr` `` or `OcrOpts`.
- **Clickable file link:** `[engine.rs](../../src-tauri/src/engine.rs)` — paths
  relative to `docs/wiki/`, so `../../src-tauri/src/...` and `../../src/...`.

### Diagram conventions

- All Mermaid, fenced ` ```mermaid `. Validated to render on GitHub.
- One `graph TD` (or `flowchart TD`) per concept; nodes ≤ ~6 words.
- **Consistent shape semantics across all pages:**
  - Rounded rectangles = processes / functions.
  - Rectangles = data/code artifacts (files, structs).
  - Diamonds = decisions.
  - Cylinders = persistent state (disk, localStorage).
  - Dashed arrows = async/event flow; solid = direct call.
- Every diagram has a one-line caption above it.

### Cross-linking

- Each page opens with `← Back to Home` and ends with a **Related** footer
  linking 1–3 sibling pages.
- In-prose links use relative paths: `[OCR Pipeline](./OCR-Pipeline.md)`.
- `Home.md` has a nav grid linking all seven other pages.
- No orphan pages.

### Source-of-truth banner

Every page starts with this HTML comment (not rendered on GitHub, visible in
source):

```html
<!-- Wiki page · hand-curated · source of truth is the code, not this file.
     Last reviewed: 2026-07-27. Refresh when the referenced code changes. -->
```

### Relationship to existing docs

| Existing doc | Role | Relationship to wiki |
|---|---|---|
| `README.md` | User-facing install/download | Wiki goes deeper; README gets one new line linking the wiki. |
| `AGENTS.md` | Agent editing conventions | Wiki is the human mirror. No duplication; wiki links to `AGENTS.md` where relevant. |
| `docs/superpowers/specs/*` | Design history | Wiki references them as "design rationale" where a page benefits. |
| `docs/notes/*` | Scratch notes | Not linked from the wiki. |

### Formatting specifics

- GitHub-flavored Markdown.
- Tables for the command catalog and override matrix.
- No HTML except the comment banner.
- Admonitions via blockquote: `> **Note:** ...`.
- No emojis except the single 📚 in the README link.

## Diagram Inventory (9 total)

| Page | Diagram | Type |
|---|---|---|
| Home | System map | `graph LR` |
| Architecture | Layered architecture | `graph TD` |
| Architecture | Threading model | `graph LR` |
| OCR-Pipeline | Full pipeline flowchart | `flowchart TD` |
| Engines | Engine decision matrix (2×2) | `graph TD` |
| Engines | Model loading & override flow | `flowchart TD` |
| Backend-Reference | Module dependency graph | `graph LR` |
| Frontend-Reference | Component tree | `graph TD` |
| Building-and-Releasing | CI matrix | `graph TD` |

## Verification (before claiming done)

1. `find docs/wiki -name '*.md'` → exactly 8 files present.
2. Grep for dead internal links: every `](./X.md)` and `](../../...)` target
   must resolve to a real file on disk.
3. Grep for stray `TODO` / `TBD` / placeholder text → none.
4. Mermaid sanity: count ` ```mermaid ` open fences = 9, each closed with a
   matching ` ``` `.
5. `git status` shows exactly: 8 new files under `docs/wiki/` + 1 modified
   `README.md`. Nothing else.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Docs drift from code over time. | High (eventually) | Medium | Per-page source-of-truth banner + `Last reviewed` date. Manual refresh; accepted for a snapshot wiki. |
| Mermaid syntax doesn't render on GitHub. | Low | High | Use only well-supported Mermaid constructs; balanced nodes/edges; valid identifiers. |
| Dead cross-links. | Medium | Low | Verification step 2 grep-checks every link target. |
| Duplication with `AGENTS.md` causes skew. | Medium | Medium | Wiki is human-focused; no copy-paste of agent conventions. Wiki links out to `AGENTS.md`. |
| Scope creep into roadmap/changelog. | Low | Low | Explicit non-goal (N5). |

## Out of Scope

- Auto-generation script or CI to refresh the wiki.
- The DeepWiki MCP server integration (chose local-only).
- Editing any source code. Docs-only change.
- New dependencies, `package.json`, or `Cargo.toml` changes.
- Roadmap, changelog, contribution guide.
