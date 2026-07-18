<script lang="ts">
  import type { Job } from "./ocr";
  import { plainText } from "./result";

  interface Props {
    job: Job | null;
  }
  let { job }: Props = $props();

  // The recognized text is a projection of the structured `OcrResult` (lines
  // joined with "\n"). Falls back to "" until the job is done.
  let displayText = $derived.by(() => {
    if (!job || job.status !== "done" || !job.result) return "";
    return plainText(job.result).replace(/\s+$/, "");
  });

  let copied = $state(false);

  async function copy() {
    if (!displayText) return;
    await navigator.clipboard.writeText(displayText);
    copied = true;
    setTimeout(() => (copied = false), 1300);
  }
</script>

<div class="panel" role="region" aria-label="Recognized text">
  <div class="head">
    <span class="title">Text</span>
    {#if job?.status === "done"}
      <span class="meta">{job.elapsedMs} ms</span>
      <button class="copy" onclick={copy} disabled={!displayText.trim()}>
        {copied ? "Copied ✓" : "Copy"}
      </button>
    {/if}
  </div>

  <div class="body">
    {#if !job}
      <div class="placeholder">Select an image to see its text.</div>
    {:else if job.status === "error"}
      <div class="error">
        <strong>OCR failed for this image.</strong>
        <pre>{job.error}</pre>
      </div>
    {:else if job.status === "running"}
      <div class="placeholder"><span class="spin" aria-hidden="true"></span> Recognizing…</div>
    {:else if job.status === "queued"}
      <div class="placeholder">Queued — run OCR to extract text.</div>
    {:else if displayText.trim()}
      <pre class="text">{displayText}</pre>
    {:else}
      <div class="placeholder">No text recognized. Try a different engine or image.</div>
    {/if}
  </div>
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
  }
  .title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin-right: auto;
  }
  .meta {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text-faint);
  }
  .copy {
    font-size: 11px;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 3px 9px;
  }
  .copy:hover:not(:disabled) { border-color: var(--accent-dim); }
  .copy:disabled { opacity: 0.4; cursor: not-allowed; }
  .body {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
  }
  .text {
    margin: 0;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .placeholder {
    color: var(--text-faint);
    font-size: 13px;
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .error {
    color: var(--danger);
    background: var(--danger-soft);
    border: 1px solid var(--danger);
    border-radius: 8px;
    padding: 14px;
    font-size: 13px;
  }
  .error pre {
    margin: 8px 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    opacity: 0.85;
  }
  .spin {
    width: 12px;
    height: 12px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    display: inline-block;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
