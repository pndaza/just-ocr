<script lang="ts">
  import type { Job } from "./ocr";

  interface Props {
    job: Job | null;
  }
  let { job }: Props = $props();
</script>

<div class="panel" role="region" aria-label="Image preview">
  <div class="head">
    <span class="title">{job ? job.name : "Preview"}</span>
    {#if job}
      <span class="status-pill {job.status}">
        {#if job.status === "queued"}Queued
        {:else if job.status === "running"}Recognizing…
        {:else if job.status === "done"}Done · {job.confidence}% conf
        {:else if job.status === "error"}Error{/if}
      </span>
    {/if}
  </div>
  <div class="stage">
    {#if job}
      <img src={job.url} alt={job.name} />
    {:else}
      <div class="placeholder">
        <div class="ph-icon">▢</div>
        <p>Select an image to preview</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    border-right: 1px solid var(--border);
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
    font-family: var(--mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 60%;
  }
  .status-pill {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--surface);
    color: var(--text-dim);
  }
  .status-pill.done { color: var(--ok); background: var(--ok-soft); }
  .status-pill.running { color: var(--accent); background: var(--accent-soft); }
  .status-pill.error { color: var(--danger); background: var(--danger-soft); }
  .stage {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 20px;
    background: var(--bg-inset);
  }
  .stage img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
  }
  .placeholder {
    color: var(--text-faint);
    text-align: center;
  }
  .ph-icon {
    font-size: 40px;
    color: var(--border-strong);
    margin-bottom: 8px;
  }
</style>
