import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Test-side config (vitest prefers this over vite.config.ts). Mirrors the
// app config's svelte plugin but forces BROWSER module resolution — under
// vitest's node-ish SSR transform, `svelte` otherwise resolves to its server
// build and client components can't `mount()`. Component tests opt into a
// DOM per-file with `// @vitest-environment jsdom`; the rest stay node.
export default defineConfig({
  plugins: [svelte()],
  resolve: {
    conditions: ["browser"],
  },
});
