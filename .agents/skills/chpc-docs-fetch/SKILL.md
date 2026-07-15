---
name: chpc-docs-fetch
description: Refresh the repository's offline CHPC documentation from official CHPC wiki pages. Use when asked to update local CHPC survival, quick-start, GPU, or Python guides.
---

Fetch only the official CHPC pages mapped below, extract the substantive content as clean Markdown, and retain a source URL and fetch date at the top of each file.

- `https://wiki.chpc.ac.za/survival_guide` → `docs/chpc/survival_guide.md`
- `https://wiki.chpc.ac.za/quick:start` → `docs/chpc/quick_start.md`
- `https://wiki.chpc.ac.za/guide:gpu` → `docs/chpc/gpu_guide.md`
- `https://wiki.chpc.ac.za/guide:python` → `docs/chpc/python_guide.md`

Remove navigation, sidebars, and footers. Preserve queue allocation, module-loading, Python-environment, and storage guidance. Do not overwrite local additions without reviewing the diff.
