# Docs Fetch

Fetch CHPC wiki documentation and save as markdown for offline reference.

## URLs to Fetch

| Page | URL | Save As |
|------|-----|---------|
| Survival Guide | https://wiki.chpc.ac.za/survival_guide | `docs/chpc/survival_guide.md` |
| Quick Start | https://wiki.chpc.ac.za/quick:start | `docs/chpc/quick_start.md` |
| GPU Guide | https://wiki.chpc.ac.za/guide:gpu | `docs/chpc/gpu_guide.md` |
| Python Guide | https://wiki.chpc.ac.za/guide:python | `docs/chpc/python_guide.md` |

## Steps

1. Fetch each URL
2. Convert HTML content to clean markdown
3. Save to `docs/chpc/` directory
4. Remove navigation elements, sidebars, footers -- keep only main content
5. Add a header with source URL and fetch date

## Notes

- These docs provide critical reference for PBS queue options, GPU allocations, module loading, and Python setup
- They should be refreshed periodically as CHPC updates their systems
