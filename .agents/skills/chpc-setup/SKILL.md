---
name: chpc-setup
description: Set up or synchronize this AutoResearch repository on the CHPC cluster. Use when asked to clone the repo, configure its Python environment, or prepare CHPC for experiments.
---

Use the configured CHPC values in `.env` without exposing secrets. The intended first-time route is login node → `chpclic1` internet node → `$CHPC_LUSTRE_PATH`.

For first-time setup: clone the configured repository into `$CHPC_LUSTRE_PATH/$CHPC_REPO_NAME`, load the configured Python module, create and activate `.venv`, install `requirements.txt` and the editable package, create `logs/`, and create a private `.env` from `.env.example`.

For sync: pull `develop` from the clone, then refresh dependencies if they changed. Keep CHPC on `develop`; do not check out feature branches there.

For a private repository, recommend a dedicated read-only GitHub deploy key on CHPC. Do not copy a local private key or commit `.env`.
