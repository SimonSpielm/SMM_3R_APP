# 3R Self-Motion Manifold

Minimal marimo app for a single branch.

## Run locally

```bash
pip install -r requirements.txt
marimo run smm_single_branch_marimo.py
```

## Export

```bash
marimo export html-wasm smm_single_branch_marimo.py -o dist --mode run
```

## Deploy

Push the repo to GitHub, enable **Settings → Pages → GitHub Actions**, and push to `main`.
