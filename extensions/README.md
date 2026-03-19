# extensions/

This folder is the default extension-spec directory used by `uv run sft-label start`.

- Put your own `.yaml` / `.yml` extension specs here if you want the launcher to auto-load them.
- The bundled `ui_web_analysis_v1.example.yaml` is an **example** for Web UI dataset analysis / mix optimization.
- You can edit that example in place, duplicate it, or keep your own specs beside it.

The launcher scans this folder each time, so updates placed here are picked up automatically.
