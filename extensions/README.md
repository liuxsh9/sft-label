# extensions/

This folder is the default extension-spec directory used by `uv run sft-label start`.

- Put your own `.yaml` / `.yml` extension specs here if you want the launcher to auto-load them.
- This folder is only the launcher's default place to **look for spec files**. Files here do **not** run automatically; an extension runs only if you enable extension labeling and select that spec for the run.
- The bundled `ui_web_analysis_v1.example.yaml` is an **example** for Web UI dataset analysis / mix optimization.
- For your first custom extension, prefer copying that file to a new filename, changing `id` first, then editing `trigger`, `prompt`, and `schema`.

The launcher scans this folder each time, so updates placed here are picked up automatically.
