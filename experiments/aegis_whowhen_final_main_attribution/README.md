# Final task-independent main attribution experiment

This directory contains only the execution harness for the frozen prediction implementation `main-95e2d9ee2ba7c1fd` and frozen sample `sample-b7d4cd5fad781ef7`.

It does not modify prompts, taxonomies, schemas, parser, serializer, retry behavior, model, provider request settings, or scoring.

## Setup

Add the following to the repository-root `.env`:

```dotenv
FINAL_MAIN_OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
```

## Offline verification

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_final_main_attribution\run.py --preflight-only
```

## Definitive run

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_final_main_attribution\run.py
```

The command is append-only and resumable. If the terminal closes, run the exact same command again. It never imports prediction outputs from an older namespace.
