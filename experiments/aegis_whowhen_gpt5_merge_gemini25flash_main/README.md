# GPT-5 merge and Gemini 2.5 Flash main experiment

This is isolated from the completed Flash-Lite experiment.

Add to the repository `.env`:

```dotenv
AEGIS_WHOWHEN_GPT5_FLASH_MAIN_OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
```

Offline preflight:

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py --preflight-only
```

Run the complete resumable pipeline:

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py
```

The command first creates a GPT-5 merge, then runs 2,388 Gemini 2.5 Flash attribution rows. The provisional reserve is about $10.62 and the hard cap is $50.
