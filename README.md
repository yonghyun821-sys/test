# finaltest experiment bundle

This directory is a copy of the completed **GPT-5 taxonomy merge + Gemini 2.5 Flash main attribution** experiment and the local files required to inspect or reproduce it.

## Primary experiment

- `experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/`
  - experiment code and configuration
  - GPT-5 merged-taxonomy artifacts
  - Gemini 2.5 Flash prediction cache and API ledgers
  - completed run outputs and final reports

Completed run:

`experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/runs/main-372488a304ede3bc--sample-b7d4cd5fad781ef7/`

The main report is `MAIN_RESULTS.md` in that directory.

## Copied dependencies

- `experiments/aegis_whowhen_final_main_attribution/`
- `experiments/aegis_whowhen_main_fixed_guidance/`
  - frozen 597-trajectory sample, gold data, taxonomies, prompts, and identity audits
- `experiments/aegis_whowhen_pilot_fixed_guidance/`
  - merge implementation and prompts used by the primary experiment
- `src/taxonomy_experiment/`
- `pyproject.toml`

## Secret handling

The real repository `.env` was intentionally **not copied**. Only the experiment's `.env.example` is included. If this copy is used independently, create `finaltest/.env` and add:

```dotenv
AEGIS_WHOWHEN_GPT5_FLASH_MAIN_OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
```

## 다른 위치에서 실험 실행하기

아래 명령은 Windows PowerShell 또는 명령 프롬프트에서 실행한다. Python 3.11 이상과 유효한 OpenRouter API 키가 필요하다.

### 1. 폴더 이동 및 가상환경 준비

`finaltest` 폴더를 원하는 위치로 복사한 후, 해당 폴더를 작업 디렉터리로 연다.

```powershell
cd C:\원하는\경로\finaltest
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

설치되는 주요 의존성은 `openai`, `pydantic`, `PyYAML`, `tiktoken`이며 정확한 목록은 `pyproject.toml`에 있다.

### 2. OpenRouter API 키 설정

프로젝트 루트, 즉 이 README와 같은 위치에 `.env`를 만든다.

```powershell
Copy-Item experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\.env.example .env
notepad .env
```

`.env` 내용은 다음과 같아야 한다.

```dotenv
AEGIS_WHOWHEN_GPT5_FLASH_MAIN_OPENROUTER_API_KEY=sk-or-v1-실제_API_KEY
```

필요한 경우 다음 항목도 추가할 수 있다.

```dotenv
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://example.com
```

`.env`는 외부 공유나 Git 커밋에서 제외한다.

### 3. 오프라인 사전 검사

이 단계는 API를 호출하지 않는다. 고정 표본, 코드, taxonomy, prompt 해시와 비용 사전 계산을 점검한다.

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py --preflight-only
```

오류 없이 사전 검사 결과가 출력되어야 한다. API 호출 전에 반드시 이 단계를 먼저 실행한다.

### 4. 기존 완료 결과 확인 또는 재분석

이 묶음에는 이미 완료된 결과와 캐시가 포함되어 있다. 따라서 기본 명령을 실행하면 기존 진행 상태를 재사용하며, 2,388개 예측을 처음부터 다시 호출하지 않는다.

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py
```

API를 호출하지 않고 기존 결과만 다시 분석하려면 다음을 실행한다.

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py --analyze-only
```

### 5. API를 처음부터 다시 호출하는 신규 재현 실험

완료 결과 보존을 위해 원본 `finaltest`는 보관하고, 폴더 전체를 별도의 작업용 사본으로 복사한 뒤 진행하는 것을 권장한다.

작업용 사본에서 `experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/experiment.yaml`을 `experiment.rerun.yaml`로 복사한다.

```powershell
Copy-Item experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\experiment.yaml experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\experiment.rerun.yaml
notepad experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\experiment.rerun.yaml
```

`experiment.rerun.yaml`에서 아래 출력 경로 다섯 개를 새로운 경로로 변경한다. 입력 표본이나 taxonomy 경로는 변경하지 않는다.

```yaml
cost_budget:
  ledger_path: experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/rerun_runtime/merge_api_ledger.jsonl

paths:
  artifacts: experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/rerun_artifacts/merge
  cache: experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/rerun_runtime/cache
  _merge_ledger: experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/rerun_runtime/merge_api_ledger.jsonl
  run_parent: experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/rerun_runs
```

`cost_budget.ledger_path`와 `paths._merge_ledger`는 반드시 같은 파일을 가리켜야 한다. 기존 `artifacts`, `runtime`, `runs` 폴더를 삭제할 필요는 없다.

먼저 신규 설정을 오프라인 검사한다.

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py --config experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\experiment.rerun.yaml --preflight-only
```

검사가 통과하면 전체 실험을 시작한다.

```powershell
.\.venv\Scripts\python.exe -u experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\run.py --config experiments\aegis_whowhen_gpt5_merge_gemini25flash_main\experiment.rerun.yaml
```

실행 순서는 다음과 같다.

1. GPT-5가 AEGIS와 Who&When taxonomy를 병합한다.
2. 병합 결과를 결정론적으로 검증하고 고정한다.
3. Gemini 2.5 Flash가 597개 trajectory에 네 조건을 적용하여 총 2,388개 attribution을 수행한다.
4. exact native-label accuracy, paired comparison, bootstrap confidence interval과 비용 장부를 생성한다.
5. 중단 후 같은 명령을 다시 실행하면 생성된 행까지 이어서 진행한다.

설정상의 전체 비용 상한은 `$50`이다. 복사된 원 실행의 기록 비용은 약 `$5.37`이었지만, 신규 실행의 실제 비용은 응답 길이와 OpenRouter 가격에 따라 달라질 수 있다.

### 6. 신규 결과 확인

신규 실행 결과는 다음 구조로 저장된다.

```text
experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/rerun_runs/
└── main-<implementation-id>--sample-b7d4cd5fad781ef7/
    ├── MAIN_RESULTS.md
    ├── main_accuracy.csv
    ├── main_predictions.jsonl
    ├── main_run_status.json
    ├── pipeline_cost_summary.json
    ├── merged_taxonomy.json
    └── merge_audit.json
```

가장 먼저 `main_run_status.json`의 `status`가 `FINAL_MAIN_EXPERIMENT_COMPLETE`인지 확인한 뒤 `MAIN_RESULTS.md`를 읽는다. 비용은 `pipeline_cost_summary.json`에서 확인한다.

## Verification

- The primary experiment directory was copied byte-for-byte: 2,435 files, with no missing or hash-mismatched files.
- The frozen implementation identity check passes for all code, prompts, taxonomies, configuration, sampled inputs, and gold data.
- Original repository files were not moved or deleted.
