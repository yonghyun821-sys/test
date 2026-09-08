from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json
from taxonomy_experiment.token_count import TokenCounter


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class MissingAPIKeyError(RuntimeError):
    pass


class BudgetExceededError(RuntimeError):
    pass


class TruncatedResponseError(RuntimeError):
    pass


class CachedLLM:
    def __init__(self, config: dict[str, Any], cache_dir: Path):
        api = config["api"]
        key_name = api["api_key_env"]
        api_key = os.getenv(key_name)
        if not api_key:
            raise MissingAPIKeyError(
                f"{key_name} is not set. Dataset inspection and offline tests can run without it; live merge, attribution, and evaluation cannot."
            )
        base_url = str(api["base_url"])
        default_headers: dict[str, str] = {}
        if api.get("app_title"):
            default_headers["X-OpenRouter-Title"] = str(api["app_title"])
        referer_env = api.get("http_referer_env")
        if referer_env and os.getenv(referer_env):
            default_headers["HTTP-Referer"] = os.environ[referer_env]
        client_args: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": float(api["timeout_seconds"]),
            "max_retries": 0,
        }
        if default_headers:
            client_args["default_headers"] = default_headers
        self.client = OpenAI(**client_args)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = int(api["max_retries"])
        self.base_url = base_url
        self.provider_defaults = dict(api.get("provider_defaults", {}))
        self.model_configs = tuple(config["models"].values())
        budget = config["cost_budget"]
        self.inference_cost_cap = float(budget["max_inference_usd"])
        self.ledger_baseline_usd = float(budget.get("ledger_baseline_usd", 0.0))
        self.input_token_safety_multiplier = float(budget["input_token_safety_multiplier"])
        ledger_path = Path(budget["ledger_path"])
        self.cost_ledger_path = (
            ledger_path if ledger_path.is_absolute() else config["_root"] / ledger_path
        )

    @staticmethod
    def _schema(model: type[BaseModel]) -> dict[str, Any]:
        return model.model_json_schema()

    @classmethod
    def _response_format(cls, model: type[BaseModel]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "strict": True,
                "schema": cls._schema(model),
            },
        }

    def _model_config(self, model: str) -> dict[str, Any]:
        matches = [item for item in self.model_configs if item["name"] == model]
        if not matches:
            raise ValueError(f"No configured pricing entry for model {model!r}")
        pricing_variants = {
            json.dumps(item["pricing_usd_per_million"], sort_keys=True)
            for item in matches
        }
        if len(pricing_variants) != 1:
            raise ValueError(f"Conflicting configured pricing entries for model {model!r}")
        return matches[0]

    def _ledger_spend(self) -> float:
        if not self.cost_ledger_path.exists():
            return 0.0
        lifetime_spend = sum(
            float(row.get("cost_usd", 0.0)) for row in iter_jsonl(self.cost_ledger_path)
        )
        return max(0.0, lifetime_spend - self.ledger_baseline_usd)

    def _reserve_cost(
        self, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> tuple[float, int]:
        model_config = self._model_config(model)
        prices = model_config["pricing_usd_per_million"]
        counter = TokenCounter.for_model(model, model_config.get("tokenizer_encoding"))
        raw_input_tokens = counter.count(system_prompt + "\n" + user_prompt)
        guarded_input_tokens = int(
            raw_input_tokens * self.input_token_safety_multiplier + 0.999999
        )
        reserve = (
            guarded_input_tokens * float(prices["input"])
            + max_output_tokens * float(prices["output"])
        ) / 1_000_000
        return reserve, guarded_input_tokens

    def _check_runtime_budget(
        self, model: str, system_prompt: str, user_prompt: str, max_output_tokens: int
    ) -> tuple[float, int, float]:
        reserve, guarded_input_tokens = self._reserve_cost(
            model, system_prompt, user_prompt, max_output_tokens
        )
        spent = self._ledger_spend()
        if spent + reserve > self.inference_cost_cap:
            raise BudgetExceededError(
                f"OpenRouter hard budget would be exceeded: spent=${spent:.6f}, "
                f"next-request reserve=${reserve:.6f}, cap=${self.inference_cost_cap:.2f}"
            )
        return reserve, guarded_input_tokens, spent

    def _record_billed_response(
        self,
        *,
        raw_response: dict[str, Any],
        cache_key: str,
        attempt: int,
        model: str,
        reserve_usd: float,
        guarded_input_tokens: int,
        max_output_tokens: int,
    ) -> float:
        usage = raw_response.get("usage") or {}
        reported_cost = usage.get("cost")
        if reported_cost is None:
            model_config = self._model_config(model)
            prices = model_config["pricing_usd_per_million"]
            prompt_tokens = int(usage.get("prompt_tokens") or guarded_input_tokens)
            completion_tokens = int(
                usage.get("completion_tokens") or max_output_tokens
            )
            reported_cost = (
                prompt_tokens * float(prices["input"])
                + completion_tokens * float(prices["output"])
            ) / 1_000_000
        cost = float(reported_cost) if reported_cost is not None else reserve_usd
        append_jsonl(
            self.cost_ledger_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cache_key": cache_key,
                "attempt": attempt,
                "model": model,
                "response_id": raw_response.get("id"),
                "provider": raw_response.get("provider"),
                "finish_reason": (
                    (raw_response.get("choices") or [{}])[0].get("finish_reason")
                ),
                "cost_usd": cost,
                "reported_usage": usage,
            },
        )
        return cost

    def fingerprint(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        reasoning_effort: str | None,
        seed: int,
        provider: dict[str, Any] | None,
        max_output_tokens: int,
        response_model: type[BaseModel],
    ) -> tuple[str, dict[str, Any]]:
        resolved_provider = {**self.provider_defaults, **(provider or {})}
        max_tokens_parameter = str(
            self._model_config(model).get("max_tokens_parameter", "max_tokens")
        )
        if max_tokens_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                f"Unsupported max token parameter: {max_tokens_parameter!r}"
            )
        request = {
            "api": "openrouter_chat_completions",
            "base_url": self.base_url,
            "provider": resolved_provider,
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "reasoning": {"effort": reasoning_effort} if reasoning_effort else None,
            "seed": seed,
            max_tokens_parameter: max_output_tokens,
            "response_format": self._response_format(response_model),
        }
        payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), request

    def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        reasoning_effort: str | None,
        seed: int,
        provider: dict[str, Any] | None,
        max_output_tokens: int,
        response_model: type[ResponseModel],
    ) -> tuple[ResponseModel, dict[str, Any]]:
        key, request = self.fingerprint(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            seed=seed,
            provider=provider,
            max_output_tokens=max_output_tokens,
            response_model=response_model,
        )
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            parsed = response_model.model_validate(cached["parsed_response"])
            return parsed, {
                "cache_key": key,
                "cache_path": str(cache_path),
                "cache_hit": True,
                "raw_output_text": cached.get("raw_output_text", ""),
                "response_metadata": cached.get("response_metadata", {}),
            }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                reserve_usd, guarded_input_tokens, _ = self._check_runtime_budget(
                    model, system_prompt, user_prompt, max_output_tokens
                )
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": request["messages"],
                    "seed": seed,
                    "response_format": request["response_format"],
                    "extra_body": {"provider": request["provider"]},
                }
                max_tokens_parameter = str(
                    self._model_config(model).get(
                        "max_tokens_parameter", "max_tokens"
                    )
                )
                kwargs[max_tokens_parameter] = max_output_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if reasoning_effort is not None:
                    kwargs["extra_body"]["reasoning"] = {"effort": reasoning_effort}
                response = self.client.chat.completions.create(**kwargs)
                raw_dump = response.model_dump(mode="json")
                billed_cost = self._record_billed_response(
                    raw_response=raw_dump,
                    cache_key=key,
                    attempt=attempt,
                    model=model,
                    reserve_usd=reserve_usd,
                    guarded_input_tokens=guarded_input_tokens,
                    max_output_tokens=max_output_tokens,
                )
                if not response.choices:
                    raise ValueError("OpenRouter response contained no choices")
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    write_json(
                        cache_path.with_suffix(".truncated.json"),
                        {
                            "cache_key": key,
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                            "model": response.model,
                            "provider": raw_dump.get("provider"),
                            "finish_reason": choice.finish_reason,
                            "max_output_tokens": max_output_tokens,
                            "raw_output_text": choice.message.content,
                            "usage": (
                                response.usage.model_dump(mode="json")
                                if response.usage
                                else None
                            ),
                        },
                    )
                    raise TruncatedResponseError(
                        f"OpenRouter response reached max_tokens={max_output_tokens}; "
                        "increase the limit or tighten the prompt before retrying."
                    )
                refusal = getattr(choice.message, "refusal", None)
                if refusal:
                    raise ValueError(f"OpenRouter model refused the request: {refusal}")
                raw_output_text = choice.message.content
                if not isinstance(raw_output_text, str) or not raw_output_text.strip():
                    raise ValueError("OpenRouter response contained no JSON text")
                parsed = response_model.model_validate_json(raw_output_text)
                response_metadata = {
                    "id": response.id,
                    "model": response.model,
                    "provider": raw_dump.get("provider"),
                    "finish_reason": choice.finish_reason,
                    "usage": response.usage.model_dump(mode="json") if response.usage else None,
                    "cost_usd": billed_cost,
                    "created": getattr(response, "created", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                }
                write_json(
                    cache_path,
                    {
                        "cache_key": key,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "request": request,
                        "parsed_response": parsed.model_dump(mode="json"),
                        "raw_output_text": raw_output_text,
                        "response_metadata": response_metadata,
                        "raw_response": raw_dump,
                    },
                )
                return parsed, {
                    "cache_key": key,
                    "cache_path": str(cache_path),
                    "cache_hit": False,
                    "raw_output_text": raw_output_text,
                    "response_metadata": response_metadata,
                }
            except (BudgetExceededError, TruncatedResponseError):
                raise
            except Exception as exc:  # API, transport, refusal, and validation failures share retry policy.
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}") from last_error
