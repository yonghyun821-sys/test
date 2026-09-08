from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCounter:
    model: str
    encoding_name: str
    exact_tokenizer_available: bool
    _encoding: object | None

    @classmethod
    def for_model(cls, model: str, encoding_name: str | None = None) -> "TokenCounter":
        try:
            # On managed Windows hosts this lets requests/tiktoken honor the
            # operating-system trust store when the encoding is first cached.
            try:
                import truststore

                truststore.inject_into_ssl()
            except (ImportError, AttributeError):
                pass
            import tiktoken

            if encoding_name:
                encoding = tiktoken.get_encoding(encoding_name)
                return cls(model, f"{encoding.name} (configured)", False, encoding)
            try:
                encoding = tiktoken.encoding_for_model(model)
                return cls(model, encoding.name, True, encoding)
            except KeyError:
                encoding = tiktoken.get_encoding("o200k_base")
                return cls(model, "o200k_base (model fallback)", False, encoding)
        except Exception:
            return cls(model, "utf8-bytes/4 estimate", False, None)

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))  # type: ignore[attr-defined]
        return math.ceil(len(text.encode("utf-8")) / 4)
