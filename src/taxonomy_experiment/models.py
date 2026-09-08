from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictedError(StrictModel):
    error_type: str = Field(min_length=1, max_length=160)
    explanation: str = Field(min_length=1)
    evidence: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1, max_length=2)
    failure_step: int | None

    @field_validator("failure_step", mode="before")
    @classmethod
    def normalize_failure_step(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "n/a", "na", "none", "null", "unknown"}:
                return None
            if normalized.lstrip("-").isdigit():
                return int(normalized)
        return value


class AttributionResult(StrictModel):
    failure_present: bool
    predicted_errors: list[PredictedError] = Field(min_length=1, max_length=1)
    overall_root_cause: str = Field(min_length=1)


class CompactPredictedError(StrictModel):
    error_type: str = Field(min_length=1, max_length=160)
    explanation: str = Field(min_length=1, max_length=800)
    evidence: list[Annotated[str, Field(min_length=1, max_length=240)]] = Field(
        min_length=1, max_length=2
    )
    failure_step: int | None

    @field_validator("failure_step", mode="before")
    @classmethod
    def normalize_failure_step(cls, value: Any) -> Any:
        return PredictedError.normalize_failure_step(value)


class CompactAttributionResult(StrictModel):
    """Explicit bounded schema used only after two length-truncated responses."""

    failure_present: bool
    predicted_errors: list[CompactPredictedError] = Field(min_length=1, max_length=1)
    overall_root_cause: str = Field(min_length=1, max_length=500)


class SemanticEvaluation(StrictModel):
    correct: bool
    score: int = Field(ge=1, le=5)
    reason: str

    @model_validator(mode="after")
    def correctness_matches_score(self) -> "SemanticEvaluation":
        expected = self.score >= 4
        if self.correct != expected:
            raise ValueError("correct must be true exactly when score is 4 or 5")
        return self


class BinaryEvaluation(StrictModel):
    correct: bool
    reason: str = Field(min_length=1)


class NativeRelabelResult(StrictModel):
    native_label: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=500)


class NativeRelabelDraft(StrictModel):
    native_label: str = Field(min_length=1, max_length=160)
    alternative_labels: list[Annotated[str, Field(min_length=1, max_length=160)]] = (
        Field(min_length=2, max_length=2)
    )
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def all_labels_are_distinct(self) -> "NativeRelabelDraft":
        labels = [self.native_label, *self.alternative_labels]
        if len({label.strip().casefold() for label in labels}) != 3:
            raise ValueError("native_label and alternative_labels must be distinct")
        return self


class NativeRelabelVerification(StrictModel):
    choice: Literal["A", "B", "C"]
    reason: str = Field(min_length=1, max_length=500)


class NativeRelabelDraftLongReason(NativeRelabelDraft):
    """Compatibility schema for providers that do not enforce maxLength."""

    reason: str = Field(min_length=1, max_length=4000)


class NativeRelabelVerificationLongReason(NativeRelabelVerification):
    """Compatibility schema for providers that do not enforce maxLength."""

    reason: str = Field(min_length=1, max_length=4000)


class CandidateAdjudicationResult(StrictModel):
    choice: Literal["A", "B"]
    reason: str = Field(min_length=1, max_length=500)


class CandidateAssessment(StrictModel):
    candidate_id: Literal["A", "B", "C", "D"]
    mechanism_match: Literal["full", "substantial", "partial", "none"]
    trajectory_support: Literal["strong", "some", "none", "contradicted"]
    symptom_only: bool
    conflicting_primary_cause: bool
    reason: str = Field(min_length=1)


class ComparativeEvaluation(StrictModel):
    reference_consistency: Literal["consistent", "uncertain", "conflicting"]
    reference_consistency_reason: str = Field(min_length=1)
    candidates: list[CandidateAssessment] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def all_candidates_exactly_once(self) -> "ComparativeEvaluation":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if set(candidate_ids) != {"A", "B", "C", "D"} or len(set(candidate_ids)) != 4:
            raise ValueError("candidates must contain A, B, C, and D exactly once")
        return self


class Provenance(StrictModel):
    taxonomy_id: str
    original_category_id: str


class MergedCategory(StrictModel):
    id: str
    name: str
    definition: str
    provenance: list[Provenance]


class MergedTaxonomyResult(StrictModel):
    name: str
    description: str
    categories: list[MergedCategory]


def trajectory_for_prompt(record: dict[str, Any]) -> str:
    safe = {
        # Storage/evaluation IDs may be strengthened without invalidating a frozen
        # prompt cache. The prompt ID is provenance only and is never used as a key.
        "trajectory_id": record.get("prompt_trajectory_id", record["trajectory_id"]),
        "domain": record["domain"],
        "instruction": record.get("instruction", ""),
        "context": record.get("context", {}),
        "steps": record["steps"],
    }
    import json

    return json.dumps(safe, ensure_ascii=False, indent=2)
