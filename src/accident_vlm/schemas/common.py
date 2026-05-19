from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Status(StrEnum):
    OBSERVED = "observed"
    COMPUTED = "computed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EvidenceField(BaseModel):
    value: Any
    raw: str | None = None
    status: Status
    confidence: Confidence
    source: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    note: str | None = None

    @field_validator("source", "evidence")
    @classmethod
    def reject_blank_items(cls, items: list[str]) -> list[str]:
        stripped_items = []
        for item in items:
            stripped_item = item.strip()
            if not stripped_item:
                raise ValueError("source and evidence items cannot be blank")
            stripped_items.append(stripped_item)
        return stripped_items

    @model_validator(mode="after")
    def require_source_for_non_unknown(self) -> "EvidenceField":
        if self.status != Status.UNKNOWN and not self.source:
            raise ValueError("source is required when status is not unknown")
        return self
