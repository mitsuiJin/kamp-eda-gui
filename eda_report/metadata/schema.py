"""Dataset Guidebook에서 얻은 데이터셋 메타데이터 스키마.

설계 원칙(C): EDA Module은 변수의 의미를 추론하지 않는다. 변수 타입·target·시간축 같은
"무엇을 분석할지"를 정하는 정보는 Dataset Guidebook에서 받아서 **검증**만 하고, 없으면
dtype 수준의 객관적 사실만으로 동작한다.

역할 분담:
- EDA 실행에 직접 쓰는 정보 → `columns` / `target_columns` / `datetime_column` 등 구조 정보
- 도메인 해석용 정보(공정 설명·장비·수집 배경) → `domain_context` (AI Context로만 전달, EDA는 읽지 않음)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

DeclaredType = Literal["numeric", "categorical", "datetime", "text", "unknown"]

VALID_DECLARED_TYPES: set[str] = {"numeric", "categorical", "datetime", "text", "unknown"}

MetadataSource = Literal["none", "json", "pdf_draft_unverified"]


@dataclass
class ColumnMetadata:
    """Dataset Guidebook이 명시한 컬럼 정보. 확인되지 않은 값은 그대로 unknown/None으로 둔다."""

    name: str
    declared_type: DeclaredType = "unknown"
    description: str | None = None
    unit: str | None = None
    categories: list[str] | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ColumnMetadata:
        declared = str(data.get("declared_type", "unknown")).strip().lower()
        if declared not in VALID_DECLARED_TYPES:
            declared = "unknown"
        return cls(
            name=str(data["name"]),
            declared_type=declared,  # type: ignore[arg-type]
            description=data.get("description"),
            unit=data.get("unit"),
            categories=data.get("categories"),
            notes=data.get("notes"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetMetadata:
    dataset_name: str | None = None
    process: str | None = None
    columns: list[ColumnMetadata] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    datetime_column: str | None = None
    sampling_interval: str | None = None
    collection_period: str | None = None
    domain_context: dict = field(default_factory=dict)
    source: MetadataSource = "none"

    def column(self, name: str) -> ColumnMetadata | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def declared_type(self, name: str) -> str:
        column = self.column(name)
        return column.declared_type if column else "unknown"

    def is_empty(self) -> bool:
        return not self.columns and not self.target_columns and self.datetime_column is None

    @classmethod
    def empty(cls) -> DatasetMetadata:
        return cls(source="none")

    @classmethod
    def from_dict(cls, data: dict) -> DatasetMetadata:
        source = str(data.get("source", "json"))
        if source not in {"none", "json", "pdf_draft_unverified"}:
            source = "json"
        return cls(
            dataset_name=data.get("dataset_name"),
            process=data.get("process"),
            columns=[ColumnMetadata.from_dict(c) for c in data.get("columns", [])],
            target_columns=[str(t) for t in data.get("target_columns", [])],
            datetime_column=data.get("datetime_column"),
            sampling_interval=data.get("sampling_interval"),
            collection_period=data.get("collection_period"),
            domain_context=data.get("domain_context", {}),
            source=source,  # type: ignore[arg-type]
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> DatasetMetadata:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["columns"] = [c.to_dict() for c in self.columns]
        return payload

    def save_json(self, path: str | Path) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(path)
