"""Dataset Guidebook 메타데이터를 실제 CSV/TXT 데이터와 대조해 검증한다.

핵심 원칙: **검증할 수 없는 정보를 임의로 보정하지 않는다.** 불일치가 발견되면 값을 고치는
대신 상태(type_mismatch / missing_in_data / unverifiable)로 기록하고, EDA는 `confirmed`된
정보만 분석 대상 선정에 사용한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

from eda_report.metadata.schema import DatasetMetadata

ColumnValidationStatus = Literal[
    "confirmed",  # Dataset Guidebook 정보와 실제 데이터가 일치
    "type_mismatch",  # Dataset Guidebook 타입과 실제 데이터가 충돌
    "missing_in_data",  # Dataset Guidebook에 있으나 실제 데이터에 없는 컬럼
    "not_in_metadata",  # 실제 데이터에 있으나 Dataset Guidebook에 없는 컬럼
    "unverifiable",  # Dataset Guidebook이 타입을 명시하지 않아 검증 대상이 아님
]


@dataclass
class ColumnValidation:
    name: str
    status: ColumnValidationStatus
    declared_type: str
    observed_dtype: str
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetadataValidationReport:
    source: str
    columns: list[ColumnValidation] = field(default_factory=list)
    target_status: dict = field(default_factory=dict)
    datetime_status: dict = field(default_factory=dict)

    def confirmed_type(self, name: str) -> str | None:
        """EDA가 신뢰해도 되는(=실제 데이터와 일치가 확인된) 선언 타입만 돌려준다."""
        for column in self.columns:
            if column.name == name and column.status == "confirmed":
                return column.declared_type
        return None

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for column in self.columns:
            counts[column.status] = counts.get(column.status, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "status_counts": self.status_counts(),
            "columns": [c.to_dict() for c in self.columns],
            "target_status": self.target_status,
            "datetime_status": self.datetime_status,
        }


def _numeric_parse_rate(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    return float(pd.to_numeric(non_null, errors="coerce").notna().mean())


def _datetime_parse_rate(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    parsed = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _validate_column(name: str, declared: str, series: pd.Series, datetime_min_rate: float) -> ColumnValidation:
    observed = str(series.dtype)

    if declared == "unknown":
        return ColumnValidation(name, "unverifiable", declared, observed, "Dataset Guidebook에 타입이 명시되지 않음")

    if declared == "numeric":
        if pd.api.types.is_numeric_dtype(series):
            return ColumnValidation(name, "confirmed", declared, observed)
        rate = _numeric_parse_rate(series)
        return ColumnValidation(
            name, "type_mismatch", declared, observed,
            f"numeric으로 선언됐으나 dtype이 {observed}이며 숫자 변환 성공률 {rate:.1%}",
        )

    if declared == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnValidation(name, "confirmed", declared, observed)
        rate = _datetime_parse_rate(series)
        if rate >= datetime_min_rate:
            return ColumnValidation(name, "confirmed", declared, observed, f"문자열이지만 날짜 변환 성공률 {rate:.1%}")
        return ColumnValidation(
            name, "type_mismatch", declared, observed, f"datetime으로 선언됐으나 날짜 변환 성공률 {rate:.1%}"
        )

    if declared == "categorical":
        n_unique = int(series.nunique(dropna=True))
        detail = f"고유값 {n_unique}개 관찰됨"
        if n_unique == len(series.dropna()) and n_unique > 1:
            detail += " (모든 행이 서로 다른 값 — 실제로 범주형인지 확인 필요)"
        return ColumnValidation(name, "confirmed", declared, observed, detail)

    # text
    return ColumnValidation(name, "confirmed", declared, observed)


def validate_metadata(
    metadata: DatasetMetadata, df: pd.DataFrame, datetime_min_rate: float = 0.99
) -> MetadataValidationReport:
    report = MetadataValidationReport(source=metadata.source)
    data_columns = list(df.columns)

    for column in metadata.columns:
        if column.name not in data_columns:
            report.columns.append(
                ColumnValidation(
                    column.name, "missing_in_data", column.declared_type, "-",
                    "Dataset Guidebook에 정의됐으나 실제 데이터에 존재하지 않는 컬럼",
                )
            )
            continue
        report.columns.append(
            _validate_column(column.name, column.declared_type, df[column.name], datetime_min_rate)
        )

    declared_names = {c.name for c in metadata.columns}
    for name in data_columns:
        if name not in declared_names:
            report.columns.append(
                ColumnValidation(name, "not_in_metadata", "unknown", str(df[name].dtype),
                                 "실제 데이터에 있으나 Dataset Guidebook에 정의되지 않은 컬럼")
            )

    if metadata.target_columns:
        present = [t for t in metadata.target_columns if t in data_columns]
        absent = [t for t in metadata.target_columns if t not in data_columns]
        report.target_status = {
            "declared": metadata.target_columns,
            "present_in_data": present,
            "missing_in_data": absent,
            "usable": present,
        }
    else:
        report.target_status = {"declared": [], "present_in_data": [], "missing_in_data": [], "usable": []}

    if metadata.datetime_column:
        if metadata.datetime_column not in data_columns:
            report.datetime_status = {
                "declared": metadata.datetime_column,
                "usable": None,
                "detail": "Dataset Guidebook이 지정한 시간 컬럼이 실제 데이터에 없음",
            }
        else:
            rate = _datetime_parse_rate(df[metadata.datetime_column])
            usable = rate >= datetime_min_rate
            report.datetime_status = {
                "declared": metadata.datetime_column,
                "usable": metadata.datetime_column if usable else None,
                "parse_rate": rate,
                "detail": f"날짜 변환 성공률 {rate:.1%}",
            }
    else:
        report.datetime_status = {"declared": None, "usable": None, "detail": "Dataset Guidebook에 시간 컬럼 정보 없음"}

    return report
