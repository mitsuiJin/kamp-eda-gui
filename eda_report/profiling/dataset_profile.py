"""컬럼 프로파일을 모아 분석 대상 집합을 만든다.

여기서 만드는 집합(numeric/categorical/datetime)은 "이 데이터로 어떤 분석이 성립하는가"를
판단하는 입력이며, 변수의 도메인 의미와는 무관하다. target은 Dataset Guidebook/사용자 지정
으로만 정해지고, 지정된 target은 role="target"으로 표시되어 설명변수 집합(numeric_columns 등)
에서 자연히 빠진다 — PCA/Clustering 등 feature 입력에도 포함되지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from eda_report.config import AnalysisThresholds
from eda_report.io.manifest import ParseManifest
from eda_report.metadata.schema import DatasetMetadata
from eda_report.metadata.validator import MetadataValidationReport
from eda_report.profiling.column_profile import ColumnProfile, classify_target_kind, profile_column


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    excluded_columns: dict[str, str] = field(default_factory=dict)
    target_columns: list[str] = field(default_factory=list)
    parse_manifest: ParseManifest | None = None
    metadata: DatasetMetadata | None = None
    validation: MetadataValidationReport | None = None

    def column(self, name: str) -> ColumnProfile:
        for profile in self.columns:
            if profile.name == name:
                return profile
        raise KeyError(name)

    @property
    def datetime_column(self) -> str | None:
        return self.datetime_columns[0] if self.datetime_columns else None


def build_dataset_profile(
    df: pd.DataFrame,
    manifest: ParseManifest,
    metadata: DatasetMetadata | None = None,
    validation: MetadataValidationReport | None = None,
    thresholds: AnalysisThresholds | None = None,
    target_columns: list[str] | None = None,
) -> DatasetProfile:
    thresholds = thresholds or AnalysisThresholds()
    metadata = metadata or DatasetMetadata.empty()
    targets = list(target_columns or [])

    profiles = [
        profile_column(
            df[name],
            declared_type=metadata.declared_type(name),
            confirmed_type=validation.confirmed_type(name) if validation else None,
            datetime_parse_min_rate=thresholds.datetime_parse_min_rate,
            categorical_max_cardinality=thresholds.categorical_max_cardinality,
        )
        for name in df.columns
    ]

    # storage dtype(role)과 analysis role을 분리한다: target으로 지정된 컬럼은 저장 형식이
    # numeric/categorical 무엇이었든 role="target"으로 덮어써서, 이후 role 기준 집계(01_overview
    # 등)가 "18 numeric feature + 1 target"처럼 분석 역할 기준으로 일관되게 나오게 한다. 이미
    # empty/constant로 판정된 컬럼은 그 판정이 더 근본적인 데이터 품질 사실이므로 덮어쓰지 않는다.
    is_multilabel = len(targets) > 1
    profiles = [
        replace(
            p,
            role="target",
            role_source="target_designation",
            target_kind=classify_target_kind(df[p.name], is_multilabel_member=is_multilabel),
        )
        if p.name in targets and p.role not in ("empty", "constant")
        else p
        for p in profiles
    ]

    excluded = {
        p.name: ("모든 값이 결측" if p.role == "empty" else "값이 하나뿐인 상수 컬럼")
        for p in profiles
        if p.role in {"empty", "constant"}
    }

    def collect(role: str) -> list[str]:
        return [p.name for p in profiles if p.role == role and p.name not in targets]

    return DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=profiles,
        numeric_columns=collect("numeric"),
        categorical_columns=collect("categorical"),
        datetime_columns=collect("datetime"),
        text_columns=collect("text"),
        excluded_columns=excluded,
        target_columns=targets,
        parse_manifest=manifest,
        metadata=metadata,
        validation=validation,
    )


def exclude_columns(profile: DatasetProfile, exclude: list[str]) -> DatasetProfile:
    """설명변수 집합에서 특정 컬럼(주로 target)을 제거한 프로파일을 반환한다."""
    blocked = set(exclude)
    return replace(
        profile,
        numeric_columns=[c for c in profile.numeric_columns if c not in blocked],
        categorical_columns=[c for c in profile.categorical_columns if c not in blocked],
        datetime_columns=[c for c in profile.datetime_columns if c not in blocked],
        text_columns=[c for c in profile.text_columns if c not in blocked],
    )
