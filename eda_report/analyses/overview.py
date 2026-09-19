"""01. Dataset Overview — 데이터 규모와 구성에 대한 객관적 사실을 정리한다.

role_counts는 analysis role 기준으로 집계한다(storage dtype 기준이 아니다) — target으로
지정된 컬럼은 저장 형식이 numeric이어도 role="target"으로 표시되므로, "19 numeric + 1
datetime"처럼 target이 섞여 보이는 대신 "18 numeric + 1 target + 1 datetime"으로 나온다.
"""

from __future__ import annotations

import pandas as pd

from eda_report.analyses.base import AnalysisResult
from eda_report.profiling.column_profile import target_kind_label
from eda_report.profiling.dataset_profile import DatasetProfile


def _role_display_label(role: str, target_kind: str | None) -> str:
    if role == "target" and target_kind:
        return f"target ({target_kind_label(target_kind)})"
    return role


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    role_counts: dict[str, int] = {}
    for column in profile.columns:
        label = _role_display_label(column.role, column.target_kind)
        role_counts[label] = role_counts.get(label, 0) + 1

    duplicates = int(df.duplicated().sum())
    metadata = profile.metadata
    overview_table = pd.DataFrame(
        {
            "항목": ["행 수", "열 수", "중복행 수", "메모리 사용량(MB)", "Dataset Guidebook 메타데이터"],
            "값": [
                f"{profile.n_rows:,}",
                f"{profile.n_cols:,}",
                f"{duplicates:,}",
                f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f}",
                {"none": "없음", "json": "JSON으로 제공됨", "pdf_draft_unverified": "PDF 초안(00_manifest에서 검증 결과 확인)"}.get(
                    metadata.source if metadata else "none", "없음"
                ),
            ],
        }
    )
    role_table = (
        pd.DataFrame({"role": list(role_counts), "count": list(role_counts.values())})
        .sort_values("count", ascending=False)
    )

    return AnalysisResult(
        section_id="01_overview",
        title="Dataset Overview (데이터 개요)",
        purpose="데이터의 행/열 규모, 변수 역할(analysis role) 구성, 중복행 수를 확인합니다.",
        rationale=(
            "role은 변수의 도메인 의미가 아니라 분석 대상 선정을 위한 구분입니다(numeric/categorical/"
            "datetime/text/target, 그리고 값이 하나뿐인 constant·전부 결측인 empty). storage dtype이 "
            "아니라 analysis role 기준 집계이므로, target으로 지정된 컬럼은 저장 형식과 무관하게 "
            "target으로 별도 집계됩니다(PCA/Clustering 등 feature 입력에도 포함되지 않습니다)."
        ),
        parameters={"duplicate_rows": duplicates},
        tables=[overview_table, role_table],
        ai_context={
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "duplicate_rows": duplicates,
            "role_counts": role_counts,
            "numeric_columns": profile.numeric_columns,
            "categorical_columns": profile.categorical_columns,
            "datetime_columns": profile.datetime_columns,
            "excluded_columns": profile.excluded_columns,
            "target_columns": profile.target_columns,
        },
    )
