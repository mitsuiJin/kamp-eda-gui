"""02. Column Profile — 컬럼별 관찰 사실(dtype, 결측률, 고유값 수, 역할)을 보여준다.

역할(role)은 추론이 아니라 (a) target으로 지정된 경우 "target", (b) 그 외엔 pandas dtype
기준으로 정해지며, role_source 컬럼에 그 출처가 남는다.

description(선택)은 사람이 검수한 컬럼 설명(`--column-glossary`)이 있을 때만 표시되는
순수 참고 텍스트다 — role/target 판정에는 전혀 관여하지 않는다.
"""

from __future__ import annotations

import pandas as pd

from eda_report.analyses.base import AnalysisResult, Finding
from eda_report.profiling.column_profile import target_kind_label
from eda_report.profiling.dataset_profile import DatasetProfile

_ROLE_SOURCE_LABEL = {
    "dtype": "dtype",
    "data_quality": "데이터품질",
    "target_designation": "target 지정",
}


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    column_glossary: dict[str, str] = params.get("column_glossary") or {}

    rows = []
    findings: list[Finding] = []
    for column in profile.columns:
        row = {
            "column": column.name,
            "role": column.role,
            "target_kind": target_kind_label(column.target_kind) if column.target_kind else "-",
            "role_source": _ROLE_SOURCE_LABEL.get(column.role_source, column.role_source),
            "dtype": column.observed_dtype,
            "missing_rate": round(column.missing_rate, 4),
            "n_unique": column.n_unique,
        }
        if column_glossary:
            row["description"] = column_glossary.get(column.name, "-")
        rows.append(row)

    for name, reason in profile.excluded_columns.items():
        findings.append(
            Finding("excluded_column", [name], None, "info", f"{name}: {reason}이라 이후 분석 대상에서 제외됨")
        )

    table = pd.DataFrame(rows)
    rationale = (
        "role_source는 역할 판단이 target 지정에서 온 것인지 dtype에서 온 것인지를 나타냅니다. "
        "상수·전체결측 컬럼은 이후 분석에서 자동 제외됩니다."
    )
    if column_glossary:
        rationale += " description은 사람이 검수한 컬럼 설명으로 표시 참고용일 뿐, role/target 판정에는 쓰이지 않습니다."

    return AnalysisResult(
        section_id="02_column_profile",
        title="Column Profile (컬럼별 프로파일)",
        purpose="각 변수의 dtype, 결측률, 고유값 수, 분석 역할(role)을 확인합니다.",
        rationale=rationale,
        input_columns=[c.name for c in profile.columns],
        tables=[table],
        findings=findings,
        ai_context={
            "column_profiles": [c.to_dict() for c in profile.columns],
            "excluded_columns": profile.excluded_columns,
            "column_descriptions": column_glossary,
        },
    )
