"""02. Column Profile — 컬럼별 관찰 사실과 Dataset Guidebook 검증 결과를 함께 보여준다.

역할(role)은 추론이 아니라 (a) Dataset Guidebook이 명시한 타입이 실제 데이터와 일치가
확인된 경우 그 타입, (b) target으로 지정된 경우 "target", (c) 그 외엔 pandas dtype 기준으로
정해지며, role_source 컬럼에 그 출처가 남는다.
"""

from __future__ import annotations

import pandas as pd

from eda_report.analyses.base import AnalysisResult, Finding
from eda_report.profiling.column_profile import target_kind_label
from eda_report.profiling.dataset_profile import DatasetProfile

_ROLE_SOURCE_LABEL = {
    "metadata": "Dataset Guidebook",
    "dtype": "dtype",
    "data_quality": "데이터품질",
    "target_designation": "target 지정",
}


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    validation = profile.validation
    validation_by_column = (
        {c.name: c for c in validation.columns} if validation else {}
    )

    rows = []
    findings: list[Finding] = []
    for column in profile.columns:
        validated = validation_by_column.get(column.name)
        rows.append(
            {
                "column": column.name,
                "role": column.role,
                "target_kind": target_kind_label(column.target_kind) if column.target_kind else "-",
                "role_source": _ROLE_SOURCE_LABEL.get(column.role_source, column.role_source),
                "dtype": column.observed_dtype,
                "guidebook_type": column.declared_type,
                "metadata_check": validated.status if validated else "-",
                "missing_rate": round(column.missing_rate, 4),
                "n_unique": column.n_unique,
            }
        )

    for name, reason in profile.excluded_columns.items():
        findings.append(
            Finding("excluded_column", [name], None, "info", f"{name}: {reason}이라 이후 분석 대상에서 제외됨")
        )

    if validation:
        for validated in validation.columns:
            if validated.status == "type_mismatch":
                findings.append(
                    Finding(
                        "metadata_type_mismatch", [validated.name], None, "warning",
                        f"{validated.name}: Dataset Guidebook 타입({validated.declared_type})과 실제 데이터가 일치하지 않음 — {validated.detail}",
                    )
                )
            elif validated.status == "missing_in_data":
                findings.append(
                    Finding(
                        "metadata_missing_column", [validated.name], None, "warning",
                        f"{validated.name}: Dataset Guidebook에 정의됐으나 실제 데이터에 존재하지 않음",
                    )
                )

    table = pd.DataFrame(rows)
    counts = validation.status_counts() if validation else {}
    target_rows = [r for r in rows if r["role"] == "target"]
    rationale = (
        f"{len(profile.excluded_columns)}개 컬럼이 상수 또는 전체 결측이라 이후 분석에서 제외됩니다. "
        "role_source는 그 역할 판단이 Dataset Guidebook에서 온 것인지(target 지정 포함) dtype에서 "
        "온 것인지를 나타냅니다."
    )
    if target_rows:
        target_desc = ", ".join(f"{r['column']}({r['target_kind']})" for r in target_rows)
        rationale += f"\n\ntarget으로 지정된 컬럼: {target_desc}. 저장 dtype과 무관하게 분석 role은 target이며, feature 입력에서 제외됩니다."
    if counts:
        rationale += "\n\nDataset Guidebook 검증 결과: " + ", ".join(f"{k} {v}건" for k, v in counts.items())
        rationale += " (불일치는 값을 고치지 않고 상태로만 기록합니다)"

    return AnalysisResult(
        section_id="02_column_profile",
        title="Column Profile (컬럼별 프로파일)",
        purpose=(
            "각 변수의 dtype, 결측률, 고유값 수와 함께 Dataset Guidebook이 선언한 타입이 실제 "
            "데이터와 일치하는지 확인한 결과를 정리합니다."
        ),
        rationale=rationale,
        input_columns=[c.name for c in profile.columns],
        tables=[table],
        findings=findings,
        ai_context={
            "column_profiles": [c.to_dict() for c in profile.columns],
            "excluded_columns": profile.excluded_columns,
            "metadata_validation": validation.to_dict() if validation else None,
        },
    )
