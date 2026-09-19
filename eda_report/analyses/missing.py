"""03. Missing Values — 결측치 크기와 패턴(특히 공동결측)을 확인한다."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from eda_report.analyses.base import AnalysisResult, Finding, round_floats
from eda_report.profiling.dataset_profile import DatasetProfile


def _find_co_missing_groups(df: pd.DataFrame, missing_columns: list[str]) -> list[tuple[list[str], float]]:
    """결측 개수가 같은 컬럼끼리만 마스크 일치 여부를 확인한다(전체 O(n^2) 비교를 피하기 위함)."""
    by_count: dict[int, list[str]] = defaultdict(list)
    for col in missing_columns:
        by_count[int(df[col].isna().sum())].append(col)

    result: list[tuple[list[str], float]] = []
    for cols in by_count.values():
        if len(cols) < 2:
            continue
        masks = {c: df[c].isna() for c in cols}
        visited: set[str] = set()
        for i, c1 in enumerate(cols):
            if c1 in visited:
                continue
            group = [c1]
            for c2 in cols[i + 1:]:
                if c2 not in visited and masks[c1].equals(masks[c2]):
                    group.append(c2)
                    visited.add(c2)
            if len(group) >= 2:
                result.append((group, float(masks[c1].mean())))
    return result


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    missing_rate = df.isna().mean().sort_values(ascending=False)
    missing_rate = missing_rate[missing_rate > 0]

    co_missing_groups = _find_co_missing_groups(df, list(missing_rate.index))
    findings = [
        Finding(
            flag_type="conditional_missing",
            columns=cols,
            metric=rate,
            severity="info",
            message_ko=f"{', '.join(cols)}: 결측 위치가 완전히 동일함(결측률 {rate:.1%})",
        )
        for cols, rate in co_missing_groups
    ]

    table = missing_rate.rename("missing_rate").reset_index().rename(columns={"index": "column"})

    return AnalysisResult(
        section_id="03_missing",
        title="Missing Values (결측치)",
        purpose=(
            "변수별 결측 비율과, 여러 변수가 같은 행에서 함께 결측되는 패턴이 있는지 확인합니다. "
            "결측이 발생한 원인은 데이터만으로 확정할 수 없으므로 관찰된 패턴만 제시합니다."
        ),
        rationale="결측 위치가 완전히 동일한 컬럼 그룹은 수집 단계가 같을 가능성을 시사하지만, 원인은 Dataset Guidebook이나 현장 확인이 필요합니다.",
        input_columns=list(missing_rate.index),
        tables=[round_floats(table)],
        findings=findings,
        ai_context={
            "missing_rate": missing_rate.to_dict(),
            "co_missing_groups": [{"columns": c, "rate": r} for c, r in co_missing_groups],
        },
    )
