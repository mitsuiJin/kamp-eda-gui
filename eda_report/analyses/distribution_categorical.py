"""06. Categorical Distribution — 범주형 변수의 값별 빈도를 확인한다.

범주 수가 표시 한계를 넘으면 Top-N만 그리되, 전체 범주 수와 전체 빈도표는 AI Context에
그대로 기록해 정보가 사라지지 않게 한다(설계 원칙 §9).
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, with_description
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

GRID_SIZE = 2


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = profile.categorical_columns
    fig_dir = params["fig_dir"]
    top_n = thresholds.categorical_display_top_n
    column_glossary: dict[str, str] = params.get("column_glossary") or {}

    figures = []
    value_counts: dict[str, dict] = {}
    truncated: list[str] = []
    for i in range(0, len(columns), GRID_SIZE):
        chunk = columns[i:i + GRID_SIZE]
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
        axes = [axes] if len(chunk) == 1 else list(axes)
        for ax, col in zip(axes, chunk):
            counts = df[col].value_counts(dropna=False).sort_values(ascending=False)
            # 결측(NaN)도 하나의 범주로 보여주되, 축 라벨은 문자열로 정규화한다
            # (NaN을 그대로 matplotlib에 넘기면 카테고리 축에서 오류가 발생한다).
            counts.index = ["(결측)" if pd.isna(idx) else str(idx) for idx in counts.index]
            value_counts[col] = {str(k): int(v) for k, v in counts.items()}
            shown = counts
            title = col
            if len(counts) > top_n:
                shown = counts.head(top_n)
                truncated.append(col)
                title = f"{col} (전체 {len(counts)}개 중 상위 {top_n}개)"
            ax.bar(list(shown.index), shown.values, color="#55A868")
            ax.set_title(with_description(title, col, column_glossary), fontsize=9)
            ax.set_ylabel("빈도", fontsize=8)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"categorical_dist_{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="bar", image_path=path, caption=", ".join(chunk)))

    rationale = "막대 높이는 관측 건수입니다."
    if truncated:
        rationale += f" {', '.join(truncated)}는 상위 {top_n}개만 표시했습니다(전체는 JSON 참고)."

    return AnalysisResult(
        section_id="06_categorical_distribution",
        title="Categorical Distribution (범주형 변수 분포)",
        purpose="막대그래프로 각 범주형 변수의 값별 관측 건수를 확인합니다. 특정 값에 치우쳐 있는지 볼 수 있습니다.",
        rationale=rationale,
        input_columns=columns,
        parameters={"display_top_n": top_n, "truncated_columns": truncated},
        figures=figures,
        ai_context={
            "columns": columns,
            "value_counts": value_counts,
            "total_categories": {c: len(v) for c, v in value_counts.items()},
        },
    )
