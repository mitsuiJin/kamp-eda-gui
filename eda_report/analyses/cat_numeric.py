"""10. Categorical × Numerical — 범주별로 수치형 변수 값의 분포를 비교한다."""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, round_floats, select_display_columns
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

MAX_CATEGORICAL_COLUMNS = 2
MAX_NUMERIC_COLUMNS = 4
GRID_SIZE = 2


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    # 범주 수가 적은 변수부터 사용한다: 그룹이 적을수록 그룹당 표본이 많아 분포 비교가 안정적이고,
    # 박스플롯 한 장에 담을 수 있기 때문이다(도메인 중요도와 무관한 표시상의 기준).
    cat_columns = sorted(profile.categorical_columns, key=lambda c: df[c].nunique())[:MAX_CATEGORICAL_COLUMNS]
    num_columns = select_display_columns(df, profile.numeric_columns, MAX_NUMERIC_COLUMNS)

    fig_dir = params["fig_dir"]
    figures = []
    summaries = []
    for cat in cat_columns:
        groups_order = list(df[cat].value_counts().index)
        if len(groups_order) > thresholds.categorical_display_top_n:
            groups_order = groups_order[: thresholds.categorical_display_top_n]
        for i in range(0, len(num_columns), GRID_SIZE):
            chunk = num_columns[i:i + GRID_SIZE]
            fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
            axes = [axes] if len(chunk) == 1 else list(axes)
            for ax, num in zip(axes, chunk):
                data = [df.loc[df[cat] == group, num].dropna() for group in groups_order]
                ax.boxplot(data, tick_labels=[str(g) for g in groups_order])
                ax.set_title(f"{num} by {cat}", fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=7)
            fig.tight_layout()
            path = os.path.join(fig_dir, f"cat_numeric_{cat.replace('.', '_')}_{i}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures.append(Figure(kind="boxplot", image_path=path, caption=f"{cat}의 {len(groups_order)}개 그룹 기준"))

        grouped = df.groupby(cat)[num_columns].agg(["count", "mean", "std"])
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        summaries.append(grouped.reset_index())

    return AnalysisResult(
        section_id="10_cat_numeric",
        title="Categorical × Numerical (범주별 수치형 분포 비교)",
        purpose="범주형 변수의 그룹별로 수치형 변수의 분포가 어떻게 관찰되는지 비교합니다.",
        rationale=(
            f"비교 대상 선정 — 범주형: 범주 수가 적은 순으로 {cat_columns}(그룹당 표본이 많아 비교가 "
            f"안정적), 수치형: 표준편차가 큰 순으로 {num_columns}. 그래프 읽는 법: 각 박스는 해당 "
            "그룹의 중간 50% 구간, 박스 안 선은 중앙값입니다. 그룹 간 위치 차이는 관찰된 사실이며, "
            "그 차이의 원인은 여기서 판단하지 않습니다."
        ),
        input_columns=cat_columns + num_columns,
        parameters={
            "categorical_selection": "범주 수 오름차순",
            "numeric_selection": "표준편차 내림차순",
            "categorical_columns": cat_columns,
            "numeric_columns": num_columns,
        },
        figures=figures,
        tables=[round_floats(s) for s in summaries],
        ai_context={
            "group_statistics": [s.to_dict(orient="records") for s in summaries],
        },
    )
