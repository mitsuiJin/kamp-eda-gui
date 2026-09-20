"""05. Numerical Distribution — 수치형 변수의 분포 형태를 히스토그램으로 확인한다.

이상치 판정(박스플롯)은 07번에서 다루므로 여기서는 중복하지 않는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, with_description
from eda_report.profiling.dataset_profile import DatasetProfile

GRID_SIZE = 2


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    columns = profile.numeric_columns
    fig_dir = params["fig_dir"]
    column_glossary: dict[str, str] = params.get("column_glossary") or {}

    figures = []
    for i in range(0, len(columns), GRID_SIZE):
        chunk = columns[i:i + GRID_SIZE]
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
        axes = [axes] if len(chunk) == 1 else list(axes)
        for ax, col in zip(axes, chunk):
            ax.hist(df[col].dropna(), bins=30, color="#4C72B0")
            ax.set_title(with_description(col, col, column_glossary), fontsize=9)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("빈도", fontsize=8)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"numeric_hist_{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="hist", image_path=path, caption=", ".join(chunk)))

    return AnalysisResult(
        section_id="05_numerical_distribution",
        title="Numerical Distribution (수치형 변수 분포)",
        purpose="히스토그램으로 각 수치형 변수의 값이 어떤 구간에 몰려 있는지 확인합니다. 봉우리가 여러 개이거나 특정 값에 치우친 형태를 볼 수 있습니다.",
        rationale="변수마다 단위가 달라 축 범위를 서로 비교하지 마세요. 이상치 판정은 07번을 참고하세요.",
        input_columns=columns,
        parameters={"bins": 30, "charts_per_row": GRID_SIZE},
        figures=figures,
        ai_context={"columns": columns},
    )
