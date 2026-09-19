"""05. Numerical Distribution — 수치형 변수의 분포 형태를 히스토그램으로 확인한다.

이상치 판정(박스플롯)은 07번에서 다루므로 여기서는 중복하지 않는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure
from eda_report.profiling.dataset_profile import DatasetProfile

GRID_SIZE = 2


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    columns = profile.numeric_columns
    fig_dir = params["fig_dir"]
    units = {}
    if profile.metadata:
        units = {c.name: c.unit for c in profile.metadata.columns if c.unit}

    figures = []
    for i in range(0, len(columns), GRID_SIZE):
        chunk = columns[i:i + GRID_SIZE]
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
        axes = [axes] if len(chunk) == 1 else list(axes)
        for ax, col in zip(axes, chunk):
            ax.hist(df[col].dropna(), bins=30, color="#4C72B0")
            ax.set_title(col, fontsize=9)
            ax.set_xlabel(f"{col} [{units[col]}]" if col in units else col, fontsize=8)
            ax.set_ylabel("빈도", fontsize=8)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"numeric_hist_{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="hist", image_path=path, caption=", ".join(chunk)))

    return AnalysisResult(
        section_id="05_numerical_distribution",
        title="Numerical Distribution (수치형 변수 분포)",
        purpose=f"수치형 변수 {len(columns)}개 전체의 값이 어떤 구간에 어떻게 분포하는지 확인합니다.",
        rationale=(
            "그래프 읽는 법: 가로축은 변수의 값, 세로축은 그 구간에 속한 관측치 수입니다. 봉우리가 "
            "여러 개이거나 특정 값에 집중된 형태가 관찰될 수 있으며, 그 원인(제어값·운전 모드 등)은 "
            "데이터만으로 확정하지 않습니다. 변수마다 단위가 다르므로 축 범위를 서로 비교하지 "
            "마세요. 이상치 판정 결과는 07번을 참고하세요."
        ),
        input_columns=columns,
        parameters={"bins": 30, "charts_per_row": GRID_SIZE},
        figures=figures,
        ai_context={"columns": columns, "units": units},
    )
