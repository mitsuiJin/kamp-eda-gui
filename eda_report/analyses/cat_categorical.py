"""11. Categorical × Categorical — 두 범주형 변수의 값 조합 빈도와 연관 강도를 산출한다."""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from eda_report.analyses.base import AnalysisResult, Figure, round_floats
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

MAX_COLUMNS = 4


def _cramers_v(confusion: pd.DataFrame) -> float:
    if min(confusion.shape) <= 1:
        return 0.0
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.values.sum()
    return float(np.sqrt((chi2 / n) / (min(confusion.shape) - 1)))


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    # 범주 수가 적은 변수부터 사용한다: 교차표 칸 수가 두 변수의 범주 수의 곱으로 늘어나기 때문.
    columns = sorted(profile.categorical_columns, key=lambda c: df[c].nunique())[:MAX_COLUMNS]
    fig_dir = params["fig_dir"]

    figures = []
    scores = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a, b = columns[i], columns[j]
            crosstab = pd.crosstab(df[a], df[b])
            if crosstab.size > thresholds.association_max_combinations:
                continue
            v = _cramers_v(crosstab)
            scores.append({"column_a": a, "column_b": b, "cramers_v": v, "cells": int(crosstab.size)})

            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            image = ax.imshow(crosstab.values, cmap="Blues")
            ax.set_xticks(range(len(crosstab.columns)))
            ax.set_xticklabels(crosstab.columns, rotation=45, fontsize=7)
            ax.set_yticks(range(len(crosstab.index)))
            ax.set_yticklabels(crosstab.index, fontsize=7)
            ax.set_xlabel(b, fontsize=8)
            ax.set_ylabel(a, fontsize=8)
            ax.set_title(f"{a} × {b} (Cramér's V={v:.2f})", fontsize=9)
            fig.colorbar(image, ax=ax, shrink=0.8)
            fig.tight_layout()
            path = os.path.join(fig_dir, f"cat_cat_{i}_{j}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures.append(Figure(kind="heatmap", image_path=path))

    return AnalysisResult(
        section_id="11_cat_categorical",
        title="Categorical × Categorical (범주형 간 교차 분석)",
        purpose="두 범주형 변수의 값 조합이 어떤 빈도로 나타나는지와 연관 강도(Cramér's V)를 산출합니다.",
        rationale=(
            f"범주 수가 적은 변수부터 최대 {MAX_COLUMNS}개를 골라 모든 쌍을 비교했습니다: {columns}. "
            "그래프 읽는 법: 칸의 색이 진할수록 그 값 조합의 관측 건수가 많다는 뜻이고, Cramér's V는 "
            "0(연관 없음)~1(완전 연관) 사이의 연관 강도입니다. 연관 강도는 인과관계를 뜻하지 않습니다."
        ),
        input_columns=columns,
        parameters={"selection": "범주 수 오름차순", "columns": columns},
        figures=figures,
        tables=[round_floats(pd.DataFrame(scores))] if scores else [],
        ai_context={"cramers_v": scores},
    )
