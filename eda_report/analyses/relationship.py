"""09. Feature Relationship — 선택된 변수 쌍의 산점도를 그린다.

모든 변수 조합을 그리면 지면이 폭발하므로 일부만 그리고, 어떤 기준으로 골랐는지를
파라미터와 설명에 기록한다(설계 원칙 §9).
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, downsample
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

GRID_SIZE = 2


def _ranked_pairs(corr: pd.DataFrame) -> list[tuple[str, str, float]]:
    cols = list(corr.columns)
    pairs = [
        (cols[i], cols[j], float(corr.iloc[i, j]))
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
        if pd.notna(corr.iloc[i, j])
    ]
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = profile.numeric_columns
    max_pairs = thresholds.scatter_max_pairs

    pearson = df[columns].corr(method="pearson")
    spearman = df[columns].corr(method="spearman")

    linear = _ranked_pairs(pearson)[: max(1, max_pairs // 2)]
    linear_keys = {(a, b) for a, b, _ in linear}
    monotonic = [
        (a, b, value)
        for a, b, value in _ranked_pairs(spearman)
        if (a, b) not in linear_keys
    ][: max(1, max_pairs // 2)]

    selected = [(a, b, value, "Pearson 상위") for a, b, value in linear]
    selected += [(a, b, value, "Spearman 상위(Pearson 상위 제외)") for a, b, value in monotonic]

    fig_dir = params["fig_dir"]
    sample = downsample(df[columns], max_points=3000)
    figures = []
    for i in range(0, len(selected), GRID_SIZE):
        chunk = selected[i:i + GRID_SIZE]
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
        axes = [axes] if len(chunk) == 1 else list(axes)
        for ax, (a, b, value, kind) in zip(axes, chunk):
            color = "#4C72B0" if kind.startswith("Pearson") else "#C44E52"
            ax.scatter(sample[a], sample[b], s=6, alpha=0.4, color=color)
            ax.set_xlabel(a, fontsize=8)
            ax.set_ylabel(b, fontsize=8)
            ax.set_title(f"{kind} ({value:+.2f})", fontsize=9)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"relationship_{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="scatter", image_path=path))

    return AnalysisResult(
        section_id="09_relationship",
        title="Feature Relationship (변수 쌍 산점도)",
        purpose="상관계수만으로는 보이지 않는 두 변수의 실제 분포 형태를 산점도로 확인합니다.",
        rationale=(
            f"전체 변수 조합 대신 {len(selected)}쌍만 표시했습니다. 선택 기준: Pearson 상관계수 절대값 "
            f"상위 {len(linear)}쌍(파란색)과, 여기에 포함되지 않은 것 중 Spearman 상관계수 절대값 상위 "
            f"{len(monotonic)}쌍(빨간색)입니다. Spearman만 높은 쌍은 직선이 아닌 형태로 함께 움직일 "
            "수 있습니다. 점 하나가 관측치 한 건이며, 표시 속도를 위해 최대 3,000건만 표본으로 그렸습니다."
        ),
        input_columns=columns,
        parameters={
            "selection_rule": "Pearson 상위 + (중복 제외) Spearman 상위",
            "max_pairs": max_pairs,
            "selected_pairs": [[a, b, kind] for a, b, _, kind in selected],
            "plot_sample_size": int(len(sample)),
        },
        figures=figures,
        ai_context={
            "pearson_top_pairs": linear,
            "spearman_top_pairs": monotonic,
        },
    )
