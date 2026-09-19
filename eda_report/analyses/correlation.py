"""08. Correlation Analysis — Pearson(선형)과 Spearman(순위) 상관계수를 구분해 산출한다.

상관관계는 인과관계가 아니다. 이 섹션은 "두 변수가 함께 움직이는 정도"라는 관찰 사실만
제시하며, 어느 변수가 다른 변수의 원인인지 판단하지 않는다.

Correlation Network는 heatmap과 정보가 중복되어 PDF에는 넣지 않고, 임계값 이상 상관쌍
목록(edge list)만 AI Context에 기록한다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

from eda_report.analyses.base import AnalysisResult, Figure, Finding, round_floats
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile


def _hierarchical_order(corr: pd.DataFrame) -> list[str]:
    if corr.shape[0] < 3:
        return list(corr.columns)
    distance = (1 - corr.abs()).values
    condensed = np.nan_to_num(distance[np.triu_indices_from(distance, k=1)], nan=1.0)
    return [corr.columns[i] for i in leaves_list(linkage(condensed, method="average"))]


def _pairs(corr: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = (
        corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "column_a", "level_1": "column_b"})
    )
    frame["method"] = method
    frame["abs_correlation"] = frame["correlation"].abs()
    return frame


def _heatmap(corr: pd.DataFrame, order: list[str], title: str, path: str) -> None:
    ordered = corr.loc[order, order]
    size = max(5.5, 0.32 * len(order))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(ordered.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = profile.numeric_columns
    fig_dir = params["fig_dir"]

    pearson = df[columns].corr(method="pearson")
    spearman = df[columns].corr(method="spearman")
    order = _hierarchical_order(pearson)

    pearson_path = os.path.join(fig_dir, "correlation_pearson.png")
    spearman_path = os.path.join(fig_dir, "correlation_spearman.png")
    _heatmap(pearson, order, "Pearson 상관계수 (직선 관계)", pearson_path)
    _heatmap(spearman, order, "Spearman 상관계수 (순위 기반 단조 관계)", spearman_path)

    all_pairs = pd.concat([_pairs(pearson, "pearson"), _pairs(spearman, "spearman")], ignore_index=True)
    top_pearson = (
        all_pairs[all_pairs["method"] == "pearson"]
        .sort_values("abs_correlation", ascending=False)
        .head(thresholds.top_correlation_pairs)
    )
    spearman_lookup = {
        (row["column_a"], row["column_b"]): row["correlation"]
        for _, row in all_pairs[all_pairs["method"] == "spearman"].iterrows()
    }
    top_table = top_pearson[["column_a", "column_b", "correlation"]].rename(columns={"correlation": "pearson"})
    top_table["spearman"] = [
        spearman_lookup.get((a, b), float("nan")) for a, b in zip(top_table["column_a"], top_table["column_b"])
    ]

    high_pairs = all_pairs[all_pairs["abs_correlation"] >= thresholds.high_correlation_threshold]

    findings = [
        Finding(
            flag_type="high_correlation",
            columns=[row["column_a"], row["column_b"]],
            metric=float(row["pearson"]),
            severity="info",
            message_ko=(
                f"{row['column_a']} - {row['column_b']}: Pearson {row['pearson']:.2f}, "
                f"Spearman {row['spearman']:.2f}로 함께 움직이는 정도가 크게 관찰됨"
            ),
        )
        for _, row in top_table.iterrows()
        if abs(row["pearson"]) >= thresholds.high_correlation_threshold
    ]

    return AnalysisResult(
        section_id="08_correlation",
        title="Correlation Analysis (상관관계 분석)",
        purpose=(
            "수치형 변수 쌍이 함께 움직이는 정도를 Pearson(직선 관계)과 Spearman(순위 기반 단조 "
            "관계) 두 가지로 산출합니다. 상관관계는 인과관계를 뜻하지 않습니다."
        ),
        rationale=(
            "Pearson은 두 변수가 직선에 가깝게 함께 변하는 정도(-1~1), Spearman은 값의 크기 대신 "
            "순위만 보고 한 방향으로 함께 변하는 정도를 나타냅니다. 두 값이 크게 다르면 관계가 "
            "직선형이 아닐 수 있습니다. 히트맵은 서로 비슷하게 움직이는 변수끼리 가까이 오도록 "
            "순서를 재배열했고, 칸의 색이 진할수록(빨강=양, 파랑=음) 상관이 강함을 뜻합니다."
        ),
        input_columns=columns,
        parameters={
            "methods": ["pearson", "spearman"],
            "high_correlation_threshold": thresholds.high_correlation_threshold,
            "top_pairs_shown": thresholds.top_correlation_pairs,
        },
        figures=[
            Figure(kind="heatmap", image_path=pearson_path, caption="Pearson (직선 관계 강도)"),
            Figure(kind="heatmap", image_path=spearman_path, caption="Spearman (순위 기반 단조 관계 강도)"),
        ],
        tables=[round_floats(top_table)],
        findings=findings,
        ai_context={
            "pearson_matrix": pearson.round(4).to_dict(),
            "spearman_matrix": spearman.round(4).to_dict(),
            "high_correlation_pairs": high_pairs[["method", "column_a", "column_b", "correlation"]].to_dict(orient="records"),
        },
    )
