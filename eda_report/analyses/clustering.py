"""13. Clustering — 수치형 변수 공간에서 관측치가 어떻게 묶이는지 관찰한다.

설계 원칙 §3-[4]: Target의 클래스 불균형은 비지도 학습 알고리즘 선택의 근거가 아니다.
이전 구현에는 "target 소수 클래스 비율이 낮아 DBSCAN을 우선 적용" 로직이 있었으나 논리적
근거가 없어 제거했다. 지금은 알고리즘을 자동 전환하지 않고 KMeans 하나로 고정하며,
사용한 알고리즘·파라미터·전처리를 모두 기록한다.

군집 결과는 "이렇게 묶인다"는 관찰이며, 각 군집이 어떤 공정 상태인지는 판단하지 않는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from eda_report.analyses.base import AnalysisResult, Figure, downsample, round_floats
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

TOP_VARS_PER_CLUSTER = 2
SILHOUETTE_SAMPLE_SIZE = 5000


def _pick_k(scaled: np.ndarray, k_min: int, k_max: int) -> tuple[int, dict[int, float]]:
    """실루엣 점수가 가장 높은 k를 고른다(점수는 전부 기록해 근거를 남긴다)."""
    sample = scaled
    if len(scaled) > SILHOUETTE_SAMPLE_SIZE:
        rng = np.random.RandomState(0)
        sample = scaled[rng.choice(len(scaled), SILHOUETTE_SAMPLE_SIZE, replace=False)]

    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(sample)
        scores[k] = float(silhouette_score(sample, labels))
    return max(scores, key=scores.get), scores


def _cap_for_display(profile_table: pd.DataFrame, cols: list[str], max_clusters: int) -> pd.DataFrame:
    if len(profile_table) <= max_clusters:
        return profile_table
    ordered = profile_table.sort_values("n_obs", ascending=False)
    top, rest = ordered.iloc[:max_clusters], ordered.iloc[max_clusters:]
    rest_n = rest["n_obs"].sum()
    weighted_mean = rest[cols].multiply(rest["n_obs"], axis=0).sum() / rest_n
    other = pd.DataFrame([{"n_obs": rest_n, **weighted_mean.to_dict()}], index=[f"기타({len(rest)}개 군집 합계)"])
    return pd.concat([top, other])


def _describe_cluster(cluster_id, z_row: pd.Series, n_obs: int) -> str:
    """군집 하나에 대해 "전체 평균과 상대적으로 차이가 크게 관찰된 변수"를 나열한다.

    "이 변수가 군집을 정의/결정한다"는 인과적 표현은 쓰지 않는다 — z-score 크기 순으로 관찰된
    사실만 나열한다.
    """
    top = z_row.reindex(z_row.abs().sort_values(ascending=False).index).head(TOP_VARS_PER_CLUSTER)
    parts = ", ".join(f"{name}(z={v:+.1f})" for name, v in top.items())
    return f"군집 {cluster_id}({n_obs:,}건): {parts}"


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = profile.numeric_columns
    sub = df[columns].dropna()
    scaled = StandardScaler().fit_transform(sub)

    best_k, silhouette_scores = _pick_k(scaled, thresholds.clustering_k_min, thresholds.clustering_k_max)
    labels = KMeans(n_clusters=best_k, random_state=0, n_init=10).fit_predict(scaled)

    pca_2d = PCA(n_components=2, random_state=0).fit_transform(scaled)
    plot_idx = downsample(pd.DataFrame(pca_2d, columns=["pc1", "pc2"]).assign(label=labels), max_points=8000)

    fig_dir = params["fig_dir"]
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(sorted(set(labels))):
        mask = plot_idx["label"] == label
        ax.scatter(plot_idx.loc[mask, "pc1"], plot_idx.loc[mask, "pc2"],
                   s=6, alpha=0.6, color=cmap(idx % 10), label=str(label))
    ax.set_xlabel("PC1 (변수를 2축으로 압축한 좌표, 실제 단위 아님)")
    ax.set_ylabel("PC2")
    ax.set_title(f"군집 산점도 (KMeans k={best_k}, PCA 2D 표시)", fontsize=10)
    ax.legend(title="cluster", fontsize=7, loc="best")
    fig.tight_layout()
    scatter_path = os.path.join(fig_dir, "clustering_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    full_table = sub.assign(cluster=labels).groupby("cluster")[columns].mean()
    full_table.insert(0, "n_obs", pd.Series(labels).value_counts().sort_index())
    display_table = _cap_for_display(full_table, columns, thresholds.max_displayed_clusters)
    z = (display_table[columns] - sub[columns].mean()) / sub[columns].std().replace(0, np.nan)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    z.T.plot(kind="bar", ax=ax2)
    ax2.axhline(0, color="#999999", linewidth=0.8)
    ax2.set_xlabel("변수")
    ax2.set_ylabel("z-score (0 = 전체 평균과 동일)")
    ax2.set_title("Cluster Profile — 군집별 평균이 전체 평균과 다른 정도", fontsize=10)
    ax2.tick_params(axis="x", rotation=90, labelsize=7)
    ax2.legend(title="cluster", fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig2.tight_layout()
    profile_path = os.path.join(fig_dir, "clustering_profile.png")
    fig2.savefig(profile_path, dpi=150)
    plt.close(fig2)

    descriptions = [
        _describe_cluster(idx, z.loc[idx], int(display_table.loc[idx, "n_obs"])) for idx in display_table.index
    ]
    score_text = ", ".join(f"k={k}: {v:.3f}" for k, v in silhouette_scores.items())

    return AnalysisResult(
        section_id="13_clustering",
        title="Clustering (군집분석)",
        purpose=(
            "라벨 없이 수치형 변수 값만으로 관측치를 몇 개 그룹으로 묶어, 그룹 간에 어떤 변수 "
            "값이 다르게 나타나는지 관찰합니다. 각 군집이 어떤 공정 상태를 뜻하는지는 판정하지 "
            "않습니다."
        ),
        rationale=(
            f"알고리즘: KMeans, 전처리: StandardScaler(변수별 단위 차이 제거), 탐색한 k 범위: "
            f"{thresholds.clustering_k_min}~{thresholds.clustering_k_max}(제조 공정이 이 개수만큼의 "
            f"상태를 가진다는 도메인 규칙이 아니라, 현재 구현이 탐색해보는 범위일 뿐입니다). "
            f"실루엣 점수(같은 군집 내 응집도와 다른 군집과의 분리도를 종합한 지표, -1~1, 클수록 "
            f"군집 구분이 뚜렷함): {score_text} → 점수가 가장 높은 k={best_k}를 선택했습니다. 이 "
            "선택은 후보 k들을 내부적으로 비교한 결과일 뿐, k=best가 실제 공정 상태의 개수라는 "
            "의미는 아닙니다. 군집 번호(0/1/2...)는 크기나 순서와 무관한 이름표입니다."
            "\n\n산점도의 색은 군집 번호이고 축은 표시를 위해 2차원으로 압축한 좌표라 실제 변수 "
            "단위가 아닙니다. 아래 Cluster Profile은 각 군집의 평균이 전체 평균에서 표준편차 몇 배 "
            "떨어져 있는지(z-score)를 보여줍니다 — 막대가 0에서 멀수록 그 변수에서 해당 군집이 "
            "전체와 다르게 관찰됐다는 뜻이며, 그 변수가 군집을 정의하거나 그 값을 유발했다는 뜻은 "
            "아닙니다.\n\n군집별로 전체 평균과 상대적으로 차이가 크게 관찰된 변수: "
            + " | ".join(descriptions)
        ),
        input_columns=columns,
        parameters={
            "algorithm": "KMeans",
            "k": best_k,
            "k_search_range": [thresholds.clustering_k_min, thresholds.clustering_k_max],
            "silhouette_scores": silhouette_scores,
            "scaling": "StandardScaler",
            "random_state": 0,
        },
        figures=[
            Figure(kind="cluster_scatter", image_path=scatter_path, caption=f"군집 수: {best_k}"),
            Figure(kind="cluster_profile", image_path=profile_path,
                   caption="z-score: 전체 평균에서 표준편차 몇 배 떨어져 있는지"),
        ],
        tables=[round_floats(display_table.reset_index().rename(columns={"index": "cluster"}))],
        ai_context={
            "algorithm": "KMeans",
            "k": best_k,
            "silhouette_scores": silhouette_scores,
            "cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
            "cluster_centroids": full_table.to_dict(orient="index"),
            "cluster_descriptions": descriptions,
        },
    )
