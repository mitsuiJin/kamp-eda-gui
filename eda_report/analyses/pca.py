"""12. PCA — 수치형 변수를 소수의 축으로 압축했을 때의 구조를 관찰한다.

Loading은 "각 원본 변수가 그 축을 구성하는 데 기여한 정도(가중치)"이며, 원인·중요도·인과관계를
뜻하지 않는다. 2D 투영 좌표의 거리도 압축 과정에서 손실된 정보가 있으므로, 설명분산 비율을
함께 제시해 과잉 해석을 막는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eda_report.analyses.base import AnalysisResult, Figure, round_floats
from eda_report.profiling.dataset_profile import DatasetProfile

MAX_LOADING_ROWS = 15
TOP_LOADING_VARS_PER_PC = 3
MAX_COMPONENTS = 10


def _describe_component(loadings: pd.DataFrame, pc: str) -> str:
    ordered = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
    top = ordered.head(TOP_LOADING_VARS_PER_PC)
    parts = ", ".join(f"{name}({value:+.2f})" for name, value in top.items())
    return f"{pc}: {parts}"


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    columns = profile.numeric_columns
    sub = df[columns].dropna()
    scaled = StandardScaler().fit_transform(sub)

    n_components = min(len(columns), MAX_COMPONENTS)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_

    fig_dir = params["fig_dir"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(range(1, n_components + 1), np.cumsum(explained), marker="o")
    ax1.set_xlabel("주성분 개수")
    ax1.set_ylabel("누적 설명분산 비율")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("누적 설명분산", fontsize=10)
    ax2.scatter(scores[:, 0], scores[:, 1], s=6, alpha=0.4, color="#4C72B0")
    ax2.set_xlabel(f"PC1 (설명분산 {explained[0]:.1%})")
    ax2.set_ylabel(f"PC2 (설명분산 {explained[1]:.1%})")
    ax2.set_title("PC1 vs PC2 투영", fontsize=10)
    fig.tight_layout()
    scatter_path = os.path.join(fig_dir, "pca_scree_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    n_loading = min(3, n_components)
    pc_names = [f"PC{i + 1}" for i in range(n_loading)]
    loadings = pd.DataFrame(pca.components_[:n_loading].T, index=columns, columns=pc_names)
    top_rows = loadings.abs().sum(axis=1).sort_values(ascending=False).head(MAX_LOADING_ROWS).index

    fig2, ax = plt.subplots(figsize=(6.5, 4.5))
    loadings.loc[top_rows].plot(kind="barh", ax=ax)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("loading (해당 축을 구성하는 가중치)")
    ax.set_title("PC1~PC3 Loading (절대값 상위 변수)", fontsize=10)
    fig2.tight_layout()
    loading_path = os.path.join(fig_dir, "pca_loadings.png")
    fig2.savefig(loading_path, dpi=150)
    plt.close(fig2)

    descriptions = " / ".join(_describe_component(loadings, pc) for pc in pc_names)

    return AnalysisResult(
        section_id="12_pca",
        title="PCA (주성분분석)",
        purpose=(
            "서로 연관된 수치형 변수들을 적은 수의 축(주성분)으로 압축했을 때 전체 변동이 어떻게 "
            "설명되는지, 어떤 변수들이 같은 축에 함께 기여하는지 관찰합니다."
        ),
        rationale=(
            "PC1은 데이터의 변동이 가장 큰 방향, PC2는 PC1과 겹치지 않는 방향 중 그다음으로 변동이 "
            "큰 방향으로 만들어진 축입니다. 변수 단위 차이를 없애기 위해 StandardScaler로 표준화 후 "
            "계산했습니다.\n\n왼쪽 그래프의 누적 설명분산은 축을 몇 개 썼을 때 원본 변동의 몇 %가 "
            f"설명되는지를 나타냅니다(PC1+PC2 = {explained[:2].sum():.1%}). 이 비율이 낮으면 오른쪽 "
            "2D 투영에서의 점 사이 거리만으로 전체 구조를 판단할 수 없습니다.\n\nLoading은 각 원본 "
            "변수가 그 축을 구성하는 데 들어간 가중치(-1~1)이며 변수의 중요도나 인과관계가 아닙니다. "
            "같은 축에서 같은 부호로 큰 값을 갖는 변수들은 함께 변동하는 경향이 관찰된 것입니다. "
            "이번 데이터의 축별 상위 기여 변수: " + descriptions
        ),
        input_columns=columns,
        parameters={
            "scaling": "StandardScaler",
            "n_components": n_components,
            "rows_used": int(len(sub)),
            "explained_variance_ratio": [float(v) for v in explained],
        },
        figures=[
            Figure(kind="pca_scree", image_path=scatter_path,
                   caption=f"PC1+PC2 누적 설명분산 {explained[:2].sum():.1%}"),
            Figure(kind="pca_loading", image_path=loading_path, caption="막대 길이 = 해당 축 구성 가중치의 크기"),
        ],
        tables=[round_floats(loadings.reset_index().rename(columns={"index": "column"}))],
        ai_context={
            "explained_variance_ratio": [float(v) for v in explained],
            "cumulative_explained_variance": [float(v) for v in np.cumsum(explained)],
            "loadings": loadings.to_dict(orient="index"),
            "component_top_contributors": {pc: _describe_component(loadings, pc) for pc in pc_names},
        },
    )
