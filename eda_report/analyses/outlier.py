"""07. Outlier Analysis — IQR 기반(변수 1개씩) + 변수 조합 기반(다변량).

중요한 구분: 여기서 말하는 이상치는 **통계적 이상치**다. 공정 이상·설비 고장·불량과 동일한
의미가 아니며, EDA는 그 판단을 하지 않는다.

IQR=0 처리(설계 원칙 §3-[1]): 제어된 공정 변수는 Q1=Median=Q3이 되어 IQR=0이 나올 수 있다.
이때 IQR 공식을 그대로 적용하면 정상적인 ±1 편차까지 전부 이상치로 잡히므로, 해당 변수는
IQR 분석을 수행하지 않고 사유를 기록한다. "IQR=0 → 이상치 없음"으로 처리하지 않으며,
대체 탐지 기법을 자동으로 끼워 넣지도 않는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from eda_report.analyses.base import (
    AnalysisResult,
    Figure,
    Finding,
    round_floats,
    select_display_columns,
)
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

GRID_SIZE = 2
TOP_DEVIATING_VARS = 8


def _iqr_row(name: str, series: pd.Series) -> dict:
    non_null = series.dropna()
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = float(q3 - q1)
    mode_values = non_null.mode()
    mode_value = mode_values.iloc[0] if len(mode_values) else None
    row = {
        "column": name,
        "iqr": iqr,
        "n_unique": int(non_null.nunique()),
        "mode": mode_value,
        "mode_ratio": float((non_null == mode_value).mean()) if mode_value is not None else float("nan"),
        "min": float(non_null.min()) if len(non_null) else float("nan"),
        "max": float(non_null.max()) if len(non_null) else float("nan"),
    }

    if iqr <= 0:
        row.update({
            "iqr_status": "SKIPPED",
            "skip_reason": "IQR=0(Q1=중앙값=Q3)이라 IQR 기준 이상치 범위를 정의할 수 없음",
            "outlier_count": None,
            "outlier_rate": None,
        })
        return row

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (non_null < lower) | (non_null > upper)
    row.update({
        "iqr_status": "SUCCESS",
        "skip_reason": None,
        "outlier_count": int(mask.sum()),
        "outlier_rate": float(mask.mean()),
    })
    return row


def run_iqr(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = profile.numeric_columns
    rows = [_iqr_row(name, df[name]) for name in columns]
    table = pd.DataFrame(rows)

    analysed = table[table["iqr_status"] == "SUCCESS"].sort_values("outlier_rate", ascending=False)
    skipped = table[table["iqr_status"] == "SKIPPED"]

    findings: list[Finding] = [
        Finding(
            flag_type="iqr_outlier",
            columns=[row["column"]],
            metric=float(row["outlier_rate"]),
            severity="info",
            message_ko=f"{row['column']}: IQR 기준 통계적 이상치 비율 {row['outlier_rate']:.1%}로 관찰됨",
        )
        for _, row in analysed.head(5).iterrows()
        if row["outlier_rate"] > 0
    ]
    findings += [
        Finding(
            flag_type="iqr_not_defined",
            columns=[row["column"]],
            metric=0.0,
            severity="info",
            message_ko=(
                f"{row['column']}: IQR=0(값의 {row['mode_ratio']:.1%}가 {row['mode']}로 동일, "
                f"관측 범위 {row['min']:g}~{row['max']:g})으로 IQR 기준 이상치 분석을 수행하지 않음"
            ),
        )
        for _, row in skipped.iterrows()
    ]

    display_columns = select_display_columns(df, list(analysed["column"]), thresholds.max_display_columns)
    fig_dir = params["fig_dir"]
    figures = []
    for i in range(0, len(display_columns), GRID_SIZE):
        chunk = display_columns[i:i + GRID_SIZE]
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
        axes = [axes] if len(chunk) == 1 else list(axes)
        for ax, col in zip(axes, chunk):
            ax.boxplot(df[col].dropna())
            ax.set_title(col, fontsize=9)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"outlier_box_{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="boxplot", image_path=path, caption=", ".join(chunk)))

    rationale = (
        "IQR(중간 50% 구간의 폭)의 1.5배를 벗어난 값을 통계적 이상치로 집계했습니다. 이는 통계적 "
        "정의일 뿐이며 공정 이상·설비 고장·불량을 뜻하지 않습니다. 그래프 읽는 법: 박스는 중간 "
        "50% 구간, 박스 안 선은 중앙값, 수염 밖 동그라미가 위 기준을 벗어난 값입니다."
    )
    if len(skipped):
        rationale += (
            f"\n\n{len(skipped)}개 변수(" + ", ".join(skipped["column"].head(5)) +
            (" 등" if len(skipped) > 5 else "") +
            ")는 IQR=0이라 이 기준으로 이상치 범위를 정의할 수 없어 분석에서 제외했습니다. "
            "이상치가 없다는 뜻이 아니라, 이 방법으로는 판정할 수 없다는 뜻입니다."
        )
    if len(display_columns) < len(analysed):
        rationale += f"\n\n그래프는 분석된 {len(analysed)}개 중 표준편차가 큰 {len(display_columns)}개만 표시했습니다."

    return AnalysisResult(
        section_id="07_outlier_iqr",
        title="Outlier Analysis - IQR (통계적 이상치, 변수 1개씩)",
        purpose=(
            "각 수치형 변수에서 IQR 기준을 벗어난 값이 얼마나 관찰되는지 집계합니다. 관찰된 "
            "사실만 제시하며, 그 값이 실제 공정 이상인지에 대한 판단은 하지 않습니다."
        ),
        rationale=rationale,
        input_columns=columns,
        parameters={
            "method": "IQR 1.5x",
            "skipped_columns_iqr_zero": list(skipped["column"]),
            "displayed_columns": display_columns,
        },
        tables=[round_floats(table)],
        findings=findings,
        figures=figures,
        ai_context={
            "outlier_table": table.to_dict(orient="records"),
            "analysed_columns": list(analysed["column"]),
            "skipped_columns": list(skipped["column"]),
        },
    )


def run_multivariate(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    columns = profile.numeric_columns
    sub = df[columns].dropna()
    scaled = StandardScaler().fit_transform(sub)
    model = IsolationForest(random_state=0, contamination="auto")
    outlier_mask = model.fit_predict(scaled) == -1
    outlier_rate = float(outlier_mask.mean())

    fig_dir = params["fig_dir"]
    pca_2d = PCA(n_components=2, random_state=0).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(pca_2d[~outlier_mask, 0], pca_2d[~outlier_mask, 1], s=6, alpha=0.35,
               color="#4C72B0", label=f"판정 외 ({(~outlier_mask).sum():,}건)")
    ax.scatter(pca_2d[outlier_mask, 0], pca_2d[outlier_mask, 1], s=14, alpha=0.9,
               color="#C44E52", label=f"이상치로 판정 ({outlier_mask.sum():,}건)")
    ax.set_xlabel("PC1 (변수를 2축으로 압축한 좌표, 실제 단위 아님)")
    ax.set_ylabel("PC2")
    ax.set_title("변수 조합 기준 이상치 판정 결과 (PCA 2D 표시)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    scatter_path = os.path.join(fig_dir, "outlier_multivariate_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    z = (sub - sub.mean()) / sub.std().replace(0, np.nan)
    outlier_z_mean = z[outlier_mask].mean().sort_values(key=lambda s: s.abs(), ascending=False)
    top_vars = outlier_z_mean.head(TOP_DEVIATING_VARS)
    fig2, ax2 = plt.subplots(figsize=(6, 3.6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top_vars.values]
    ax2.barh(list(top_vars.index)[::-1], list(top_vars.values)[::-1], color=colors[::-1])
    ax2.axvline(0, color="#999999", linewidth=0.8)
    ax2.set_xlabel("이상치 판정 그룹의 평균 z-score (0 = 전체 평균과 동일)")
    ax2.set_title("판정 그룹이 전체 평균과 차이가 큰 변수", fontsize=10)
    fig2.tight_layout()
    bar_path = os.path.join(fig_dir, "outlier_multivariate_top_vars.png")
    fig2.savefig(bar_path, dpi=150)
    plt.close(fig2)

    top_desc = ", ".join(f"{name}({value:+.1f})" for name, value in top_vars.head(3).items())

    return AnalysisResult(
        section_id="07_outlier_multivariate",
        title="Outlier Analysis - 변수 조합 기준 (다변량)",
        purpose=(
            "변수 하나씩 보면 범위 안에 있어도 여러 변수의 조합이 다수 관측치와 다른 경우를 "
            "집계합니다. IQR 분석이 다루지 못하는 조합 관점의 관찰 결과이며, 공정 이상 여부를 "
            "판정하지 않습니다."
        ),
        rationale=(
            "이상치 판정은 Isolation Forest(변수 조합이 다른 관측치와 얼마나 떨어져 있는지로 점수를 "
            "매기는 방법)가 전체 수치형 변수를 대상으로 수행했으며, 변수 스케일 차이를 없애기 위해 "
            "표준화 후 계산했습니다. PCA는 판정에 관여하지 않았고, 그 판정 결과를 사람이 눈으로 볼 "
            "수 있도록 2차원 좌표로 압축해 위치만 보여주는 용도로만 쓰였습니다 — 왼쪽 그래프의 "
            "PC1/PC2 좌표는 실제 센서 단위가 없는 투영 좌표입니다. 오른쪽 그래프는 판정된 관측치들이 "
            "각 변수에서 전체 평균과 얼마나 차이 나는지(z-score)를 보여줄 뿐이며, 그 변수가 "
            "이상치의 원인이라는 뜻이 아닙니다."
        ),
        input_columns=columns,
        parameters={"method": "IsolationForest", "contamination": "auto", "scaling": "StandardScaler", "random_state": 0},
        figures=[
            Figure(kind="scatter", image_path=scatter_path),
            Figure(kind="bar", image_path=bar_path, caption="양수=판정 그룹이 평균보다 높음, 음수=평균보다 낮음"),
        ],
        tables=[round_floats(outlier_z_mean.rename("판정그룹_평균_zscore").reset_index().rename(columns={"index": "column"}))],
        findings=[
            Finding(
                flag_type="multivariate_outlier",
                columns=columns,
                metric=outlier_rate,
                severity="info",
                message_ko=(
                    f"변수 조합 기준으로 전체의 {outlier_rate:.1%}가 이상치로 판정됨. 전체 평균과 "
                    f"차이가 큰 변수는 {top_desc} 순으로 관찰됨"
                ),
            )
        ],
        ai_context={
            "outlier_rate": outlier_rate,
            "outlier_row_indices": sub.index[outlier_mask].tolist()[:200],
            "outlier_group_mean_zscore": outlier_z_mean.to_dict(),
        },
    )
