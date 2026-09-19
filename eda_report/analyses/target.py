"""15. Target Analysis — Dataset Guidebook/실행 옵션에서 target이 명시된 경우에만 수행한다.

EDA는 target을 추측하지 않는다. 또한 "평균 차이가 크다 = 중요한 변수"라고 단정하지 않으며,
클래스별 통계 차이라는 관찰 사실과 그 크기(표준화된 차이)만 제시한다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, Finding, round_floats, select_display_columns
from eda_report.profiling.dataset_profile import DatasetProfile

MAX_CLASS_DISPLAY = 20
TOP_DIFF_VARS = 4


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    target_columns = params["target_columns"]
    fig_dir = params["fig_dir"]

    if len(target_columns) > 1:
        return _run_multilabel(df, target_columns, fig_dir)

    target = target_columns[0]
    series = df[target].dropna()
    n_unique = int(series.nunique())

    if n_unique == 2:
        return _run_binary(df, profile, target, fig_dir)
    if pd.api.types.is_numeric_dtype(series) and n_unique > MAX_CLASS_DISPLAY:
        return _run_continuous(df, profile, target, fig_dir)
    return _run_multiclass(df, profile, target, fig_dir)


def _class_table(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    table = pd.DataFrame({"count": counts, "ratio": series.value_counts(normalize=True, dropna=False)})
    table.index.name = "class"
    return table.reset_index()


def _group_difference_table(df: pd.DataFrame, target: str, numeric_columns: list[str], classes: list) -> pd.DataFrame:
    """클래스별 평균과 표준화된 차이(pooled std 기준)를 산출한다."""
    valid = df[df[target].notna()]
    rows = []
    for column in numeric_columns:
        groups = [valid.loc[valid[target] == cls, column].dropna() for cls in classes]
        if any(len(g) < 2 for g in groups):
            continue
        means = [float(g.mean()) for g in groups]
        pooled_std = float(np.sqrt(np.mean([g.var(ddof=1) for g in groups])))
        diff = means[-1] - means[0]
        rows.append(
            {
                "column": column,
                f"mean[{classes[0]}]": means[0],
                f"mean[{classes[-1]}]": means[-1],
                "mean_diff": diff,
                "standardized_diff": diff / pooled_std if pooled_std > 0 else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.reindex(table["standardized_diff"].abs().sort_values(ascending=False).index)
    return table


def _run_binary(df: pd.DataFrame, profile: DatasetProfile, target: str, fig_dir: str) -> AnalysisResult:
    series = df[target].dropna()
    ratio_table = _class_table(series)
    minority_ratio = float(ratio_table["ratio"].min())
    classes = sorted(series.unique())
    numeric_columns = [c for c in profile.numeric_columns if c != target]

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.bar(ratio_table["class"].astype(str), ratio_table["ratio"], color=["#4C72B0", "#C44E52"])
    for i, ratio in enumerate(ratio_table["ratio"]):
        ax.text(i, ratio, f"{ratio:.2%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("비율")
    ax.set_title(f"{target} 클래스 구성비", fontsize=10)
    fig.tight_layout()
    ratio_path = os.path.join(fig_dir, "target_class_ratio.png")
    fig.savefig(ratio_path, dpi=150)
    plt.close(fig)
    figures = [Figure(kind="bar", image_path=ratio_path, caption="클래스별 관측 비율")]

    diff_table = _group_difference_table(df, target, numeric_columns, classes)
    top_columns = (
        list(diff_table["column"].head(TOP_DIFF_VARS))
        if not diff_table.empty
        else select_display_columns(df, numeric_columns, TOP_DIFF_VARS)
    )

    if top_columns:
        fig2, axes = plt.subplots(1, len(top_columns), figsize=(3.6 * len(top_columns), 3.2))
        axes = [axes] if len(top_columns) == 1 else list(axes)
        valid = df[df[target].notna()]
        for ax, column in zip(axes, top_columns):
            data = [valid.loc[valid[target] == cls, column].dropna() for cls in classes]
            ax.boxplot(data, tick_labels=[str(c) for c in classes])
            ax.set_title(column, fontsize=9)
            ax.set_xlabel(target, fontsize=8)
        fig2.tight_layout()
        box_path = os.path.join(fig_dir, "target_group_box.png")
        fig2.savefig(box_path, dpi=150)
        plt.close(fig2)
        figures.append(
            Figure(kind="boxplot", image_path=box_path,
                   caption="표준화된 클래스 간 평균 차이가 큰 순으로 선택된 변수의 클래스별 분포")
        )

    findings = [
        Finding(
            flag_type="class_ratio",
            columns=[target],
            metric=minority_ratio,
            severity="info",
            message_ko=f"{target}: 소수 클래스 비율 {minority_ratio:.2%}로 관찰됨(클래스 구성이 한쪽에 치우침)",
        )
    ]

    return AnalysisResult(
        section_id="15_target_binary",
        title="Target Analysis - 이진 (목표변수 분석)",
        purpose=(
            f"지정된 target({target})의 클래스 구성비와, 클래스별로 수치형 변수 값의 분포가 어떻게 "
            "다르게 관찰되는지 정리합니다."
        ),
        rationale=(
            "아래 표는 클래스별 평균과 그 차이를 pooled 표준편차로 나눈 표준화된 차이(단위가 다른 "
            "변수끼리 비교하기 위함)입니다. 표준화된 차이가 크다는 것은 두 클래스에서 값의 분포 "
            "위치가 다르게 관찰됐다는 뜻이며, 그 변수가 target의 원인이라거나 모델에서 중요하다는 "
            "뜻은 아닙니다. 박스플롯은 이 표의 상위 변수들을 클래스별로 그린 것입니다."
        ),
        input_columns=[target] + numeric_columns,
        parameters={
            "target": target,
            "classes": [str(c) for c in classes],
            "minority_ratio": minority_ratio,
            "ranking_metric": "standardized_diff (pooled std)",
            "displayed_columns": top_columns,
        },
        figures=figures,
        tables=[round_floats(ratio_table)] + ([round_floats(diff_table)] if not diff_table.empty else []),
        findings=findings,
        ai_context={
            "target": target,
            "class_ratio": ratio_table.set_index("class")["ratio"].to_dict(),
            "group_differences": diff_table.to_dict(orient="records") if not diff_table.empty else [],
        },
    )


def _run_multiclass(df: pd.DataFrame, profile: DatasetProfile, target: str, fig_dir: str) -> AnalysisResult:
    ratio_table = _class_table(df[target].dropna())
    shown = ratio_table.head(MAX_CLASS_DISPLAY)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(shown["class"].astype(str), shown["count"], color="#55A868")
    ax.set_ylabel("관측 건수")
    ax.set_title(f"{target} 클래스별 관측 건수", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    path = os.path.join(fig_dir, "target_multiclass_freq.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return AnalysisResult(
        section_id="15_target_multiclass",
        title="Target Analysis - 다중클래스 (목표변수 분석)",
        purpose=f"지정된 target({target})의 클래스별 관측 건수와 구성비를 정리합니다.",
        rationale=f"클래스가 {len(ratio_table)}개 관찰됐으며 그래프에는 상위 {len(shown)}개를 표시했습니다(전체는 Context에 기록).",
        input_columns=[target],
        parameters={"target": target, "n_classes": int(len(ratio_table))},
        figures=[Figure(kind="bar", image_path=path)],
        tables=[round_floats(shown)],
        ai_context={"target": target, "class_ratio": ratio_table.set_index("class")["ratio"].to_dict()},
    )


def _run_continuous(df: pd.DataFrame, profile: DatasetProfile, target: str, fig_dir: str) -> AnalysisResult:
    series = df[target].dropna()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.hist(series, bins=30, color="#4C72B0")
    ax.set_xlabel(target)
    ax.set_ylabel("빈도")
    ax.set_title(f"{target} 분포 (왜도 {series.skew():.2f})", fontsize=10)
    fig.tight_layout()
    path = os.path.join(fig_dir, "target_continuous_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    numeric_columns = [c for c in profile.numeric_columns if c != target]
    pearson = df[numeric_columns + [target]].corr(method="pearson")[target].drop(target)
    spearman = df[numeric_columns + [target]].corr(method="spearman")[target].drop(target)
    table = pd.DataFrame({"pearson": pearson, "spearman": spearman})
    table = table.reindex(table["pearson"].abs().sort_values(ascending=False).index).reset_index()
    table = table.rename(columns={"index": "column"})

    return AnalysisResult(
        section_id="15_target_continuous",
        title="Target Analysis - 연속값 (목표변수 분석)",
        purpose=f"지정된 target({target})의 분포와, 각 수치형 변수와의 상관계수를 산출합니다.",
        rationale="상관계수는 함께 움직이는 정도를 뜻하며 인과관계가 아닙니다. Pearson(직선)과 Spearman(순위 기반)을 함께 제시합니다.",
        input_columns=[target] + numeric_columns,
        parameters={"target": target, "skew": float(series.skew())},
        figures=[Figure(kind="histogram", image_path=path)],
        tables=[round_floats(table)],
        ai_context={
            "target": target,
            "skew": float(series.skew()),
            "correlation_with_target": table.set_index("column").to_dict(orient="index"),
        },
    )


def _run_multilabel(df: pd.DataFrame, target_columns: list[str], fig_dir: str) -> AnalysisResult:
    frequency = df[target_columns].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.4 * len(target_columns)), 3.6))
    ax.bar(frequency.index.astype(str), frequency.values, color="#C44E52")
    ax.set_ylabel("관측 건수")
    ax.set_title("target 컬럼별 관측 건수", fontsize=10)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    path = os.path.join(fig_dir, "target_multilabel.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    co_occurrence = int((df[target_columns].sum(axis=1) > 1).sum())

    return AnalysisResult(
        section_id="15_target_multilabel",
        title="Target Analysis - 다중 컬럼 (목표변수 분석)",
        purpose="target이 여러 개의 이진 컬럼으로 기록된 경우, 컬럼별 관측 건수와 동시 발생 건수를 정리합니다.",
        rationale=f"한 행에서 2개 이상의 target 컬럼이 동시에 1인 경우가 {co_occurrence:,}건 관찰됐습니다.",
        input_columns=list(target_columns),
        parameters={"target_columns": list(target_columns), "rows_with_multiple": co_occurrence},
        figures=[Figure(kind="bar", image_path=path)],
        tables=[frequency.rename("count").reset_index().rename(columns={"index": "target_column"})],
        ai_context={
            "target_columns": list(target_columns),
            "frequency": frequency.to_dict(),
            "rows_with_multiple_targets": co_occurrence,
        },
    )
