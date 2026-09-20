"""07. Outlier Analysis — IQR 기반(변수 1개씩).

중요한 구분: 여기서 말하는 이상치는 **통계적 이상치**다. 공정 이상·설비 고장·불량과 동일한
의미가 아니며, EDA는 그 판단을 하지 않는다 — "이상치"라는 표현은 IQR 통계 기준에 따른 탐지
결과라는 뜻으로만 쓰며, 원인이나 조치사항은 제시하지 않는다.

IQR=0 처리: 제어된 공정 변수는 Q1=Q3이 되어 IQR=0이 나올 수 있다. 이 경우에도 특별 취급 없이
동일한 공식(하한=Q1-1.5×IQR, 상한=Q3+1.5×IQR)을 그대로 적용한다 — IQR=0이면 하한=상한=Q1(=Q3)
이 되므로, 이 값과 다른 모든 값이 IQR 기준 이상치로 계산된다. 이는 "이 변수는 분석할 수 없다"는
뜻이 아니라 IQR 방법 자체의 성질이며, 결과 문구에서도 그렇게 있는 그대로 설명한다 — N/A나
"분석 불가"로 컬럼을 제외하지 않는다.

모든 수치형 컬럼에 대해 이상치 분석(집계 + 박스플롯)을 예외 없이 수행한다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, Finding, round_floats, with_description
from eda_report.profiling.dataset_profile import DatasetProfile

GRID_SIZE = 2


def _iqr_row(name: str, series: pd.Series) -> dict:
    non_null = series.dropna()
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = float(q3 - q1)
    mode_values = non_null.mode()
    mode_value = mode_values.iloc[0] if len(mode_values) else None

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (non_null < lower) | (non_null > upper)

    return {
        "column": name,
        "iqr": iqr,
        "n_unique": int(non_null.nunique()),
        "mode": mode_value,
        "mode_ratio": float((non_null == mode_value).mean()) if mode_value is not None else float("nan"),
        "min": float(non_null.min()) if len(non_null) else float("nan"),
        "max": float(non_null.max()) if len(non_null) else float("nan"),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_count": int(mask.sum()),
        "outlier_rate": float(mask.mean()),
    }


def run_iqr(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    columns = profile.numeric_columns
    rows = [_iqr_row(name, df[name]) for name in columns]
    table = pd.DataFrame(rows)
    zero_iqr_set = set(table.loc[table["iqr"] <= 0, "column"])

    ranked = table.sort_values("outlier_rate", ascending=False)
    findings: list[Finding] = [
        Finding(
            flag_type="iqr_outlier",
            columns=[row["column"]],
            metric=float(row["outlier_rate"]),
            severity="info",
            message_ko=f"{row['column']}: IQR 기준 통계적 이상치 비율 {row['outlier_rate']:.1%}로 관찰됨",
        )
        for _, row in ranked.head(5).iterrows()
        if row["outlier_rate"] > 0
    ]
    findings += [
        Finding(
            flag_type="iqr_zero_bound",
            columns=[row["column"]],
            metric=float(row["outlier_rate"]),
            severity="info",
            message_ko=(
                f"{row['column']}: IQR=0(Q1=Q3={row['mode']:g})으로 하한·상한이 {row['mode']:g}로 "
                f"계산됨 — 이 값과 다른 값은 모두 IQR 기준 이상치로 판정되어 비율 "
                f"{row['outlier_rate']:.1%}로 집계됨"
            ),
        )
        for _, row in table[table["column"].isin(zero_iqr_set)].iterrows()
    ]

    fig_dir = params["fig_dir"]
    column_glossary: dict[str, str] = params.get("column_glossary") or {}
    figures = []
    for i in range(0, len(columns), GRID_SIZE):
        chunk = columns[i:i + GRID_SIZE]
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.6 * len(chunk), 3.8))
        axes = [axes] if len(chunk) == 1 else list(axes)
        for ax, col in zip(axes, chunk):
            ax.boxplot(df[col].dropna())
            title = f"{col} (IQR=0)" if col in zero_iqr_set else col
            ax.set_title(with_description(title, col, column_glossary), fontsize=9)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"outlier_box_{i}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        caption = ", ".join(chunk)
        if any(c in zero_iqr_set for c in chunk):
            caption += " — (IQR=0) 표시된 변수는 하한=상한=최빈값으로 계산되어, 최빈값과 다른 값이 모두 이상치로 집계됨"
        figures.append(Figure(kind="boxplot", image_path=path, caption=caption))

    rationale = (
        "박스는 중간 50% 구간, 박스 안 선은 중앙값, 수염 밖 동그라미가 하한(Q1-1.5×IQR)~상한"
        "(Q3+1.5×IQR)을 벗어난 값입니다. 일부 변수는 IQR=0(Q1=Q3)이라 하한과 상한이 그 값과 "
        "동일하게 계산되며, 이 경우 그 값과 다른 모든 값이 이상치로 집계됩니다(표의 iqr·"
        "lower_bound·upper_bound 참고, 그래프 제목에 '(IQR=0)' 표시) — IQR 방법 자체의 성질이며 "
        "해당 변수를 분석에서 제외한다는 뜻이 아닙니다."
    )

    return AnalysisResult(
        section_id="07_outlier_iqr",
        title="Outlier Analysis - IQR (통계적 이상치, 변수 1개씩)",
        purpose=(
            "박스플롯으로 각 수치형 변수의 분포를 보여주고, IQR(사분위범위) 기준을 벗어난 값을 "
            "통계적 이상치로 집계합니다. 이 판정은 통계적 정의일 뿐 공정 이상·설비 고장·불량을 "
            "의미하지 않습니다."
        ),
        rationale=rationale,
        input_columns=columns,
        parameters={
            "method": "IQR 1.5x (하한=Q1-1.5×IQR, 상한=Q3+1.5×IQR)",
            "zero_iqr_columns": list(zero_iqr_set),
            "displayed_columns": columns,
        },
        tables=[round_floats(table)],
        findings=findings,
        figures=figures,
        ai_context={
            "outlier_table": table.to_dict(orient="records"),
            "zero_iqr_columns": list(zero_iqr_set),
        },
    )
