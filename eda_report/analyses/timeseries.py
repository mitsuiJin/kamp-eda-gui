"""16. Time Series Analysis — 시간 순서에 따른 값의 변화를 관찰한다.

설계 원칙: 유효한 모든 수치형 feature에 대해 시간 순서 기준 요약통계를 계산한다 — 지면
제약 때문에 PDF에 대표 변수만 시각화하는 것은 허용하지만, "분석 대상이 대표 변수 몇 개뿐"인
것처럼 보이면 안 된다. 전체 계산 결과는 AI Context에 그대로 남긴다.

rolling window는 고정값이 아니라 데이터 길이에서 산출하고(설계 원칙 §10), 사용한 값과
산출 근거를 파라미터로 기록한다. sampling interval이 Dataset Guidebook에서 검증되지 않았다면
"172초"처럼 시간 단위로 표현하지 않고 "172개 관측치"처럼 개수로만 표현한다. 관찰 기능이며
설비 이상 여부를 판정하지 않는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

# 표준편차가 아니라 변동계수(CV = 표준편차 / |평균|)로 대표 변수를 고른다: 표준편차만 쓰면
# 값의 절대 크기가 큰 변수(예: 온도 300대)가 상대적으로 안정적이어도 무조건 상위로 뽑히고,
# 값의 크기가 작은 변수(예: 0~1 범위)는 실제로 변동이 커도 밀려나는 문제가 있다. CV는 평균 대비
# 상대적인 변동 정도를 보므로 스케일이 다른 변수를 더 공정하게 비교할 수 있다(단, 새로운 도메인
# 규칙이 아니라 잘 알려진 일반 통계량이다). 평균이 0에 가까운 변수는 CV가 정의되지 않으므로
# 표준편차로 대체한다.
_CV_MEAN_EPSILON = 1e-9


def _summarize_column(series: pd.Series) -> dict:
    non_null = series.dropna()
    if non_null.empty:
        return {
            "n_obs": 0, "mean": None, "std": None, "min": None, "max": None,
            "first_value": None, "last_value": None, "coefficient_of_variation": None,
        }
    mean = float(non_null.mean())
    std = float(non_null.std()) if len(non_null) > 1 else 0.0
    cv = std / abs(mean) if abs(mean) > _CV_MEAN_EPSILON else None
    return {
        "n_obs": int(len(non_null)),
        "mean": mean,
        "std": std,
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "first_value": float(non_null.iloc[0]),
        "last_value": float(non_null.iloc[-1]),
        "coefficient_of_variation": cv,
    }


def _rank_for_display(summary: dict[str, dict], max_n: int) -> list[str]:
    """변동계수 내림차순, CV가 정의되지 않은 변수는 표준편차 내림차순으로 뒤에 배치한다."""
    with_cv = [(name, s["coefficient_of_variation"]) for name, s in summary.items() if s["coefficient_of_variation"] is not None]
    without_cv = [(name, s["std"] or 0.0) for name, s in summary.items() if s["coefficient_of_variation"] is None]
    with_cv.sort(key=lambda item: item[1], reverse=True)
    without_cv.sort(key=lambda item: item[1], reverse=True)
    ordered = [name for name, _ in with_cv] + [name for name, _ in without_cv]
    return ordered[:max_n]


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    datetime_column = params["datetime_column"]
    datetime_source = params.get("datetime_source", "데이터에서 확인")
    fig_dir = params["fig_dir"]

    ordered = df.assign(**{datetime_column: pd.to_datetime(df[datetime_column], errors="coerce", format="mixed")})
    ordered = ordered.sort_values(datetime_column)

    numeric_columns = [c for c in profile.numeric_columns if c != datetime_column]

    # 1) 유효한 모든 수치형 feature에 대해 시간 순서 기준 요약통계를 계산한다(대표 변수만이
    #    아니라 전체) — PDF에는 아래에서 대표만 그리지만, 계산 자체는 전체를 대상으로 한다.
    all_columns_summary = {column: _summarize_column(ordered[column]) for column in numeric_columns}

    display_columns = _rank_for_display(all_columns_summary, thresholds.timeseries_max_series)
    window = thresholds.rolling_window(len(ordered))

    figures = []
    for column in display_columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(ordered[datetime_column], ordered[column], linewidth=0.6, alpha=0.6, label="원본 값")
        rolling = ordered[column].rolling(window, min_periods=1).mean()
        ax.plot(ordered[datetime_column], rolling, linewidth=1.4, color="#C44E52",
                label=f"이동평균({window}개 관측치)")
        ax.set_title(column, fontsize=10)
        ax.set_ylabel(column, fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"timeseries_{column.replace('.', '_')}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="line", image_path=path, caption=column))

    span = [str(ordered[datetime_column].min()), str(ordered[datetime_column].max())]
    n_total = len(numeric_columns)
    n_shown = len(display_columns)

    return AnalysisResult(
        section_id="16_time_series",
        title="Time Series Analysis (시계열 관찰)",
        purpose=(
            f"시간 컬럼({datetime_column}) 기준으로 정렬한 뒤, 유효한 수치형 변수 전체({n_total}개)에 "
            "대해 시간 순서 기준 요약통계(시작값/끝값/평균/표준편차/최소·최대)를 계산합니다. 변화의 "
            "원인이나 이상 여부는 판정하지 않습니다."
        ),
        rationale=(
            f"시간 변수 출처: {datetime_source}. 관측 구간: {span[0]} ~ {span[1]}. 전체 {n_total}개 "
            f"수치형 변수를 모두 계산했으며, 지면 제약으로 PDF에는 변동계수(coefficient of "
            f"variation, 표준편차/|평균| — 값의 절대 크기와 무관하게 상대적 변동이 큰 변수를 "
            f"찾기 위한 기준) 상위 {n_shown}개만 그래프로 표시합니다. 나머지 {n_total - n_shown}개의 "
            "요약통계도 AI Context(all_columns_summary)에 그대로 기록되어 있어 정보가 사라지지 "
            "않습니다.\n\n옅은 선은 원본 값, 진한 선은 이동평균(직전 N개 값의 평균을 한 칸씩 옮겨 "
            f"계산, N={window})입니다. 이동평균 구간은 전체 {len(ordered):,}행의 "
            f"{thresholds.rolling_window_ratio:.0%}에 해당하는 값으로 데이터 길이에서 산출했습니다"
            "(고정값이 아님). 실제 수집 주기(sampling interval)가 Dataset Guidebook에서 검증되지 "
            "않았다면 이 구간을 초/분 등 시간 단위로 환산하지 않고 '관측치 개수'로만 표현합니다."
        ),
        input_columns=[datetime_column] + numeric_columns,
        parameters={
            "datetime_column": datetime_column,
            "datetime_source": datetime_source,
            "rolling_window": window,
            "rolling_window_rule": f"max({thresholds.rolling_window_min}, min(행수×{thresholds.rolling_window_ratio}, {thresholds.rolling_window_max}))",
            "rolling_window_unit": "관측치 개수(검증된 sampling interval이 없어 시간 단위로 환산하지 않음)",
            "total_numeric_columns": n_total,
            "analyzed_columns": numeric_columns,
            "displayed_columns": display_columns,
            "display_selection_method": "변동계수(표준편차/|평균|) 내림차순 — 정의되지 않는 경우(평균≈0) 표준편차로 대체",
        },
        figures=figures,
        ai_context={
            "datetime_column": datetime_column,
            "time_span": span,
            "rolling_window": window,
            "analyzed_columns": numeric_columns,
            "displayed_columns": display_columns,
            "total_numeric_columns": n_total,
            "all_columns_summary": all_columns_summary,
        },
    )
