"""16. Time Series Analysis — 시간 순서에 따른 값의 변화를 관찰한다.

설계 원칙: 유효한 모든 수치형 feature에 대해 시간 순서 기준 요약통계를 계산한다 — 지면
제약 때문에 PDF에 대표 변수만 시각화하는 것은 허용하지만, "분석 대상이 대표 변수 몇 개뿐"인
것처럼 보이면 안 된다. 전체 계산 결과는 AI Context에 그대로 남긴다.

rolling window는 고정값이 아니라 데이터 길이에서 산출하고(설계 원칙 §10), 사용한 값과
산출 근거를 파라미터로 기록한다. 수집 주기(sampling interval)는 도메인 문서가 아니라
타임스탬프 자체에서 검증한다 — 정렬된 연속 타임스탬프 간격이 충분히 일정하면(임계값은
config.timeseries_interval_uniform_min_rate) "검증된 주기"로 보고 이동평균 구간을 시간
단위로도 함께 표시하며, 불규칙하면 관측치 개수로만 표현한다. 관찰 기능이며 설비 이상
여부를 판정하지 않는다.
"""

from __future__ import annotations

import os

import eda_report.render.mpl_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_report.analyses.base import AnalysisResult, Figure, with_description
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


def _detect_sampling_interval(
    sorted_dt: pd.Series, min_uniform_rate: float
) -> tuple[pd.Timedelta | None, float]:
    """정렬된 타임스탬프의 연속 간격이 충분히 일정하면 그 간격을 "검증된 수집 주기"로 본다.

    Dataset Guidebook 같은 외부 문서가 아니라 데이터 자체(타임스탬프 간격의 최빈값과 그
    비율)로만 판단한다 — datetime_parse_min_rate와 같은 원칙.
    """
    diffs = sorted_dt.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None, 0.0
    mode_diff = diffs.mode().iloc[0]
    uniform_rate = float((diffs == mode_diff).mean())
    if uniform_rate >= min_uniform_rate:
        return mode_diff, uniform_rate
    return None, uniform_rate


def _format_timedelta(td: pd.Timedelta) -> str:
    seconds = td.total_seconds()
    if seconds < 120:
        return f"{seconds:g}초"
    minutes = seconds / 60
    if minutes < 120:
        return f"{minutes:.1f}분"
    return f"{minutes / 60:.1f}시간"


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
    column_glossary: dict[str, str] = params.get("column_glossary") or {}

    ordered = df.assign(**{datetime_column: pd.to_datetime(df[datetime_column], errors="coerce", format="mixed")})
    ordered = ordered.sort_values(datetime_column)

    numeric_columns = [c for c in profile.numeric_columns if c != datetime_column]

    # 1) 유효한 모든 수치형 feature에 대해 시간 순서 기준 요약통계를 계산한다(대표 변수만이
    #    아니라 전체) — PDF에는 아래에서 대표만 그리지만, 계산 자체는 전체를 대상으로 한다.
    all_columns_summary = {column: _summarize_column(ordered[column]) for column in numeric_columns}

    display_columns = _rank_for_display(all_columns_summary, thresholds.timeseries_max_series)
    window = thresholds.rolling_window(len(ordered))

    interval, interval_uniform_rate = _detect_sampling_interval(
        ordered[datetime_column], thresholds.timeseries_interval_uniform_min_rate
    )
    if interval is not None:
        window_label = f"이동평균({window}개 관측치, 약 {_format_timedelta(interval * window)})"
        rolling_window_unit = (
            f"관측치 개수(수집 간격 {_format_timedelta(interval)}이 데이터 전체의 "
            f"{interval_uniform_rate:.1%}에서 확인되어 검증됨 → 약 {_format_timedelta(interval * window)}에 해당)"
        )
    else:
        window_label = f"이동평균({window}개 관측치)"
        rolling_window_unit = "관측치 개수(타임스탬프 간격이 일정하지 않아 시간 단위로 환산하지 않음)"

    figures = []
    for column in display_columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(ordered[datetime_column], ordered[column], linewidth=0.6, alpha=0.6, label="원본 값")
        rolling = ordered[column].rolling(window, min_periods=1).mean()
        ax.plot(ordered[datetime_column], rolling, linewidth=1.4, color="#C44E52", label=window_label)
        ax.set_title(with_description(column, column, column_glossary), fontsize=10)
        ax.set_ylabel(column, fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(fig_dir, f"timeseries_{column.replace('.', '_')}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(Figure(kind="line", image_path=path, caption=column))

    span = [str(ordered[datetime_column].min()), str(ordered[datetime_column].max())]
    n_total = len(numeric_columns)

    return AnalysisResult(
        section_id="16_time_series",
        title="Time Series Analysis (시계열 관찰)",
        purpose=(
            "시간 순서에 따른 값의 변화를 관찰합니다. 추세나 반복되는 패턴이 있는지 확인할 수 "
            "있으며, 변화의 원인이나 이상 여부는 판정하지 않습니다."
        ),
        rationale=(
            "옅은 선은 원본 값, 진한 선은 이동평균입니다. 지면 제약으로 변동계수(상대적 변동) 상위 "
            "변수만 그래프로 표시했으며, 전체 변수의 요약통계는 AI Context(all_columns_summary)에 "
            "기록되어 있습니다. 수집 주기는 타임스탬프 간격 자체로 검증하며(도메인 문서 아님), "
            + (
                f"이번 데이터는 {_format_timedelta(interval)} 간격이 {interval_uniform_rate:.1%} "
                "확인되어 검증됨 — 이동평균 구간을 시간 단위로도 함께 표시합니다."
                if interval is not None
                else "이번 데이터는 타임스탬프 간격이 일정하지 않아 관측치 개수로만 표현합니다."
            )
        ),
        input_columns=[datetime_column] + numeric_columns,
        parameters={
            "datetime_column": datetime_column,
            "datetime_source": datetime_source,
            "rolling_window": window,
            "rolling_window_rule": f"max({thresholds.rolling_window_min}, min(행수×{thresholds.rolling_window_ratio}, {thresholds.rolling_window_max}))",
            "rolling_window_unit": rolling_window_unit,
            "sampling_interval_verified": interval is not None,
            "sampling_interval_seconds": interval.total_seconds() if interval is not None else None,
            "sampling_interval_uniform_rate": interval_uniform_rate,
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
            "sampling_interval_seconds": interval.total_seconds() if interval is not None else None,
            "sampling_interval_uniform_rate": interval_uniform_rate,
            "analyzed_columns": numeric_columns,
            "displayed_columns": display_columns,
            "total_numeric_columns": n_total,
            "all_columns_summary": all_columns_summary,
        },
    )
