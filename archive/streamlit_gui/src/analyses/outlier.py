"""IQR(1.5×IQR) 기준의 통계적 이상치를 계산하는 모듈.

모델 기반 이상탐지는 구현하지 않는다 — 통계적 정의(IQR)에 따른 이상치만 다룬다.
"""

import pandas as pd


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def outlier_mask(series: pd.Series) -> pd.Series:
    lower, upper = iqr_bounds(series)
    return (series < lower) | (series > upper)


def outlier_summary(series: pd.Series) -> dict:
    mask = outlier_mask(series)
    count = int(mask.sum())
    total = int(series.notna().sum())
    ratio = round(count / total * 100, 2) if total > 0 else 0.0
    lower, upper = iqr_bounds(series)
    return {
        "이상치 개수": count,
        "이상치 비율(%)": ratio,
        "하한": lower,
        "상한": upper,
    }
