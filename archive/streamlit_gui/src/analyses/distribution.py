"""선택한 컬럼 하나에 대한 분포 통계(수치형) 계산을 담당하는 모듈."""

import pandas as pd


def numeric_column_stats(series: pd.Series) -> dict:
    return {
        "count": series.count(),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "25%": series.quantile(0.25),
        "median": series.median(),
        "75%": series.quantile(0.75),
        "max": series.max(),
        "skew": series.skew(),
        "missing": series.isna().sum(),
    }
