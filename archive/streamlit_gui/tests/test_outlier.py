"""outlier의 IQR 기반 이상치 계산 로직에 대한 단위 테스트."""

import pandas as pd

from src.analyses.outlier import outlier_mask, outlier_summary


def test_outlier_mask_flags_extreme_value():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 100])
    mask = outlier_mask(series)
    assert mask.iloc[-1]
    assert not mask.iloc[:-1].any()


def test_outlier_summary_counts_and_ratio():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 100])
    summary = outlier_summary(series)
    assert summary["이상치 개수"] == 1
    assert summary["이상치 비율(%)"] == round(1 / 8 * 100, 2)


def test_outlier_summary_no_outliers_in_uniform_data():
    series = pd.Series([1, 2, 3, 4, 5])
    summary = outlier_summary(series)
    assert summary["이상치 개수"] == 0
    assert summary["이상치 비율(%)"] == 0.0
