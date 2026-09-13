"""distribution의 수치형 컬럼 통계 계산 로직에 대한 단위 테스트."""

import pandas as pd

from src.analyses.distribution import numeric_column_stats


def test_numeric_column_stats_basic_values():
    series = pd.Series([1, 2, 3, 4, None], name="x")
    stats = numeric_column_stats(series)
    assert stats["count"] == 4
    assert stats["mean"] == 2.5
    assert stats["min"] == 1
    assert stats["max"] == 4
    assert stats["median"] == 2.5
    assert stats["missing"] == 1


def test_numeric_column_stats_skew_of_symmetric_data_is_near_zero():
    series = pd.Series([1, 2, 3, 4, 5])
    stats = numeric_column_stats(series)
    assert abs(stats["skew"]) < 1e-9
