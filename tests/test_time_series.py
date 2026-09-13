"""time_series의 시간축 판별/정렬 로직에 대한 단위 테스트."""

import pandas as pd

from src.analyses.time_series import datetime_parseable_columns, sort_by_time


def test_datetime_parseable_columns_excludes_numeric_and_plain_text():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "ts": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    candidates = datetime_parseable_columns(df)
    assert candidates == ["ts"]


def test_sort_by_time_orders_chronologically():
    df = pd.DataFrame(
        {
            "ts": ["2020-01-03", "2020-01-01", "2020-01-02"],
            "value": [3, 1, 2],
        }
    )
    sorted_df = sort_by_time(df, "ts", "value")
    assert list(sorted_df["value"]) == [1, 2, 3]
