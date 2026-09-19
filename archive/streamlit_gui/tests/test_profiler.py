"""profiler의 데이터 개요/컬럼 프로파일/기술통계 계산 로직에 대한 단위 테스트."""

import pandas as pd

from src.profiler import categorical_summary, column_profile, dataset_overview, numeric_summary


def test_dataset_overview_counts_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    overview = dataset_overview(df)
    assert overview["행 수"] == 3
    assert overview["열 수"] == 2
    assert overview["중복행 수"] == 1


def test_column_profile_reports_missing_and_unique():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "y"]})
    profile = column_profile(df)

    a_row = profile[profile["컬럼명"] == "a"].iloc[0]
    assert a_row["결측치 수"] == 1
    assert round(a_row["결측률(%)"], 2) == round(1 / 3 * 100, 2)

    b_row = profile[profile["컬럼명"] == "b"].iloc[0]
    assert b_row["unique 수"] == 2


def test_numeric_summary_only_includes_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    summary = numeric_summary(df)
    assert list(summary.index) == ["a"]
    assert "mean" in summary.columns


def test_numeric_summary_empty_when_no_numeric_columns():
    df = pd.DataFrame({"a": ["x", "y"]})
    assert numeric_summary(df).empty


def test_categorical_summary_reports_top_and_freq():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "y"]})
    summary = categorical_summary(df)
    assert list(summary.index) == ["b"]
    assert summary.loc["b", "top"] == "y"
    assert summary.loc["b", "freq"] == 2
