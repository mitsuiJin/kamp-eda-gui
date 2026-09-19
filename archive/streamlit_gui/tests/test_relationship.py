"""relationship의 상관관계 계산 로직에 대한 단위 테스트."""

import pandas as pd

from src.analyses.relationship import correlation_matrix


def test_correlation_matrix_perfect_positive_and_negative():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 3, 2, 1]})
    corr = correlation_matrix(df, ["a", "b", "c"])
    assert corr.loc["a", "b"] == 1.0
    assert corr.loc["a", "c"] == -1.0


def test_correlation_matrix_only_uses_selected_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 1, 1]})
    corr = correlation_matrix(df, ["a", "b"])
    assert list(corr.columns) == ["a", "b"]
