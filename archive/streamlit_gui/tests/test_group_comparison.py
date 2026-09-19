"""group_comparison의 그룹별 count/비율/mean/std 계산 로직에 대한 단위 테스트."""

import pandas as pd

from src.analyses.group_comparison import group_counts, group_numeric_summary


def test_group_counts_reports_count_and_ratio():
    df = pd.DataFrame({"grp": ["A", "A", "A", "B"]})
    counts = group_counts(df, "grp")

    a_row = counts[counts["그룹"] == "A"].iloc[0]
    assert a_row["count"] == 3
    assert a_row["비율(%)"] == 75.0

    b_row = counts[counts["그룹"] == "B"].iloc[0]
    assert b_row["count"] == 1
    assert b_row["비율(%)"] == 25.0


def test_group_numeric_summary_computes_mean_and_std_per_group():
    df = pd.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1, 3, 10, 12]})
    summary = group_numeric_summary(df, "grp", ["val"])

    a_row = summary[summary["grp"] == "A"].iloc[0]
    assert a_row["val_mean"] == 2.0

    b_row = summary[summary["grp"] == "B"].iloc[0]
    assert b_row["val_mean"] == 11.0
