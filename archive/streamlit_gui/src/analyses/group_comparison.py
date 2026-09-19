"""사용자가 지정한 그룹 컬럼 기준으로 수치형 변수의 분포를 비교하는 로직을 담당하는 모듈.

프로그램은 그룹의 의미(품질/불량/정상/비정상 등)를 판단하지 않고, 사용자가 지정한 그룹별 통계만 계산한다.
"""

import pandas as pd


def group_counts(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    counts = df[group_column].value_counts(dropna=False)
    ratio = (counts / len(df) * 100).round(2)
    return pd.DataFrame(
        {
            "그룹": counts.index.astype(str),
            "count": counts.values,
            "비율(%)": ratio.values,
        }
    )


def group_numeric_summary(df: pd.DataFrame, group_column: str, numeric_columns: list[str]) -> pd.DataFrame:
    summary = df.groupby(group_column)[numeric_columns].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()
