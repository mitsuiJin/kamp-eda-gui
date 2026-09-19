"""사용자가 지정한 시간축·수치형 컬럼을 시간순으로 정렬해 추세를 관찰하는 로직을 담당하는 모듈.

이 모듈은 이상탐지를 하지 않는다 — 시간에 따른 변화를 보여주는 것까지가 범위다.
"""

import pandas as pd


def datetime_parseable_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for column in df.select_dtypes(include=["object", "str"]).columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            candidates.append(column)
    return candidates


def sort_by_time(df: pd.DataFrame, time_column: str, value_column: str) -> pd.DataFrame:
    result = df[[time_column, value_column]].copy()
    result[time_column] = pd.to_datetime(result[time_column], errors="coerce")
    return result.sort_values(time_column)
