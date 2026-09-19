"""선택한 수치형 변수들 간의 Pearson 상관관계 계산을 담당하는 모듈."""

import pandas as pd


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr(method="pearson")
