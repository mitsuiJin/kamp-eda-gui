"""업로드된 데이터프레임의 기본 정보·컬럼별 결측/유니크·기술통계를 계산하는 모듈."""

import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "행 수": df.shape[0],
        "열 수": df.shape[1],
        "메모리 사용량(MB)": df.memory_usage(deep=True).sum() / (1024**2),
        "중복행 수": int(df.duplicated().sum()),
    }


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    return pd.DataFrame(
        {
            "컬럼명": df.columns.astype(str),
            "dtype": df.dtypes.astype(str).values,
            "결측치 수": missing_count.values,
            "결측률(%)": (missing_count / len(df) * 100).round(2).values,
            "unique 수": df.nunique().values,
        }
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    categorical_df = df.select_dtypes(include=["object", "str", "bool", "category"])
    if categorical_df.empty:
        return pd.DataFrame()
    return categorical_df.describe().T
