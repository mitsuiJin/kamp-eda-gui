"""컬럼별 객관적 사실을 기록하고, 분석 대상 선정에 쓸 역할(role)을 정한다.

설계 원칙(C): 역할은 **Dataset Guidebook 메타데이터가 확인된 경우 그것을 그대로 쓰고**, 없을
때만 pandas dtype이라는 객관적 사실로 결정한다. 고유값 개수만으로 numeric을 categorical로
바꾸거나 target 후보를 추측하는 로직은 두지 않는다 — 제조 데이터에는 값 종류가 적은
discrete numeric 변수(예: 제어 설정값)가 정상적으로 존재하기 때문이다.

role_source로 "이 판단이 어디서 왔는지"를 항상 기록한다.

storage dtype과 analysis role의 분리: 저장 형식(numeric/categorical/datetime/text)과 분석상
역할(예: target)은 서로 다른 축이다. 예를 들어 0/1로 저장된 `passorfail`은 저장 형식은
numeric이지만, Dataset Guidebook이 이를 품질 판정 결과로 정의하고 target으로 지정했다면
분석상 역할은 "target"이다. 이 구분은 `profile_column()`이 아니라 target 목록을 아는
`dataset_profile.build_dataset_profile()`에서 role을 "target"으로 덮어쓰는 방식으로 반영된다
(target_kind가 이진/다중클래스/연속형 중 무엇인지도 함께 기록한다) — 데이터의 의미를 코드가
추론하는 것이 아니라, Guidebook/실행 옵션이 명시한 target 지정을 그대로 반영할 뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Role = Literal["numeric", "categorical", "datetime", "text", "constant", "empty", "target"]
RoleSource = Literal["metadata", "dtype", "data_quality", "target_designation"]
TargetKind = Literal["binary_categorical", "multiclass_categorical", "continuous", "multilabel_member"]

# target의 고유값 개수가 이 값을 넘고 수치형이면 연속형 target으로 분류한다(표시용 분류일 뿐이며
# analyses/target.py의 실제 분기 로직과 같은 기준을 쓴다 — 두 곳 모두 "class로 보기엔 값 종류가
# 너무 많다"는 동일한 판단을 각자 위치에서 적용한다).
TARGET_CONTINUOUS_MIN_UNIQUE = 20

_TARGET_KIND_LABEL_KO = {
    "binary_categorical": "이진 범주형",
    "multiclass_categorical": "다중클래스 범주형",
    "continuous": "연속형",
    "multilabel_member": "이진(다중 컬럼 target 구성원)",
}


def classify_target_kind(series: pd.Series, is_multilabel_member: bool = False) -> TargetKind:
    """target 컬럼 하나의 성격을 분류한다(저장 dtype이 아니라 분석 역할 관점의 분류).

    이 분류는 target으로 **지정된** 컬럼에 대해서만 호출된다 — 어떤 컬럼이 target인지 자체는
    추론하지 않는다(Dataset Guidebook/실행 옵션이 지정한 컬럼만 대상).
    """
    if is_multilabel_member:
        return "multilabel_member"
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    if n_unique == 2:
        return "binary_categorical"
    if pd.api.types.is_numeric_dtype(series) and n_unique > TARGET_CONTINUOUS_MIN_UNIQUE:
        return "continuous"
    return "multiclass_categorical"


def target_kind_label(kind: str) -> str:
    return _TARGET_KIND_LABEL_KO.get(kind, kind)


@dataclass
class ColumnProfile:
    name: str
    role: Role
    role_source: RoleSource
    observed_dtype: str
    declared_type: str
    missing_rate: float
    n_unique: int
    unique_ratio: float
    sample_values: list = field(default_factory=list)
    target_kind: str | None = None  # role == "target"일 때만 값이 있음

    def to_dict(self) -> dict:
        return {
            "column": self.name,
            "role": self.role,
            "role_source": self.role_source,
            "observed_dtype": self.observed_dtype,
            "declared_type": self.declared_type,
            "missing_rate": round(self.missing_rate, 6),
            "n_unique": self.n_unique,
            "unique_ratio": round(self.unique_ratio, 6),
            "sample_values": [str(v) for v in self.sample_values],
            "target_kind": self.target_kind,
        }


def _datetime_parse_rate(non_null: pd.Series) -> float:
    if len(non_null) == 0:
        return 0.0
    parsed = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def profile_column(
    series: pd.Series,
    declared_type: str = "unknown",
    confirmed_type: str | None = None,
    datetime_parse_min_rate: float = 0.99,
    categorical_max_cardinality: int = 50,
) -> ColumnProfile:
    """컬럼 하나의 프로파일을 만든다.

    confirmed_type이 주어지면(= 메타데이터와 실제 데이터의 일치가 검증된 경우) 그 타입을
    역할로 사용한다. 그렇지 않으면 dtype 기반으로만 결정한다.
    """
    n = len(series)
    missing_rate = float(series.isna().sum() / n) if n else 1.0
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / len(non_null) if len(non_null) else 0.0
    sample_values = non_null.unique()[:5].tolist()
    observed_dtype = str(series.dtype)

    role: Role
    role_source: RoleSource

    if missing_rate >= 1.0:
        role, role_source = "empty", "data_quality"
    elif n_unique <= 1:
        role, role_source = "constant", "data_quality"
    elif confirmed_type in {"numeric", "categorical", "datetime", "text"}:
        role, role_source = confirmed_type, "metadata"  # type: ignore[assignment]
    elif pd.api.types.is_numeric_dtype(series):
        role, role_source = "numeric", "dtype"
    elif pd.api.types.is_datetime64_any_dtype(series):
        role, role_source = "datetime", "dtype"
    elif _datetime_parse_rate(non_null) >= datetime_parse_min_rate:
        role, role_source = "datetime", "dtype"
    elif n_unique <= categorical_max_cardinality:
        role, role_source = "categorical", "dtype"
    else:
        role, role_source = "text", "dtype"

    return ColumnProfile(
        name=str(series.name),
        role=role,
        role_source=role_source,
        observed_dtype=observed_dtype,
        declared_type=declared_type,
        missing_rate=missing_rate,
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        sample_values=sample_values,
    )
