"""모든 분석 모듈이 공유하는 데이터 계약과 공통 유틸.

설계 원칙(A): 분석 결과는 "객관적으로 관찰된 사실"이다. Finding은 권장/판단이 아니라
관찰 기록이며, 분석을 수행하지 못한 경우에도 상태와 사유를 남겨 추적 가능하게 한다.

상태 구분(Manifest §7):
- SUCCESS         : 분석을 수행하고 결과를 산출함
- SKIPPED         : 실행 조건은 갖췄으나 의도적으로 수행하지 않음(예: 조합 폭발 위험)
- NOT_APPLICABLE  : 데이터 구조상 해당 분석이 성립하지 않음(예: 수치형 변수 부족)
- FAILED          : 수행을 시도했으나 계산/렌더링 오류로 실패함
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Severity = Literal["info", "warning", "critical"]
AnalysisStatus = Literal["SUCCESS", "SKIPPED", "NOT_APPLICABLE", "FAILED"]
# Core: 데이터 구조만으로 항상 시도되는 기초 분석. Advanced: 통계적 모델(IsolationForest,
# PCA, KMeans)을 쓰거나 target/시간축처럼 조건부로만 성립하는 분석 — 결과 해석에 더 주의가
# 필요하다는 신호일 뿐, Advanced가 "덜 중요하다"는 뜻은 아니다.
AnalysisTier = Literal["core", "advanced"]


@dataclass
class Figure:
    kind: str
    image_path: str
    caption: str = ""


@dataclass
class Finding:
    """관찰된 사실 한 건. 권장·원인·도메인 해석을 담지 않는다."""

    flag_type: str
    columns: list[str]
    metric: float | None
    severity: Severity
    message_ko: str


@dataclass
class AnalysisResult:
    section_id: str
    title: str
    purpose: str
    rationale: str | None = None
    tier: AnalysisTier = "core"
    status: AnalysisStatus = "SUCCESS"
    status_reason: str | None = None
    input_columns: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    duration_sec: float | None = None
    figures: list[Figure] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    ai_context: dict = field(default_factory=dict)

    @property
    def executed(self) -> bool:
        return self.status == "SUCCESS"


def not_applicable(section_id: str, title: str, purpose: str, reason: str) -> AnalysisResult:
    return AnalysisResult(
        section_id=section_id, title=title, purpose=purpose,
        status="NOT_APPLICABLE", status_reason=reason,
    )


def skipped(section_id: str, title: str, purpose: str, reason: str) -> AnalysisResult:
    return AnalysisResult(
        section_id=section_id, title=title, purpose=purpose,
        status="SKIPPED", status_reason=reason,
    )


def failed(section_id: str, title: str, purpose: str, reason: str) -> AnalysisResult:
    return AnalysisResult(
        section_id=section_id, title=title, purpose=purpose,
        status="FAILED", status_reason=reason,
    )


def select_display_columns(df: pd.DataFrame, columns: list[str], max_n: int) -> list[str]:
    """지면 제약으로 일부만 그릴 때 사용할 컬럼을 고른다(선정 기준은 Context에 기록된다).

    기준: 표준편차가 큰 순서. 값이 거의 변하지 않는 변수는 그래프로 볼 정보가 적기 때문이며,
    변수의 중요도를 뜻하지 않는다.
    """
    if len(columns) <= max_n:
        return list(columns)
    spread = df[columns].std(numeric_only=True).sort_values(ascending=False)
    return list(spread.index[:max_n])


def downsample(df: pd.DataFrame, max_points: int = 5000, random_state: int = 0) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=random_state).sort_index()


def with_description(
    label: str, column_name: str, column_glossary: dict[str, str] | None, max_chars: int = 30
) -> str:
    """그래프 제목/축 라벨에 컬럼 원래 이름(label)과 사람이 검수한 설명을 겹치지 않게 병기한다.

    설명을 label 뒤에 이어 붙이지 않고 줄바꿈으로 분리한다 — 같은 줄에 붙이면 그래프 폭에
    따라 글자가 겹치거나 잘리기 쉽기 때문이다. matplotlib 제목/축 라벨은 '\\n'을 그대로
    여러 줄로 렌더링한다. column_glossary가 없거나 해당 컬럼 설명이 없으면 label을 그대로
    돌려준다(원래 이름만 표시) — 이 함수는 표시 문자열만 조합할 뿐 어떤 분석 로직에도
    관여하지 않는다.
    """
    if not column_glossary:
        return label
    description = column_glossary.get(column_name)
    if not description:
        return label
    if len(description) > max_chars:
        description = description[: max_chars - 1] + "…"
    return f"{label}\n({description})"


def round_floats(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """표시용 표의 소수 자릿수만 정리한다(ai_context의 원본 수치는 건드리지 않는다)."""
    float_cols = df.select_dtypes(include="float").columns
    if len(float_cols) == 0:
        return df
    return df.assign(**{c: df[c].round(decimals) for c in float_cols})
