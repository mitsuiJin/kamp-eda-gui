"""분석별 실행 조건을 한 곳에서 정의한다(설계 원칙 §3-[5] Dynamic Skip).

모든 분석은 실행 여부와 무관하게 결정(Decision)을 반환한다 — 조건을 만족하지 못한 분석도
"조용히 빠지는" 것이 아니라 NOT_APPLICABLE/SKIPPED 상태와 사유로 Manifest에 남는다.
분석마다 조건이 다르므로 "categorical < 2" 같은 단일 규칙을 일괄 적용하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eda_report.config import AnalysisThresholds, RunConfig
from eda_report.profiling.dataset_profile import DatasetProfile


@dataclass
class AnalysisDecision:
    section_id: str
    run: bool
    reason: str | None = None
    status: str = "SUCCESS"  # run=False일 때 NOT_APPLICABLE / SKIPPED
    params: dict = field(default_factory=dict)


def _ok(section_id: str, **params) -> AnalysisDecision:
    return AnalysisDecision(section_id, True, params=params)


def _na(section_id: str, reason: str) -> AnalysisDecision:
    return AnalysisDecision(section_id, False, reason=reason, status="NOT_APPLICABLE")


def _skip(section_id: str, reason: str) -> AnalysisDecision:
    return AnalysisDecision(section_id, False, reason=reason, status="SKIPPED")


def decide_analyses(profile: DatasetProfile, run_config: RunConfig) -> list[AnalysisDecision]:
    t: AnalysisThresholds = run_config.thresholds
    numeric = profile.numeric_columns
    categorical = profile.categorical_columns
    n_rows = profile.n_rows
    decisions: list[AnalysisDecision] = [_ok("overview"), _ok("column_profile")]

    has_missing = any(p.missing_rate > 0 for p in profile.columns)
    decisions.append(
        _ok("missing") if has_missing else _na("missing", "결측치가 있는 컬럼이 없음")
    )

    if not numeric:
        reason = "분석 가능한 수치형 변수가 없음"
        decisions += [_na(x, reason) for x in ("descriptive", "distribution_numeric", "outlier_iqr")]
    else:
        decisions += [_ok("descriptive"), _ok("distribution_numeric"), _ok("outlier_iqr")]

    if len(numeric) < t.multivariate_outlier_min_numeric:
        decisions.append(
            _na("outlier_multivariate",
                f"수치형 변수 {len(numeric)}개 — 변수 조합 기준 이상치 분석에는 {t.multivariate_outlier_min_numeric}개 이상 필요")
        )
    elif n_rows < t.multivariate_outlier_min_rows:
        decisions.append(
            _na("outlier_multivariate",
                f"행 {n_rows:,}개 — 밀도 추정이 불안정해 {t.multivariate_outlier_min_rows:,}행 미만에서는 수행하지 않음")
        )
    else:
        decisions.append(_ok("outlier_multivariate"))

    decisions.append(
        _ok("distribution_categorical") if categorical
        else _na("distribution_categorical", "Dataset Guidebook/데이터에서 확인된 범주형 변수가 없음")
    )

    if len(numeric) >= t.correlation_min_numeric:
        decisions += [_ok("correlation"), _ok("relationship")]
    else:
        reason = f"수치형 변수 {len(numeric)}개 — 상관관계는 {t.correlation_min_numeric}개 이상에서만 정의됨"
        decisions += [_na("correlation", reason), _na("relationship", reason)]

    if categorical and numeric:
        decisions.append(_ok("cat_numeric"))
    else:
        decisions.append(
            _na("cat_numeric", f"범주형 {len(categorical)}개 / 수치형 {len(numeric)}개 — 양쪽 모두 1개 이상 필요")
        )

    if len(categorical) >= 2:
        decisions.append(_ok("cat_categorical"))
    else:
        decisions.append(_na("cat_categorical", f"범주형 변수 {len(categorical)}개 — 교차 분석에는 2개 이상 필요"))

    decisions.append(_decide_association(profile, t))

    if len(numeric) >= t.pca_min_numeric:
        decisions.append(_ok("pca"))
    else:
        decisions.append(_na("pca", f"수치형 변수 {len(numeric)}개 — 차원 축약에는 {t.pca_min_numeric}개 이상 필요"))

    if len(numeric) < t.clustering_min_numeric:
        decisions.append(_na("clustering", f"수치형 변수 {len(numeric)}개 — 군집분석에는 {t.clustering_min_numeric}개 이상 필요"))
    elif n_rows < t.clustering_min_rows:
        decisions.append(_na("clustering", f"행 {n_rows:,}개 — 군집분석에는 {t.clustering_min_rows:,}행 이상 필요"))
    else:
        decisions.append(_ok("clustering"))

    decisions.append(_decide_timeseries(profile, t))
    decisions.append(_decide_target(profile))
    return decisions


def _decide_association(profile: DatasetProfile, t: AnalysisThresholds) -> AnalysisDecision:
    usable = [
        p.name for p in profile.columns
        if p.role == "categorical" and p.name in profile.categorical_columns
        and p.n_unique <= t.association_max_cardinality
    ]
    if len(usable) < 2:
        return _na(
            "association",
            f"고유값 {t.association_max_cardinality}개 이하인 범주형 변수가 {len(usable)}개 — 연관규칙에는 2개 이상 필요",
        )
    combinations = 1
    for name in usable:
        combinations *= max(profile.column(name).n_unique, 1)
    if combinations > t.association_max_combinations:
        return _skip(
            "association",
            f"범주 조합 수가 약 {combinations:,}개로 {t.association_max_combinations:,}개를 초과 — 조합 폭발로 수행하지 않음",
        )
    return _ok("association", columns=usable, combinations=combinations)


def _decide_timeseries(profile: DatasetProfile, t: AnalysisThresholds) -> AnalysisDecision:
    declared = None
    if profile.validation and profile.validation.datetime_status:
        declared = profile.validation.datetime_status.get("usable")
    column = declared or profile.datetime_column
    if column is None:
        return _na("timeseries", "Dataset Guidebook과 데이터 어디에서도 시간 변수를 확인하지 못함")
    if profile.n_rows < t.min_rows_for_analysis:
        return _na("timeseries", f"행 {profile.n_rows:,}개 — 추세를 보기에 표본이 부족함")
    source = "Dataset Guidebook 지정" if declared else "데이터에서 날짜 파싱 확인"
    return _ok("timeseries", datetime_column=column, datetime_source=source)


def _decide_target(profile: DatasetProfile) -> AnalysisDecision:
    if not profile.target_columns:
        return _na("target", "Dataset Guidebook/실행 옵션에서 target이 지정되지 않음(EDA가 target을 추측하지 않음)")
    return _ok("target", target_columns=profile.target_columns)
