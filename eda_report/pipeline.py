"""전체 오케스트레이션: load → profile → decide → run → render.

설계 원칙: EDA는 데이터 자체(구조적 사실)만으로 어떤 분석이 성립하는지 판정한다. 변수의
의미는 추론하지 않으며, target만 예외적으로 실행 옵션(`--target`)으로 명시된 경우에 한해
반영한다. 모든 분석은 실행 여부와 사유가 Manifest에 기록된다.
"""

from __future__ import annotations

import os
import time
import traceback

from eda_report.analyses import (
    cat_categorical,
    cat_numeric,
    column_profile_page,
    correlation,
    descriptive,
    distribution_categorical,
    distribution_numeric,
    findings,
    glossary,
    manifest_page,
    missing,
    outlier,
    overview,
    relationship,
    target,
    timeseries,
)
from eda_report.analyses.base import AnalysisResult, failed
from eda_report.column_glossary import load_column_glossary
from eda_report.config import RunConfig
from eda_report.io.loader import load_table
from eda_report.profiling.dataset_profile import build_dataset_profile
from eda_report.selection.rules import AnalysisDecision, decide_analyses

ANALYSIS_REGISTRY = {
    "overview": overview.run,
    "column_profile": column_profile_page.run,
    "missing": missing.run,
    "descriptive": descriptive.run,
    "distribution_numeric": distribution_numeric.run,
    "distribution_categorical": distribution_categorical.run,
    "outlier_iqr": outlier.run_iqr,
    "correlation": correlation.run,
    "relationship": relationship.run,
    "cat_numeric": cat_numeric.run,
    "cat_categorical": cat_categorical.run,
    "target": target.run,
    "timeseries": timeseries.run,
}

# 조건 미충족으로 실행되지 않은 분석도 리포트에 남기기 위한 표시 정보.
SECTION_INFO = {
    "overview": ("01_overview", "Dataset Overview (데이터 개요)", "데이터의 행/열 규모와 변수 구성을 확인합니다."),
    "column_profile": ("02_column_profile", "Column Profile (컬럼별 프로파일)", "변수별 dtype·결측률·고유값 수를 정리합니다."),
    "missing": ("03_missing", "Missing Values (결측치)", "변수별 결측 비율과 공통 결측 패턴을 확인합니다."),
    "descriptive": ("04_descriptive", "Descriptive Statistics (기술통계)", "수치형 변수의 기술통계를 산출합니다."),
    "distribution_numeric": ("05_numerical_distribution", "Numerical Distribution (수치형 변수 분포)", "수치형 변수의 분포 형태를 확인합니다."),
    "distribution_categorical": ("06_categorical_distribution", "Categorical Distribution (범주형 변수 분포)", "범주형 변수의 값별 빈도를 확인합니다."),
    "outlier_iqr": ("07_outlier_iqr", "Outlier Analysis - IQR (통계적 이상치, 변수 1개씩)", "IQR 기준 통계적 이상치를 집계합니다."),
    "correlation": ("08_correlation", "Correlation Analysis (상관관계 분석)", "변수 쌍의 상관계수를 산출합니다."),
    "relationship": ("09_relationship", "Feature Relationship (변수 쌍 산점도)", "선택된 변수 쌍의 분포를 확인합니다."),
    "cat_numeric": ("10_cat_numeric", "Categorical × Numerical (범주별 수치형 분포 비교)", "범주별 수치형 분포를 비교합니다."),
    "cat_categorical": ("11_cat_categorical", "Categorical × Categorical (범주형 간 교차 분석)", "범주형 변수 간 교차 분포를 확인합니다."),
    "target": ("15_target", "Target Analysis (목표변수 분석)", "지정된 target과 다른 변수의 관계를 정리합니다."),
    "timeseries": ("16_time_series", "Time Series Analysis (시계열 관찰)", "시간에 따른 값의 변화를 관찰합니다."),
}

# Core EDA: 데이터 구조만으로 항상 시도되는 기초 분석. Advanced/Optional EDA: target·시간축처럼
# 조건부로만 성립하는 분석. "Advanced"는 신뢰도가 낮다는 뜻이 아니라, 실행 조건이 데이터
# 구조가 아니라 사용자 지정(target)이나 특정 변수 종류(시간축)에 달려 있다는 신호다.
ADVANCED_SECTIONS = {"target", "timeseries"}


def run(run_config: RunConfig) -> list[AnalysisResult]:
    os.makedirs(run_config.output_dir, exist_ok=True)
    fig_dir = os.path.join(run_config.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df, parse_manifest = load_table(run_config.input_path)

    # target은 추론하지 않는다: 실행 옵션(--target)으로만 확정한다.
    target_columns = [t for t in (run_config.target_columns or []) if t in df.columns]

    profile = build_dataset_profile(
        df, parse_manifest, thresholds=run_config.thresholds, target_columns=target_columns,
    )

    # 컬럼 설명(사람이 검수한 표시용 텍스트)은 있으면 불러오되, 실행 흐름에는 영향을 주지
    # 않는다 — role/target/시간축 판정은 이미 위에서 데이터만으로 끝났다.
    column_glossary: dict[str, str] = {}
    if run_config.column_glossary_path:
        column_glossary = load_column_glossary(run_config.column_glossary_path)
        parse_manifest.warnings.append(
            f"컬럼 설명 {len(column_glossary)}개를 {run_config.column_glossary_path}에서 불러옴"
            "(표시용, 분석 대상 선정에는 사용하지 않음)"
        )

    decisions = decide_analyses(profile, run_config)
    results: list[AnalysisResult] = []
    entries: list[dict] = []

    for decision in decisions:
        section_id, title, purpose = SECTION_INFO[decision.section_id]
        tier = "advanced" if decision.section_id in ADVANCED_SECTIONS else "core"

        if not decision.run:
            results.append(
                AnalysisResult(
                    section_id=section_id, title=title, purpose=purpose, tier=tier,
                    status=decision.status, status_reason=decision.reason,
                )
            )
            entries.append(
                {"section_id": section_id, "status": decision.status, "reason": decision.reason, "tier": tier}
            )
            continue

        params = dict(decision.params)
        params["fig_dir"] = fig_dir
        params["thresholds"] = run_config.thresholds
        params["column_glossary"] = column_glossary

        started = time.perf_counter()
        try:
            result = ANALYSIS_REGISTRY[decision.section_id](df, profile, params)
        except Exception as exc:  # 한 분석의 실패가 리포트 전체를 막지 않게 한다
            detail = f"{type(exc).__name__}: {exc}"
            result = failed(section_id, title, purpose, detail)
            result.ai_context = {"traceback": traceback.format_exc(limit=5)}
        result.tier = tier
        result.duration_sec = round(time.perf_counter() - started, 3)
        results.append(result)
        entries.append(
            {
                "section_id": result.section_id,
                "status": result.status,
                "reason": result.status_reason,
                "tier": result.tier,
                "input_columns": result.input_columns,
                "parameters": result.parameters,
                "duration_sec": result.duration_sec,
            }
        )

    results.append(findings.build_key_observations(results))
    results.append(
        manifest_page.run(df, profile, {"entries": entries, "thresholds": run_config.thresholds.to_dict()})
    )
    results.append(glossary.build_glossary())
    # 섹션 번호 순으로 정렬한다(같은 번호면 추가된 순서 유지 → manifest 다음에 glossary).
    results.sort(key=lambda r: int(r.section_id.split("_")[0]))

    if "markdown" in run_config.formats:
        from eda_report.render.context.markdown_builder import render as render_md

        render_md(results, run_config.output_dir)
    if "json" in run_config.formats:
        from eda_report.render.context.json_builder import render as render_json

        render_json(
            results, run_config.output_dir, profile=profile, run_config=run_config,
            column_glossary=column_glossary,
        )
    if "pdf" in run_config.formats:
        from eda_report.render.pdf.builder import render as render_pdf

        render_pdf(results, run_config.output_dir)

    return results
