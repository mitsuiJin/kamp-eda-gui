"""전체 오케스트레이션: load → Dataset Guidebook metadata → validate → profile → decide → run → render.

설계 원칙: EDA는 Dataset Guidebook 메타데이터를 "받아서 검증"하고, 그 결과로 어떤 분석이
성립하는지만 판정한다. 모든 분석은 실행 여부와 사유가 Manifest에 기록된다.
"""

from __future__ import annotations

import os
import time
import traceback

import pandas as pd

from eda_report.analyses import (
    association,
    cat_categorical,
    cat_numeric,
    clustering,
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
    pca,
    relationship,
    target,
    timeseries,
)
from eda_report.analyses.base import AnalysisResult, failed
from eda_report.config import RunConfig
from eda_report.io.loader import load_table
from eda_report.metadata.pdf_extract import GuidelinePdfError, extract_guideline_metadata
from eda_report.metadata.schema import DatasetMetadata
from eda_report.metadata.validator import MetadataValidationReport, validate_metadata
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
    "outlier_multivariate": outlier.run_multivariate,
    "correlation": correlation.run,
    "relationship": relationship.run,
    "cat_numeric": cat_numeric.run,
    "cat_categorical": cat_categorical.run,
    "association": association.run,
    "pca": pca.run,
    "clustering": clustering.run,
    "target": target.run,
    "timeseries": timeseries.run,
}

# 조건 미충족으로 실행되지 않은 분석도 리포트에 남기기 위한 표시 정보.
SECTION_INFO = {
    "overview": ("01_overview", "Dataset Overview (데이터 개요)", "데이터의 행/열 규모와 변수 구성을 확인합니다."),
    "column_profile": ("02_column_profile", "Column Profile (컬럼별 프로파일)", "변수별 상태와 Dataset Guidebook 검증 결과를 정리합니다."),
    "missing": ("03_missing", "Missing Values (결측치)", "변수별 결측 비율과 공통 결측 패턴을 확인합니다."),
    "descriptive": ("04_descriptive", "Descriptive Statistics (기술통계)", "수치형 변수의 기술통계를 산출합니다."),
    "distribution_numeric": ("05_numerical_distribution", "Numerical Distribution (수치형 변수 분포)", "수치형 변수의 분포 형태를 확인합니다."),
    "distribution_categorical": ("06_categorical_distribution", "Categorical Distribution (범주형 변수 분포)", "범주형 변수의 값별 빈도를 확인합니다."),
    "outlier_iqr": ("07_outlier_iqr", "Outlier Analysis - IQR (통계적 이상치, 변수 1개씩)", "IQR 기준 통계적 이상치를 집계합니다."),
    "outlier_multivariate": ("07_outlier_multivariate", "Outlier Analysis - 변수 조합 기준 (다변량)", "변수 조합 기준 이상치를 집계합니다."),
    "correlation": ("08_correlation", "Correlation Analysis (상관관계 분석)", "변수 쌍의 상관계수를 산출합니다."),
    "relationship": ("09_relationship", "Feature Relationship (변수 쌍 산점도)", "선택된 변수 쌍의 분포를 확인합니다."),
    "cat_numeric": ("10_cat_numeric", "Categorical × Numerical (범주별 수치형 분포 비교)", "범주별 수치형 분포를 비교합니다."),
    "cat_categorical": ("11_cat_categorical", "Categorical × Categorical (범주형 간 교차 분석)", "범주형 변수 간 교차 분포를 확인합니다."),
    "association": ("17_association", "Association Analysis (연관규칙)", "범주 값 조합의 동시 출현 패턴을 산출합니다."),
    "pca": ("12_pca", "PCA (주성분분석)", "수치형 변수를 소수의 축으로 압축했을 때의 구조를 관찰합니다."),
    "clustering": ("13_clustering", "Clustering (군집분석)", "수치형 변수 공간에서 관측치가 어떻게 묶이는지 관찰합니다."),
    "target": ("15_target", "Target Analysis (목표변수 분석)", "지정된 target과 다른 변수의 관계를 정리합니다."),
    "timeseries": ("16_time_series", "Time Series Analysis (시계열 관찰)", "시간에 따른 값의 변화를 관찰합니다."),
}

# Core EDA: 데이터 구조만으로 항상 시도되는 기초 분석. Advanced/Optional EDA: 통계적 모델을
# 쓰거나(IsolationForest/PCA/KMeans) target·시간축처럼 조건부로만 성립하는 분석. "Advanced"는
# 신뢰도가 낮다는 뜻이 아니라, 해석 시 알고리즘 선택/파라미터를 함께 감안해야 한다는 신호다.
ADVANCED_SECTIONS = {
    "outlier_multivariate", "pca", "clustering", "target", "timeseries", "association",
}


def _resolve_metadata(run_config: RunConfig, df: pd.DataFrame) -> tuple[DatasetMetadata, list[str]]:
    """메타데이터 출처 우선순위: 사용자 제공 JSON > Dataset Guidebook PDF 초안 > 없음."""
    notes: list[str] = []
    if run_config.metadata_path:
        metadata = DatasetMetadata.from_json_file(run_config.metadata_path)
        notes.append(f"메타데이터 JSON 사용: {run_config.metadata_path}")
        return metadata, notes

    if run_config.guideline_pdf_path:
        try:
            metadata = extract_guideline_metadata(run_config.guideline_pdf_path, list(df.columns))
            notes.append(
                f"Dataset Guidebook PDF에서 초안 추출: {run_config.guideline_pdf_path} "
                f"(컬럼 {len(metadata.columns)}개 매칭, 타입은 미확정 — 아래 검증 결과로 confirmed 여부 확인)"
            )
            return metadata, notes
        except GuidelinePdfError as exc:
            notes.append(f"Dataset Guidebook PDF 처리 실패 — 메타데이터 없이 진행: {exc}")

    return DatasetMetadata.empty(), notes


def run(run_config: RunConfig) -> list[AnalysisResult]:
    os.makedirs(run_config.output_dir, exist_ok=True)
    fig_dir = os.path.join(run_config.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df, parse_manifest = load_table(run_config.input_path)
    metadata, metadata_notes = _resolve_metadata(run_config, df)
    parse_manifest.warnings.extend(metadata_notes)

    validation: MetadataValidationReport = validate_metadata(
        metadata, df, datetime_min_rate=run_config.thresholds.datetime_parse_min_rate
    )

    # target은 추론하지 않는다: 실행 옵션 > Dataset Guidebook 순으로만 확정한다.
    target_columns = list(run_config.target_columns or [])
    if not target_columns and validation.target_status.get("usable"):
        target_columns = list(validation.target_status["usable"])
    target_columns = [t for t in target_columns if t in df.columns]

    profile = build_dataset_profile(
        df, parse_manifest, metadata=metadata, validation=validation,
        thresholds=run_config.thresholds, target_columns=target_columns,
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

        render_json(results, run_config.output_dir, profile=profile, run_config=run_config)
    if "pdf" in run_config.formats:
        from eda_report.render.pdf.builder import render as render_pdf

        render_pdf(results, run_config.output_dir)

    return results
