"""분석 모듈의 핵심 동작(특히 IQR=0 처리와 실행 조건 판정)에 대한 단위 테스트."""

import pandas as pd
import pytest

from eda_report.analyses import descriptive, outlier, timeseries
from eda_report.config import AnalysisThresholds, RunConfig
from eda_report.io.manifest import ParseManifest
from eda_report.pipeline import run as run_pipeline
from eda_report.profiling.dataset_profile import build_dataset_profile
from eda_report.selection.rules import decide_analyses

_MANIFEST = ParseManifest(
    file_path="test.csv", encoding="utf-8", delimiter=",", header_row=0,
    generated_column_names=False, n_rows=0, n_cols=0,
)


def _profile(df: pd.DataFrame, **kwargs):
    return build_dataset_profile(df, _MANIFEST, thresholds=AnalysisThresholds(), **kwargs)


def _run_config(**kwargs) -> RunConfig:
    return RunConfig(input_path="test.csv", output_dir="out", **kwargs)


def _decision(decisions, section_id):
    return next(d for d in decisions if d.section_id == section_id)


def test_iqr_zero_column_is_skipped_not_reported_as_outliers(tmp_path):
    # EX3.MELT_TEMP처럼 제어되어 한 값이 반복되면 Q1=중앙값=Q3 → IQR=0이 된다.
    controlled = [251] * 100 + [250] * 5 + [252] * 5
    varied = list(range(110))
    df = pd.DataFrame({"controlled": controlled, "varied": varied})
    profile = _profile(df)

    result = outlier.run_iqr(df, profile, {"fig_dir": str(tmp_path), "thresholds": AnalysisThresholds()})
    table = result.tables[0].set_index("column")

    assert table.loc["controlled", "iqr_status"] == "SKIPPED"
    assert pd.isna(table.loc["controlled", "outlier_rate"]) or table.loc["controlled", "outlier_rate"] is None
    assert table.loc["controlled", "mode"] == 251
    assert table.loc["varied", "iqr_status"] == "SUCCESS"
    assert "controlled" in result.parameters["skipped_columns_iqr_zero"]

    messages = " ".join(f.message_ko for f in result.findings)
    assert "IQR=0" in messages
    assert "이상치 없음" not in messages


def test_skewness_finding_has_no_transformation_recommendation():
    values = [1.0] * 200 + [-500.0]
    df = pd.DataFrame({"skewed": values})
    profile = _profile(df)

    result = descriptive.run(df, profile, {"thresholds": AnalysisThresholds()})
    text = " ".join(f.message_ko for f in result.findings) + (result.rationale or "")

    assert "왜도" in text
    for banned in ["로그 변환", "권장", "추천", "변환을 검토"]:
        assert banned not in text


def test_clustering_decision_does_not_depend_on_target_imbalance():
    balanced = pd.DataFrame({"a": range(200), "b": range(200), "t": [0, 1] * 100})
    imbalanced = pd.DataFrame({"a": range(200), "b": range(200), "t": [0] * 199 + [1]})

    for df in (balanced, imbalanced):
        profile = _profile(df, target_columns=["t"])
        decision = _decision(decide_analyses(profile, _run_config(target_columns=["t"])), "clustering")
        assert decision.run
        assert decision.params == {}  # 알고리즘 선택에 target 정보를 전달하지 않는다


def test_categorical_analyses_are_not_applicable_without_categorical_columns():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    decisions = decide_analyses(_profile(df), _run_config())

    for section in ("distribution_categorical", "cat_numeric", "cat_categorical", "association"):
        decision = _decision(decisions, section)
        assert not decision.run
        assert decision.status == "NOT_APPLICABLE"
        assert decision.reason


def test_target_analysis_not_applicable_when_target_not_specified():
    df = pd.DataFrame({"a": range(100), "b": [0, 1] * 50})
    decision = _decision(decide_analyses(_profile(df), _run_config()), "target")

    assert not decision.run
    assert decision.status == "NOT_APPLICABLE"
    assert "추측" in decision.reason


def test_timeseries_not_applicable_without_datetime_column():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    decision = _decision(decide_analyses(_profile(df), _run_config()), "timeseries")
    assert decision.status == "NOT_APPLICABLE"


def test_pca_not_applicable_with_too_few_numeric_columns():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    decision = _decision(decide_analyses(_profile(df), _run_config()), "pca")
    assert decision.status == "NOT_APPLICABLE"
    assert "수치형 변수 2개" in decision.reason


def test_association_skipped_when_combinations_explode():
    thresholds = AnalysisThresholds(association_max_combinations=10)
    df = pd.DataFrame(
        {
            "a": [f"a{i%6}" for i in range(120)],
            "b": [f"b{i%7}" for i in range(120)],
        }
    )
    profile = build_dataset_profile(df, _MANIFEST, thresholds=thresholds)
    decision = _decision(decide_analyses(profile, _run_config(thresholds=thresholds)), "association")

    assert decision.status == "SKIPPED"
    assert "조합" in decision.reason


@pytest.mark.parametrize("n_rows", [10, 50])
def test_multivariate_outlier_requires_enough_rows(n_rows):
    df = pd.DataFrame({"a": range(n_rows), "b": range(n_rows), "c": range(n_rows)})
    decision = _decision(decide_analyses(_profile(df), _run_config()), "outlier_multivariate")
    assert decision.status == "NOT_APPLICABLE"


def test_timeseries_computes_summary_for_all_columns_not_just_displayed(tmp_path):
    # 사용자 피드백 핵심 포인트: "전체 변수는 분석하고 PDF에는 대표 변수만 표시"가 실제로
    # 반영됐는지 확인한다 — displayed_columns는 4개로 잘리더라도, all_columns_summary에는
    # 6개 전부가 있어야 한다.
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="min")
    df = pd.DataFrame({"ts": dates})
    for i in range(6):
        df[f"var_{i}"] = range(i, i + n)

    thresholds = AnalysisThresholds(timeseries_max_series=4)
    profile = build_dataset_profile(df, _MANIFEST, thresholds=thresholds)
    params = {
        "datetime_column": "ts",
        "datetime_source": "테스트",
        "fig_dir": str(tmp_path),
        "thresholds": thresholds,
    }
    result = timeseries.run(df, profile, params)

    all_summary = result.ai_context["all_columns_summary"]
    assert set(all_summary.keys()) == {f"var_{i}" for i in range(6)}
    assert result.parameters["total_numeric_columns"] == 6
    assert len(result.parameters["displayed_columns"]) == 4
    assert len(result.figures) == 4
    # 표시되지 않은 변수도 요약통계 항목(평균/표준편차 등)을 온전히 갖고 있어야 한다.
    not_displayed = set(all_summary) - set(result.parameters["displayed_columns"])
    assert not_displayed
    for name in not_displayed:
        assert all_summary[name]["mean"] is not None
        assert all_summary[name]["n_obs"] == n


def test_timeseries_rolling_window_expressed_as_count_not_time_unit(tmp_path):
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="s")
    df = pd.DataFrame({"ts": dates, "value": range(n)})
    thresholds = AnalysisThresholds()
    profile = build_dataset_profile(df, _MANIFEST, thresholds=thresholds)
    result = timeseries.run(
        df, profile,
        {"datetime_column": "ts", "datetime_source": "테스트", "fig_dir": str(tmp_path), "thresholds": thresholds},
    )
    assert "관측치" in result.parameters["rolling_window_unit"]
    for banned in ["초마다", "분마다", f"{result.parameters['rolling_window']}초"]:
        assert banned not in (result.rationale or "")


def test_pipeline_assigns_core_and_advanced_tiers(tmp_path):
    df = pd.DataFrame(
        {
            "a": range(150),
            "b": [x * 1.5 for x in range(150)],
            "c": [x % 7 for x in range(150)],
            "target": [0] * 140 + [1] * 10,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    results = run_pipeline(
        RunConfig(input_path=str(csv_path), output_dir=str(tmp_path / "out"), target_columns=["target"], formats=[])
    )
    tiers = {r.section_id: r.tier for r in results}

    assert tiers["01_overview"] == "core"
    assert tiers["04_descriptive"] == "core"
    assert tiers["08_correlation"] == "core"
    assert tiers["12_pca"] == "advanced"
    assert tiers["13_clustering"] == "advanced"
    assert tiers["15_target_binary"] == "advanced"
