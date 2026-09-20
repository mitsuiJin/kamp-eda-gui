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


def test_iqr_zero_column_still_computes_outliers_via_same_formula(tmp_path):
    # EX3.MELT_TEMP처럼 제어되어 한 값이 91%를 차지하면 Q1=Q3=251 → IQR=0이 된다. 이 경우에도
    # 컬럼을 제외하지 않고 동일한 공식(하한=Q1-1.5*IQR, 상한=Q3+1.5*IQR)을 그대로 적용한다 —
    # IQR=0이면 하한=상한=251이 되어, 251이 아닌 값(250×5, 252×5)이 전부 이상치로 집계된다.
    controlled = [251] * 100 + [250] * 5 + [252] * 5
    varied = list(range(110))
    df = pd.DataFrame({"controlled": controlled, "varied": varied})
    profile = _profile(df)

    result = outlier.run_iqr(df, profile, {"fig_dir": str(tmp_path), "thresholds": AnalysisThresholds()})
    table = result.tables[0].set_index("column")

    assert table.loc["controlled", "iqr"] == 0
    assert table.loc["controlled", "lower_bound"] == 251
    assert table.loc["controlled", "upper_bound"] == 251
    assert table.loc["controlled", "outlier_count"] == 10  # 250×5 + 252×5
    # 표시용 테이블은 round_floats()로 소수 4자리까지 반올림된다.
    assert table.loc["controlled", "outlier_rate"] == pytest.approx(10 / 110, abs=1e-4)
    assert table.loc["controlled", "mode"] == 251
    assert "controlled" in result.parameters["zero_iqr_columns"]

    messages = " ".join(f.message_ko for f in result.findings)
    assert "IQR=0" in messages
    assert "정의할 수 없음" not in messages
    assert "분석 불가" not in messages


def test_iqr_zero_boundary_flags_values_different_from_mode_as_outliers(tmp_path):
    # 요청받은 검증: 특정 값이 지배적이어서 IQR=0이 되는 경우, 그 값과 다른 값들이 실제로
    # 이상치로 판정되는지 확인한다. 단순 반복 5개(5,5,5,5,5,7,8,9)는 n=8이라 pandas의 기본
    # 선형보간 분위수 계산에서 Q3가 5를 넘어가 버려(Q1=5, Q3=7.25) IQR=0이 되지 않으므로,
    # 같은 취지를 실제로 IQR=0이 되는 비율(최빈값이 75% 이상)로 재현한다: 5가 10개, 7/8/9가 각 1개.
    df = pd.DataFrame({"value": [5] * 10 + [7, 8, 9]})
    profile = _profile(df)

    result = outlier.run_iqr(df, profile, {"fig_dir": str(tmp_path), "thresholds": AnalysisThresholds()})
    row = result.tables[0].set_index("column").loc["value"]

    assert row["iqr"] == 0
    assert row["lower_bound"] == 5
    assert row["upper_bound"] == 5
    # 5가 아닌 값(7, 8, 9) 3개가 전부 이상치로 판정되어야 한다.
    assert row["outlier_count"] == 3
    assert row["outlier_rate"] == pytest.approx(3 / 13, abs=1e-4)


def test_skewness_finding_has_no_transformation_recommendation():
    values = [1.0] * 200 + [-500.0]
    df = pd.DataFrame({"skewed": values})
    profile = _profile(df)

    result = descriptive.run(df, profile, {"thresholds": AnalysisThresholds()})
    text = " ".join(f.message_ko for f in result.findings) + (result.rationale or "")

    assert "왜도" in text
    for banned in ["로그 변환", "권장", "추천", "변환을 검토"]:
        assert banned not in text


def test_categorical_analyses_are_not_applicable_without_categorical_columns():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    decisions = decide_analyses(_profile(df), _run_config())

    for section in ("distribution_categorical", "cat_numeric", "cat_categorical"):
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


def test_timeseries_uniform_interval_is_verified_and_expressed_in_time_units(tmp_path):
    # 타임스탬프 간격이 데이터 전체에서 완전히 일정하면(여기서는 1초 간격), 도메인 문서 없이도
    # 데이터 자체에서 수집 주기를 검증하고 이동평균 구간을 시간 단위로도 함께 표시해야 한다.
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="s")
    df = pd.DataFrame({"ts": dates, "value": range(n)})
    thresholds = AnalysisThresholds()
    profile = build_dataset_profile(df, _MANIFEST, thresholds=thresholds)
    result = timeseries.run(
        df, profile,
        {"datetime_column": "ts", "datetime_source": "테스트", "fig_dir": str(tmp_path), "thresholds": thresholds},
    )
    assert result.parameters["sampling_interval_verified"] is True
    assert result.parameters["sampling_interval_seconds"] == pytest.approx(1.0)
    assert "관측치" in result.parameters["rolling_window_unit"]
    assert "검증됨" in result.parameters["rolling_window_unit"]


def test_timeseries_irregular_interval_falls_back_to_count_only(tmp_path):
    # 타임스탬프 간격이 들쭉날쭉하면(수집 주기를 신뢰할 수 없으면) 여전히 관측치 개수로만
    # 표현해야 하고, 존재하지 않는 간격을 시간 단위로 지어내면 안 된다.
    n = 200
    irregular_gaps = pd.Series(pd.to_timedelta([1, 2, 1, 5, 1, 3, 1, 2] * (n // 8), unit="s"))
    dates = pd.Timestamp("2024-01-01") + irregular_gaps.cumsum()
    df = pd.DataFrame({"ts": dates, "value": range(len(dates))})
    thresholds = AnalysisThresholds()
    profile = build_dataset_profile(df, _MANIFEST, thresholds=thresholds)
    result = timeseries.run(
        df, profile,
        {"datetime_column": "ts", "datetime_source": "테스트", "fig_dir": str(tmp_path), "thresholds": thresholds},
    )
    assert result.parameters["sampling_interval_verified"] is False
    assert "관측치" in result.parameters["rolling_window_unit"]
    for banned in ["초마다", "분마다", "검증됨"]:
        assert banned not in result.parameters["rolling_window_unit"]


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
    assert tiers["15_target_binary"] == "advanced"
