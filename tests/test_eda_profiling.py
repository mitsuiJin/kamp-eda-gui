"""컬럼 역할 판정과 데이터셋 프로파일 집계에 대한 단위 테스트.

핵심 회귀 방지 포인트: 고유값이 적은 numeric 변수를 categorical로 바꾸지 않는다(설계 원칙 C).
"""

import pandas as pd

from eda_report.config import AnalysisThresholds
from eda_report.io.manifest import ParseManifest
from eda_report.metadata.schema import ColumnMetadata, DatasetMetadata
from eda_report.metadata.validator import validate_metadata
from eda_report.profiling.column_profile import profile_column
from eda_report.profiling.dataset_profile import build_dataset_profile

_MANIFEST = ParseManifest(
    file_path="test.csv", encoding="utf-8", delimiter=",", header_row=0,
    generated_column_names=False, n_rows=0, n_cols=0,
)


def test_constant_column_flagged():
    assert profile_column(pd.Series([5, 5, 5, 5], name="equip_cd")).role == "constant"


def test_fully_missing_column_flagged_empty():
    assert profile_column(pd.Series([None, None, None], name="reason")).role == "empty"


def test_low_cardinality_numeric_stays_numeric():
    # EX1.MD_TQ처럼 값이 0/72 두 종류뿐인 제어 변수도 numeric으로 유지되어야 한다.
    series = pd.Series([72] * 100 + [0] * 5, name="EX1.MD_TQ")
    profile = profile_column(series)
    assert profile.role == "numeric"
    assert profile.role_source == "dtype"


def test_high_cardinality_text_is_text_not_categorical():
    values = [f"id_{i}" for i in range(200)]
    assert profile_column(pd.Series(values, name="_id")).role == "text"


def test_datetime_string_column_detected():
    values = ["2020-10-16 04:57:47", "2020-10-16 04:58:48", "2020-10-16 04:59:48"]
    assert profile_column(pd.Series(values, name="TimeStamp")).role == "datetime"


def test_metadata_declared_type_overrides_dtype():
    # Guideline이 categorical이라고 선언하고 검증도 통과하면, 숫자 코드여도 categorical로 쓴다.
    series = pd.Series([1, 2, 3, 1, 2, 3] * 10, name="mold_code")
    profile = profile_column(series, declared_type="categorical", confirmed_type="categorical")
    assert profile.role == "categorical"
    assert profile.role_source == "metadata"


def test_dataset_profile_groups_columns_by_role():
    df = pd.DataFrame(
        {
            "equip_cd": ["S14"] * 60,
            "injection_time": [1.2, 2.3, 0.9, 3.1, 2.0, 1.8] * 10,
            "part_name": ["A", "B"] * 30,
            "shift_code": [1, 2, 3] * 20,
        }
    )
    profile = build_dataset_profile(df, _MANIFEST, thresholds=AnalysisThresholds())

    assert "injection_time" in profile.numeric_columns
    assert "shift_code" in profile.numeric_columns  # 숫자 코드를 임의로 categorical로 바꾸지 않는다
    assert profile.categorical_columns == ["part_name"]
    assert "equip_cd" in profile.excluded_columns


def test_target_columns_excluded_from_feature_sets():
    df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
    profile = build_dataset_profile(df, _MANIFEST, target_columns=["y"])
    assert profile.numeric_columns == ["x"]
    assert profile.target_columns == ["y"]


def test_numeric_stored_binary_target_gets_target_role_not_numeric():
    # passorfail처럼 0/1로 저장된 target: storage dtype은 numeric이지만 analysis role은
    # target(이진 범주형)이어야 하고, feature 집합(numeric_columns)에는 들어가면 안 된다.
    df = pd.DataFrame({"pressure": range(100), "passorfail": [0, 1] * 50})
    profile = build_dataset_profile(df, _MANIFEST, target_columns=["passorfail"])

    target_col = profile.column("passorfail")
    assert target_col.role == "target"
    assert target_col.role_source == "target_designation"
    assert target_col.target_kind == "binary_categorical"
    assert target_col.observed_dtype != "target"  # storage dtype 정보는 그대로 보존됨
    assert "passorfail" not in profile.numeric_columns
    assert "passorfail" not in profile.categorical_columns


def test_multilabel_targets_get_multilabel_member_kind():
    # 37_주조_품질보증처럼 결함이 여러 개의 개별 이진 컬럼으로 나뉘어 기록된 경우를 재현한다.
    df = pd.DataFrame({"x": range(50), "defect_a": [0, 1] * 25, "defect_b": [1, 0, 0, 0] * 12 + [1, 0]})
    profile = build_dataset_profile(df, _MANIFEST, target_columns=["defect_a", "defect_b"])
    assert profile.column("defect_a").target_kind == "multilabel_member"
    assert profile.column("defect_b").target_kind == "multilabel_member"


def test_role_counts_separate_target_from_numeric_features():
    # "19 numeric + 1 datetime"처럼 dtype 기준으로 뭉치지 않고, target은 role="target"으로
    # 따로 집계되어야 한다("18 numeric feature + 1 target"으로 구분 가능해야 함).
    df = pd.DataFrame({"a": range(50), "b": range(50), "target": [0, 1] * 25})
    profile = build_dataset_profile(df, _MANIFEST, target_columns=["target"])
    roles = {c.name: c.role for c in profile.columns}
    assert roles == {"a": "numeric", "b": "numeric", "target": "target"}


def test_metadata_validation_reports_mismatch_without_fixing():
    df = pd.DataFrame({"temp": ["251", "ERROR", "250"], "code": ["A", "B", "A"]})
    metadata = DatasetMetadata(
        columns=[ColumnMetadata("temp", "numeric"), ColumnMetadata("missing_col", "numeric")],
        source="json",
    )
    report = validate_metadata(metadata, df)

    statuses = {c.name: c.status for c in report.columns}
    assert statuses["temp"] == "type_mismatch"
    assert statuses["missing_col"] == "missing_in_data"
    assert statuses["code"] == "not_in_metadata"
    # 검증은 값을 고치지 않는다
    assert list(df["temp"]) == ["251", "ERROR", "250"]


def test_unconfirmed_metadata_type_is_not_used_as_role():
    df = pd.DataFrame({"temp": ["251", "ERROR", "250"] * 20})
    metadata = DatasetMetadata(columns=[ColumnMetadata("temp", "numeric")], source="json")
    report = validate_metadata(metadata, df)
    profile = build_dataset_profile(df, _MANIFEST, metadata=metadata, validation=report)

    # numeric 선언이 검증 실패했으므로 numeric 집합에 들어가면 안 된다
    assert "temp" not in profile.numeric_columns
