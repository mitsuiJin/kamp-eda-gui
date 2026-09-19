"""eda_report.io.loader의 인코딩/구분자/헤더 자동 감지에 대한 단위 테스트.

각 테스트는 docs/eda_auto_report_design.md §0-1에서 실측으로 확인한 실제 엣지케이스
(헤더 없음=03_CNC/13_회전기계, 2행 헤더=37_주조_품질보증, 공백구분 TXT=02_FordEngine,
설명 산문 TXT=02_FordEngine의 FordA.txt)를 축소 재현한다.
"""

import pytest

from eda_report.io.loader import NotTabularDataError, load_table


def test_normal_csv_uses_first_row_as_header(tmp_path):
    path = tmp_path / "normal.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    df, manifest = load_table(str(path))

    assert list(df.columns) == ["a", "b", "c"]
    assert manifest.header_row == 0
    assert manifest.delimiter == ","
    assert not manifest.generated_column_names
    assert manifest.n_rows == 2


def test_headerless_numeric_csv_gets_generated_columns(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text("1.0,0.0\n0.0,1.0\n1.0,1.0\n", encoding="utf-8")

    df, manifest = load_table(str(path))

    assert list(df.columns) == ["var_0", "var_1"]
    assert manifest.header_row is None
    assert manifest.generated_column_names
    assert manifest.n_rows == 3
    assert any("헤더가 없는" in w for w in manifest.warnings)


def test_two_row_header_uses_second_row_as_columns(tmp_path):
    content = (
        "Process,Process,Sensor,Sensor,Defects\n"
        "id,Product_Type,Velocity_1,Velocity_2,Short_Shot\n"
        "1,1,0.144,0.17,0\n"
        "2,2,0.141,0.172,1\n"
    )
    path = tmp_path / "two_row_header.csv"
    path.write_text(content, encoding="utf-8")

    df, manifest = load_table(str(path))

    assert list(df.columns) == ["id", "Product_Type", "Velocity_1", "Velocity_2", "Short_Shot"]
    assert manifest.header_row == 1
    assert manifest.n_rows == 2
    assert any("2행" in w for w in manifest.warnings)


def test_whitespace_delimited_txt_is_readable(tmp_path):
    content = (
        "  -1.0000000e+00  -7.9717168e-01  -6.6439208e-01  -3.7301463e-01\n"
        "   1.0000000e+00   5.2693599e-01   9.8428794e-01   1.3531202e+00\n"
    )
    path = tmp_path / "ford_style.txt"
    path.write_text(content, encoding="utf-8")

    df, manifest = load_table(str(path))

    assert manifest.delimiter == r"\s+"
    assert manifest.header_row is None
    assert df.shape == (2, 4)


def test_cp949_encoded_korean_header_is_decoded(tmp_path):
    path = tmp_path / "korean.csv"
    path.write_bytes("이름,수치\n가,1\n나,2\n".encode("cp949"))

    df, manifest = load_table(str(path))

    assert manifest.encoding == "cp949"
    assert list(df.columns) == ["이름", "수치"]
    assert manifest.n_rows == 2


def test_prose_text_file_raises_not_tabular_error(tmp_path):
    content = (
        "This data was originally used in a competition organized by a large group.\n"
        "The classification problem is to diagnose whether a certain symptom exists.\n"
        "Each case consists of many measurements collected from an automotive part.\n"
        "There are two separate problems described across several following sections.\n"
    )
    path = tmp_path / "description.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(NotTabularDataError):
        load_table(str(path))
