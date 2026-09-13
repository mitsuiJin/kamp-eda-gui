import pytest

from src.data_loader import detect_encoding, load_csv, preview_lines


def test_detect_encoding_utf8():
    raw = "이름,값\n온도,10\n".encode("utf-8")
    assert detect_encoding(raw) == "utf-8"


def test_detect_encoding_falls_back_to_cp949():
    raw = "이름,값\n온도,10\n".encode("cp949")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert detect_encoding(raw) == "cp949"


def test_detect_encoding_returns_none_when_undecodable():
    raw = b"\xff\xfe\x00\xff"
    assert detect_encoding(raw) is None


def test_load_csv_default_header():
    raw = "a,b\n1,2\n3,4\n".encode("utf-8")
    df = load_csv(raw, "utf-8", header_row=0)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_load_csv_no_header():
    raw = "1,2\n3,4\n".encode("utf-8")
    df = load_csv(raw, "utf-8", header_row=None)
    assert list(df.columns) == [0, 1]
    assert df.shape == (2, 2)


def test_load_csv_multi_row_header():
    # 37_주조_품질보증처럼 1행이 카테고리 태그, 2행이 실제 컬럼명인 구조
    raw = "Process,Process\nid,Shot\n1,1\n2,2\n".encode("utf-8")
    df = load_csv(raw, "utf-8", header_row=1)
    assert list(df.columns) == ["id", "Shot"]
    assert df.shape == (2, 2)


def test_preview_lines_limits_count():
    raw = "\n".join(str(i) for i in range(10)).encode("utf-8")
    assert preview_lines(raw, "utf-8", n=5) == ["0", "1", "2", "3", "4"]
