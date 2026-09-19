"""CSV/TXT 표 파일을 인코딩·구분자·헤더 자동 감지로 읽어들인다.

45개 KAMP 데이터셋 실측(docs/eda_auto_report_design.md §0-1)에서 확인된 3가지 문제에 대응한다:
1. 인코딩 문제 (한글 헤더가 CP949/EUC-KR인 경우)
2. 헤더 없는 파일 (첫 행이 전부 숫자 — 03_CNC, 13_회전기계, 02_FordEngine의 TXT 등)
3. 2행 헤더 (1행이 반복되는 카테고리 태그, 2행이 실제 컬럼명 — 37_주조_품질보증)

의미 판단은 하지 않는다 — 여기서 하는 모든 판정은 구조적 신호(디코딩 성공 여부, 구분자별
행당 등장 횟수, 값이 숫자로 파싱되는지)에만 근거하며, 그 결과는 전부 ParseManifest.warnings에
남겨 사람이 검증할 수 있게 한다.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from eda_report.io.manifest import ParseManifest

_ENCODING_CANDIDATES = ["utf-8-sig", "cp949", "euc-kr"]
_DELIMITER_CANDIDATES = [",", "\t", ";"]
_WHITESPACE_DELIMITER = r"\s+"

_ROW0_DUPLICATION_THRESHOLD = 0.4
_ROW1_DUPLICATION_THRESHOLD = 0.2
_ROW1_TEXT_RATIO_THRESHOLD = 0.8

_NOT_TABULAR_AVG_LEN = 40
_NOT_TABULAR_UNIQUE_RATIO = 0.9


class NotTabularDataError(Exception):
    """지정된 파일이 표 형태의 데이터로 보이지 않을 때 발생한다(예: 설명 문서, 이미지 경로 목록)."""


def _detect_encoding(raw_bytes: bytes) -> tuple[str, list[str]]:
    for encoding in _ENCODING_CANDIDATES:
        try:
            raw_bytes.decode(encoding)
            return encoding, []
        except UnicodeDecodeError:
            continue
    return "utf-8", [
        "인코딩을 자동으로 판별하지 못해 utf-8로 강제 디코딩했습니다(일부 문자가 깨질 수 있습니다)."
    ]


def _looks_numeric(token: str) -> bool:
    token = token.strip()
    if token == "":
        return False
    try:
        float(token)
        return True
    except ValueError:
        return False


def _detect_delimiter(sample_lines: list[str]) -> str:
    non_empty = [ln for ln in sample_lines if ln.strip()]
    if not non_empty:
        return ","

    for delimiter in _DELIMITER_CANDIDATES:
        counts = [line.count(delimiter) for line in non_empty]
        if min(counts) > 0 and len(set(counts)) == 1:
            return delimiter

    field_counts = [len(line.split()) for line in non_empty]
    if len(set(field_counts)) == 1 and field_counts[0] > 1:
        return _WHITESPACE_DELIMITER

    return ","


def _split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == _WHITESPACE_DELIMITER:
        return line.split()
    reader = csv.reader([line], delimiter=delimiter)
    return next(reader)


def _duplication_ratio(fields: list[str]) -> float:
    if not fields:
        return 0.0
    return 1 - (len(set(fields)) / len(fields))


def _detect_header(sample_rows: list[list[str]]) -> tuple[int | None, bool, list[str]]:
    """반환값: (pandas에 넘길 header 행 번호 또는 None, 컬럼명 자동생성 여부, 경고 문장)."""
    if not sample_rows:
        return 0, False, []

    row0 = sample_rows[0]
    if all(_looks_numeric(v) for v in row0):
        return None, True, [
            "첫 행이 전부 숫자로 판단되어 헤더가 없는 파일로 처리했습니다 "
            "(컬럼명을 var_0..var_N으로 자동 생성)."
        ]

    if len(sample_rows) >= 2:
        row1 = sample_rows[1]
        row0_dup = _duplication_ratio(row0)
        row1_dup = _duplication_ratio(row1)
        row1_text_ratio = sum(not _looks_numeric(v) for v in row1) / len(row1) if row1 else 0.0
        if (
            row0_dup >= _ROW0_DUPLICATION_THRESHOLD
            and row1_dup <= _ROW1_DUPLICATION_THRESHOLD
            and row1_text_ratio >= _ROW1_TEXT_RATIO_THRESHOLD
        ):
            return 1, False, [
                "1행은 반복되는 카테고리 태그, 2행이 실제 컬럼명으로 판단되어 2행을 헤더로 "
                "사용했습니다(1행은 건너뜀)."
            ]

    return 0, False, []


def _validate_row_structure(lines: list[str], file_size: int) -> None:
    """행 구분(줄바꿈)이 없는 파일을 조용히 0행 표로 만들지 않고 명확히 실패시킨다.

    실측: 02_FordEngine의 FordA_TRAIN.txt는 28MB 전체가 줄바꿈 없는 한 줄이라, 그대로 읽으면
    컬럼만 있고 행이 0개인 표가 만들어져 이후 분석이 전부 빈 결과로 진행된다. 행 구분자를
    임의로 추정해 보정하지 않고(원칙: 임의 보정 금지) 사유를 밝히며 실패한다.
    """
    if len(lines) <= 1 and file_size > 10_000:
        raise NotTabularDataError(
            f"줄바꿈이 없는 단일 라인 파일입니다(크기 {file_size:,}바이트). 행 구분자를 찾을 수 "
            "없어 표로 해석할 수 없습니다 — 행 길이를 임의로 추정하지 않습니다. 원본 제공 형식"
            "(예: ARFF 등 다른 배포본)이나 전처리된 파일을 사용해주세요."
        )


def _validate_tabular(df: pd.DataFrame) -> None:
    if df.shape[0] == 0:
        raise NotTabularDataError("데이터 행이 0개로 파싱되어 분석할 수 없습니다(파일 구조를 확인해주세요).")
    if df.shape[1] != 1:
        return
    values = df.iloc[:, 0].dropna().astype(str)
    if len(values) == 0:
        raise NotTabularDataError("표로 파싱할 수 있는 데이터가 없습니다.")
    avg_len = values.str.len().mean()
    unique_ratio = values.nunique() / len(values)
    if avg_len > _NOT_TABULAR_AVG_LEN and unique_ratio > _NOT_TABULAR_UNIQUE_RATIO:
        raise NotTabularDataError(
            "이 파일은 표 형태의 데이터로 보이지 않습니다(구분자를 찾지 못했고 각 행이 "
            "자유 텍스트에 가깝습니다). 분석할 표 데이터 파일 경로가 맞는지 확인해주세요."
        )


def load_table(path: str) -> tuple[pd.DataFrame, ParseManifest]:
    raw_bytes = Path(path).read_bytes()
    encoding, warnings = _detect_encoding(raw_bytes)
    text = raw_bytes.decode(encoding, errors="replace")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    _validate_row_structure(lines, len(raw_bytes))
    sample_lines = lines[:10]
    delimiter = _detect_delimiter(sample_lines)

    sample_rows = [_split_row(ln, delimiter) for ln in sample_lines[:2]]
    header_row, generated, header_warnings = _detect_header(sample_rows)
    warnings += header_warnings

    read_kwargs: dict = {"sep": delimiter, "engine": "python"}
    if header_row is None:
        df = pd.read_csv(io.StringIO(text), header=None, **read_kwargs)
        df.columns = [f"var_{i}" for i in range(df.shape[1])]
    else:
        df = pd.read_csv(io.StringIO(text), header=header_row, **read_kwargs)
        df.columns = [str(c).strip() for c in df.columns]

    _validate_tabular(df)

    manifest = ParseManifest(
        file_path=str(path),
        encoding=encoding,
        delimiter=delimiter,
        header_row=header_row,
        generated_column_names=generated,
        n_rows=len(df),
        n_cols=df.shape[1],
        warnings=warnings,
    )
    return df, manifest
