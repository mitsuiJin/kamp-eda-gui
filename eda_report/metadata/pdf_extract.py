"""Dataset Guidebook PDF에서 EDA 실행 전에 쓸 수 있는 구조 정보를 뽑아 메타데이터 초안을 만든다.

Hallucination 차단 설계(원칙 B):
1. 컬럼명을 생성하지 않는다. **실제 CSV/TXT에 존재하는 컬럼명이 PDF 본문에 등장하는 경우에만**
   해당 항목을 만든다. 따라서 데이터에 없는 컬럼이 메타데이터에 생길 수 없다.
2. 타입도 추정하지 않는다. 컬럼명 바로 뒤 구간에 타입 토큰(int/float/timestamp 등)이 실제로
   적혀 있을 때만 선언 타입으로 삼고, 애매한 토큰(string/varchar 등 범주형인지 자유 텍스트인지
   알 수 없는 경우)은 unknown으로 남긴다.
3. 여기서 뽑은 타입은 "Dataset Guidebook이 그렇게 적었다"는 사실일 뿐이며, 실제 데이터와의 일치 여부는
   metadata/validator.py가 따로 검증한다.

공정 설명·장비·수집 배경 같은 도메인 문맥은 domain_context에 원문 그대로 보존해 AI Context로만
전달한다(EDA는 이 값을 읽지 않는다).
"""

from __future__ import annotations

import re
from pathlib import Path

from eda_report.metadata.schema import ColumnMetadata, DatasetMetadata

MAX_DESCRIPTION_CHARS = 120
MAX_SPAN_CHARS = 200
MAX_CONTEXT_PAGES = 100

# Dataset Guidebook 표에 실제로 적히는 타입 토큰 → 스키마 타입.
# string/varchar류는 범주형인지 자유 텍스트인지 Dataset Guidebook만으로 알 수 없어 unknown으로 둔다
# (그 구분은 EDA가 dtype과 고유값 수라는 객관적 사실로 판정한다).
_TYPE_TOKENS = {
    "int": "numeric",
    "integer": "numeric",
    "bigint": "numeric",
    "smallint": "numeric",
    "float": "numeric",
    "double": "numeric",
    "decimal": "numeric",
    "numeric": "numeric",
    "number": "numeric",
    "real": "numeric",
    "timestamp": "datetime",
    "datetime": "datetime",
    "date": "datetime",
    "time": "datetime",
    "category": "categorical",
    "categorical": "categorical",
    "범주형": "categorical",
    "수치형": "numeric",
}

_TYPE_PATTERN = re.compile(
    r"(?<![\w가-힣])(" + "|".join(sorted(_TYPE_TOKENS, key=len, reverse=True)) + r")(?![\w가-힣])",
    re.IGNORECASE,
)


class GuidelinePdfError(RuntimeError):
    """PDF를 열 수 없거나 텍스트를 추출할 수 없을 때 발생한다."""


def _load_pages(pdf_path: str | Path) -> list[str]:
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise GuidelinePdfError("PDF 텍스트 추출에는 pymupdf가 필요합니다(pip install pymupdf).") from exc

    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise GuidelinePdfError(f"PDF를 열 수 없습니다: {exc}") from exc

    with document:
        return [document.load_page(i).get_text() for i in range(min(document.page_count, MAX_CONTEXT_PAGES))]


def _column_occurrences(text: str, data_columns: list[str]) -> list[tuple[int, int, str]]:
    """실제 데이터 컬럼명이 본문에 등장하는 위치를 모두 찾는다."""
    occurrences: list[tuple[int, int, str]] = []
    for column in data_columns:
        if len(column) < 2:
            continue
        pattern = re.compile(rf"(?<![\w.]){re.escape(column)}(?![\w.])")
        occurrences.extend((m.start(), m.end(), column) for m in pattern.finditer(text))
    occurrences.sort()
    return occurrences


def _parse_entry(span: str) -> tuple[str | None, str | None, str | None]:
    """컬럼명 뒤 구간에서 (선언 타입, 원문 타입 토큰, 설명)을 뽑는다."""
    match = _TYPE_PATTERN.search(span)
    if not match:
        description = " ".join(span.split())[:MAX_DESCRIPTION_CHARS]
        return None, None, description or None

    raw_token = match.group(1)
    declared = _TYPE_TOKENS[raw_token.lower()]
    description = " ".join(span[: match.start()].split())[:MAX_DESCRIPTION_CHARS]
    return declared, raw_token, description or None


def extract_guideline_metadata(pdf_path: str | Path, data_columns: list[str]) -> DatasetMetadata:
    pages = _load_pages(pdf_path)
    full_text = "\n".join(pages)
    occurrences = _column_occurrences(full_text, data_columns)

    best: dict[str, dict] = {}
    for index, (start, end, column) in enumerate(occurrences):
        next_start = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(full_text)
        span = full_text[end : min(next_start, end + MAX_SPAN_CHARS)]
        declared, raw_token, description = _parse_entry(span)
        entry = {"declared": declared, "raw_token": raw_token, "description": description}
        # 타입 토큰이 함께 적힌 등장 위치를 우선한다(변수 설명 표일 가능성이 높음).
        if column not in best or (declared and not best[column]["declared"]):
            best[column] = entry

    columns = [
        ColumnMetadata(
            name=name,
            declared_type=entry["declared"] or "unknown",
            description=entry["description"],
            notes=(
                f"Dataset Guidebook 표기 타입 '{entry['raw_token']}'에서 추출(미검증)"
                if entry["raw_token"]
                else "PDF에서 컬럼명 언급만 확인(타입 미확인)"
            ),
        )
        for name, entry in best.items()
    ]

    datetime_candidates = [c.name for c in columns if c.declared_type == "datetime"]
    first_line = next((line.strip() for line in full_text.splitlines() if line.strip()), None)

    return DatasetMetadata(
        dataset_name=first_line[:120] if first_line else None,
        columns=columns,
        # 시간축은 Dataset Guidebook이 datetime으로 적은 컬럼이 정확히 하나일 때만 채운다.
        datetime_column=datetime_candidates[0] if len(datetime_candidates) == 1 else None,
        domain_context={
            "guideline_pdf_path": str(pdf_path),
            "page_count": len(pages),
            "matched_columns": len(columns),
            "unmatched_columns": [c for c in data_columns if c not in best],
            "datetime_candidates": datetime_candidates,
            "text_by_page": {str(i): text for i, text in enumerate(pages, start=1)},
        },
        source="pdf_draft_unverified",
    )
