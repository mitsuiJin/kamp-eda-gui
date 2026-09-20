"""컬럼명 설명(사람이 읽을 텍스트)만 다루는 선택적 부가 기능.

중요한 경계: 이 모듈이 다루는 정보는 오직 "컬럼명 → 설명 문자열"이다. 변수 타입·target·
시간축처럼 EDA 실행 흐름(role 판정, 분석 대상 선정)에 영향을 주는 판단은 절대 하지 않는다
— 그 부분은 이미 삭제된 `metadata/` 패키지가 하던 일이며, 이번 기능은 그와 무관하게 순수
표시(display)용 주석만 추가한다.

두 단계로 나뉜다:
1. `extract_glossary_draft()` — Dataset Guidebook PDF에서 초안을 뽑는다. 컬럼명이 실제
   데이터에 존재하는 경우에만 항목을 만든다(Hallucination 방지: 없는 컬럼을 지어내지 않음).
   이 초안은 **검증되지 않은 추출 결과**이며 그대로 쓰지 않는다.
2. 사람이 초안 JSON을 검수·수정한다(이 모듈이 하지 않는 단계).
3. `load_column_glossary()` — 사람이 검수한 최종 JSON을 읽어 파이프라인에 전달한다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_DESCRIPTION_CHARS = 200
MAX_SPAN_CHARS = 200
MAX_PDF_PAGES = 100


class GlossaryExtractionError(RuntimeError):
    """PDF를 열 수 없거나 텍스트를 추출할 수 없을 때 발생한다."""


def _load_pdf_pages(pdf_path: str | Path) -> list[str]:
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise GlossaryExtractionError(
            "PDF 텍스트 추출에는 pymupdf가 필요합니다(pip install pymupdf). 이 기능은 초안 추출 "
            "도구에서만 필요하며, EDA 파이프라인 실행 자체에는 필요하지 않습니다."
        ) from exc

    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise GlossaryExtractionError(f"PDF를 열 수 없습니다: {exc}") from exc

    with document:
        return [document.load_page(i).get_text() for i in range(min(document.page_count, MAX_PDF_PAGES))]


def _column_occurrences(text: str, data_columns: list[str]) -> list[tuple[int, int, str]]:
    """실제 데이터 컬럼명이 본문에 등장하는 위치를 모두 찾는다(컬럼명을 지어내지 않는다)."""
    occurrences: list[tuple[int, int, str]] = []
    for column in data_columns:
        if len(column) < 2:
            continue
        pattern = re.compile(rf"(?<![\w.]){re.escape(column)}(?![\w.])")
        occurrences.extend((m.start(), m.end(), column) for m in pattern.finditer(text))
    occurrences.sort()
    return occurrences


def extract_glossary_draft(pdf_path: str | Path, data_columns: list[str]) -> dict:
    """Dataset Guidebook PDF에서 컬럼 설명 초안을 뽑는다. 사람의 검수 없이 그대로 쓰지 않는다."""
    pages = _load_pdf_pages(pdf_path)
    full_text = "\n".join(pages)
    occurrences = _column_occurrences(full_text, data_columns)

    # 컬럼명이 문서에 여러 번 등장하면(설명표 이후 통계표·본문에서 재언급) 가장 먼저 등장한
    # 곳을 쓴다 — 실측 결과 Guidebook은 컬럼을 표로 먼저 정의하고 뒤에서 재사용하는 구조라,
    # "가장 긴 텍스트"보다 "최초 등장"이 실제 설명을 담고 있을 확률이 훨씬 높았다.
    best: dict[str, str] = {}
    for index, (start, end, column) in enumerate(occurrences):
        if column in best:
            continue
        next_start = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(full_text)
        span = full_text[end : min(next_start, end + MAX_SPAN_CHARS)]
        description = " ".join(span.split())[:MAX_DESCRIPTION_CHARS]
        if description:
            best[column] = description

    unmatched = [c for c in data_columns if c not in best]
    return {
        "descriptions": best,
        "unmatched_columns": unmatched,
        "source_pdf": str(pdf_path),
        "page_count": len(pages),
        "note": (
            "이 파일은 PDF에서 자동 추출한 초안입니다(검증되지 않음). "
            "descriptions를 사람이 직접 확인·수정한 뒤 --column-glossary로 사용하세요."
        ),
    }


def load_column_glossary(path: str | Path) -> dict[str, str]:
    """사람이 검수한 최종 글로서리 JSON을 읽는다.

    {"컬럼명": "설명", ...} 형태와 extract_glossary_draft()가 만드는
    {"descriptions": {...}, ...} 형태를 모두 받아들인다(검수 후 그대로 재사용 가능하도록).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("descriptions"), dict):
        data = data["descriptions"]

    if not isinstance(data, dict):
        raise ValueError(f"컬럼 글로서리 형식이 올바르지 않습니다(dict가 아님): {path}")

    return {str(k): str(v) for k, v in data.items() if v}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Dataset Guidebook PDF에서 컬럼 설명 초안을 추출합니다(사람 검수 필요)."
    )
    parser.add_argument("--pdf", required=True, help="Dataset Guidebook PDF 경로")
    parser.add_argument("--input", required=True, help="실제 데이터 CSV/TXT 경로(컬럼명 대조용)")
    parser.add_argument("--output", required=True, help="초안 JSON 저장 경로")
    args = parser.parse_args()

    from eda_report.io.loader import load_table

    df, _ = load_table(args.input)
    try:
        draft = extract_glossary_draft(args.pdf, list(df.columns))
    except GlossaryExtractionError as exc:
        print(f"[추출 실패] {exc}")
        sys.exit(2)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)

    print(
        f"초안 저장: {args.output} "
        f"(매칭 {len(draft['descriptions'])}개 / 미매칭 {len(draft['unmatched_columns'])}개) "
        "— 반드시 사람이 검수한 뒤 --column-glossary로 사용하세요."
    )


if __name__ == "__main__":
    main()
