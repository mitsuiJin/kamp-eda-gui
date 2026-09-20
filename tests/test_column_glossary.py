"""컬럼 설명(글로서리) 기능에 대한 단위 테스트.

핵심 경계 확인: (1) 실제 데이터에 없는 컬럼명은 절대 지어내지 않는다(Hallucination 방지),
(2) 최종 로더는 role/target 판정에 전혀 관여하지 않고 순수 문자열 매핑만 돌려준다.
"""

import json

import pytest

from eda_report.column_glossary import extract_glossary_draft, load_column_glossary

pymupdf = pytest.importorskip("pymupdf")


def _make_pdf(path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_extract_glossary_draft_matches_only_columns_present_in_data(tmp_path):
    # 한글 폰트 임베딩 없이는 pymupdf가 테스트 PDF에 CJK를 제대로 못 그려서, 매칭 로직 자체를
    # 검증하는 데는 영향 없는 영문 설명으로 픽스처를 구성한다.
    pdf_path = tmp_path / "guideline.pdf"
    _make_pdf(
        pdf_path,
        "EX1.MD_TQ screw torque value (kgf cm)\nEX1.MELT_P_PV melt pressure sensor reading (bar)",
    )

    data_columns = ["EX1.MD_TQ", "EX1.MELT_P_PV", "NOT_IN_PDF_COLUMN"]
    draft = extract_glossary_draft(pdf_path, data_columns)

    assert "EX1.MD_TQ" in draft["descriptions"]
    assert "torque" in draft["descriptions"]["EX1.MD_TQ"]
    assert "EX1.MELT_P_PV" in draft["descriptions"]
    # 실제 데이터에 없는 컬럼명은 PDF에 있어도 애초에 조회 대상이 아니다(지어내지 않음).
    assert "NOT_IN_PDF_COLUMN" in draft["unmatched_columns"]
    assert "NOT_IN_PDF_COLUMN" not in draft["descriptions"]


def test_extract_glossary_draft_does_not_invent_columns_absent_from_pdf(tmp_path):
    pdf_path = tmp_path / "guideline.pdf"
    _make_pdf(pdf_path, "This document has nothing to do with any column.")

    draft = extract_glossary_draft(pdf_path, ["some_column"])
    assert draft["descriptions"] == {}
    assert draft["unmatched_columns"] == ["some_column"]


def test_load_column_glossary_accepts_flat_dict(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"a": "설명A", "b": "설명B"}, ensure_ascii=False), encoding="utf-8")

    result = load_column_glossary(path)
    assert result == {"a": "설명A", "b": "설명B"}


def test_load_column_glossary_accepts_draft_wrapper_format(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(
        json.dumps({"descriptions": {"a": "설명A"}, "unmatched_columns": ["b"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = load_column_glossary(path)
    assert result == {"a": "설명A"}


def test_load_column_glossary_drops_empty_descriptions(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"a": "설명A", "b": ""}, ensure_ascii=False), encoding="utf-8")

    result = load_column_glossary(path)
    assert result == {"a": "설명A"}
