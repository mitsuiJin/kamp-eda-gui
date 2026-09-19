"""AnalysisResult 리스트를 사람이 보는 시각화 중심 PDF로 렌더링한다(design.md §5 레이아웃 원칙).

섹션마다 강제로 페이지를 넘기지 않는다 — 내용이 적은 섹션(예: Overview)은 다음 섹션과 같은
페이지를 공유하고, 내용이 많은 섹션은 자연스럽게 여러 페이지로 넘어간다. 대신 각 섹션은
`KeepTogether`로 묶어 "제목만 이전 페이지 맨 아래, 본문은 다음 페이지"처럼 어색하게 잘리는
것은 막는다(섹션 전체가 한 페이지에 안 들어갈 만큼 길면 자동으로 그냥 흘려보낸다).
"""

from __future__ import annotations

import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from eda_report.analyses.base import AnalysisResult
from eda_report.render.pdf import layout

MAX_TABLE_ROWS = 20
MAX_TABLE_COLS = 8
MAX_FINDINGS_SHOWN = 8


def _build_styles() -> dict[str, ParagraphStyle]:
    layout.register_korean_font()
    sheet = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "KoTitle", parent=sheet["Heading1"], fontName=layout.FONT_BOLD, fontSize=15,
            spaceBefore=4, spaceAfter=6,
        ),
        "purpose": ParagraphStyle(
            "KoPurpose", parent=sheet["Normal"], fontName=layout.FONT_REGULAR, fontSize=10,
            textColor=colors.HexColor("#333333"), spaceAfter=4, leading=13,
        ),
        "rationale": ParagraphStyle(
            "KoRationale", parent=sheet["Normal"], fontName=layout.FONT_REGULAR, fontSize=9,
            textColor=colors.HexColor("#666666"), spaceAfter=8, leading=12,
        ),
        "finding": ParagraphStyle(
            "KoFinding", parent=sheet["Normal"], fontName=layout.FONT_REGULAR, fontSize=9,
            leftIndent=10, spaceAfter=2, leading=12,
        ),
        "caption": ParagraphStyle(
            "KoCaption", parent=sheet["Normal"], fontName=layout.FONT_REGULAR, fontSize=8,
            textColor=colors.grey, spaceAfter=6,
        ),
        "tier": ParagraphStyle(
            "KoTier", parent=sheet["Normal"], fontName=layout.FONT_REGULAR, fontSize=8,
            textColor=colors.HexColor("#8A6D00"), spaceAfter=4,
        ),
    }


MIN_COL_CHARS = 6
MAX_COL_CHARS = 55


def _table_flowable(df: pd.DataFrame, font_name: str, usable_width: float) -> Table:
    shown = df.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_COLS]
    columns = [str(c) for c in shown.columns]
    str_rows = shown.astype(str).values.tolist()

    header_style = ParagraphStyle(
        "KoCellHeader", fontName=font_name, fontSize=7, leading=9, textColor=colors.white
    )
    cell_style = ParagraphStyle("KoCell", fontName=font_name, fontSize=7, leading=9)

    data = [[Paragraph(c, header_style) for c in columns]] + [
        [Paragraph(v, cell_style) for v in row] for row in str_rows
    ]

    # 긴 텍스트(설명 등)가 셀 밖으로 넘치지 않도록, 컬럼별 최대 글자수에 비례해 폭을 배분한다
    # (raw string을 그대로 Table에 넣으면 줄바꿈이 안 돼 페이지 밖으로 잘려나가는 문제가 있었음).
    max_lens = [
        min(max([len(columns[j])] + [len(row[j]) for row in str_rows]), MAX_COL_CHARS)
        for j in range(len(columns))
    ]
    weights = [max(m, MIN_COL_CHARS) for m in max_lens]
    total_weight = sum(weights)
    col_widths = [usable_width * w / total_weight for w in weights]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


_STATUS_LABEL = {
    "SKIPPED": "생략됨(SKIPPED)",
    "NOT_APPLICABLE": "해당 없음(NOT_APPLICABLE)",
    "FAILED": "실패(FAILED)",
}
_TIER_LABEL = {"core": "Core EDA", "advanced": "Advanced / Optional EDA"}


def _section_flowables(r: AnalysisResult, styles: dict, usable_width: float) -> list:
    flow = [
        Paragraph(f"{r.section_id} — {r.title}", styles["title"]),
        Paragraph(f"[{_TIER_LABEL.get(r.tier, r.tier)}]", styles["tier"]),
        Paragraph(r.purpose, styles["purpose"]),
    ]

    if r.status != "SUCCESS":
        label = _STATUS_LABEL.get(r.status, r.status)
        flow.append(Paragraph(f"<b>[{label}]</b> {r.status_reason or ''}", styles["rationale"]))
        return flow

    if r.rationale:
        flow.append(Paragraph(r.rationale.replace("\n", "<br/>"), styles["rationale"]))

    for fig in r.figures:
        if not os.path.exists(fig.image_path):
            continue
        img = Image(fig.image_path)
        scale = usable_width / img.imageWidth
        img.drawWidth = usable_width
        img.drawHeight = img.imageHeight * scale
        flow.append(img)
        if fig.caption:
            flow.append(Paragraph(fig.caption, styles["caption"]))
        flow.append(Spacer(1, 4 * mm))

    for table_df in r.tables:
        if isinstance(table_df, pd.DataFrame) and not table_df.empty:
            flow.append(_table_flowable(table_df, layout.FONT_REGULAR, usable_width))
            flow.append(Spacer(1, 4 * mm))

    if r.findings:
        flow.append(Paragraph("<b>관찰된 사실</b>", styles["rationale"]))
        for finding in r.findings[:MAX_FINDINGS_SHOWN]:
            flow.append(Paragraph(f"• {finding.message_ko}", styles["finding"]))

    return flow


def render(results: list[AnalysisResult], output_dir: str) -> str:
    styles = _build_styles()
    path = os.path.join(output_dir, "report.pdf")
    doc = SimpleDocTemplate(
        path,
        pagesize=layout.PAGE_SIZE,
        leftMargin=layout.MARGIN,
        rightMargin=layout.MARGIN,
        topMargin=layout.MARGIN,
        bottomMargin=layout.MARGIN,
    )
    usable_width = layout.PAGE_SIZE[0] - 2 * layout.MARGIN

    story = []
    for i, r in enumerate(results):
        if i > 0:
            story.append(Spacer(1, 3 * mm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDDDDD")))
            story.append(Spacer(1, 3 * mm))
        story.append(KeepTogether(_section_flowables(r, styles, usable_width)))

    doc.build(story)
    return path
