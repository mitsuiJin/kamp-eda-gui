"""PDF 레이아웃 상수와 한글 폰트 등록(design.md §5).

Windows 기본 글꼴(맑은 고딕)을 등록한다 — reportlab 내장 폰트는 한글을 지원하지 않는다.
경로를 못 찾으면 Helvetica로 폴백하되(한글이 깨질 수 있음) 파이프라인을 죽이지 않는다.
"""

from __future__ import annotations

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PAGE_SIZE = A4
MARGIN = 18 * mm

_FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"),
]

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

_registered = False


def register_korean_font() -> None:
    global _registered, FONT_REGULAR, FONT_BOLD
    if _registered:
        return
    _registered = True
    for regular_path, bold_path in _FONT_CANDIDATES:
        if not os.path.exists(regular_path):
            continue
        pdfmetrics.registerFont(TTFont("MalgunGothic", regular_path))
        FONT_REGULAR = "MalgunGothic"
        if os.path.exists(bold_path):
            pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", bold_path))
            FONT_BOLD = "MalgunGothic-Bold"
        else:
            FONT_BOLD = "MalgunGothic"
        return
