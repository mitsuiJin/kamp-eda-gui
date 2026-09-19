"""matplotlib 한글 폰트 설정 — import 시 한 번 전역 적용된다.

차트를 그리는 모든 analyses/*.py는 `import eda_report.render.mpl_style  # noqa: F401`로
이 모듈을 먼저 import해 한글 라벨이 깨지지 않게 한다.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_KOREAN_FONT_CANDIDATES = ["Malgun Gothic", "NanumGothic", "AppleGothic"]


def _apply_korean_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


_apply_korean_font()
