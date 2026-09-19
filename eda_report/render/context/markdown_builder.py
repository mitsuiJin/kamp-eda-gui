"""AI Agent Context용 Markdown 출력(사람도 빠르게 훑어볼 수 있는 형태)."""

from __future__ import annotations

import os

import pandas as pd

from eda_report.analyses.base import AnalysisResult

MAX_TABLE_ROWS = 30

_STATUS_LABEL = {
    "SUCCESS": "실행됨",
    "SKIPPED": "생략됨",
    "NOT_APPLICABLE": "해당 없음",
    "FAILED": "실패",
}
_TIER_LABEL = {"core": "Core EDA", "advanced": "Advanced/Optional EDA"}


def _table_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(빈 표)_"
    shown = df.head(MAX_TABLE_ROWS)
    try:
        return shown.to_markdown(index=False)
    except ImportError:
        return shown.to_string(index=False)


def render(results: list[AnalysisResult], output_dir: str) -> str:
    lines = [
        "# EDA 관찰 결과 (AI Agent Context)",
        "",
        "> 이 문서는 데이터에서 관찰된 객관적 사실만 담습니다. 원인 해석·도메인 판단·후속 조치 "
        "권고는 포함하지 않습니다.",
        "",
    ]

    for result in results:
        lines.append(f"## {result.section_id} — {result.title}")
        lines.append("")
        lines.append(f"- **구분**: {_TIER_LABEL.get(result.tier, result.tier)}")
        lines.append(f"- **상태**: {_STATUS_LABEL.get(result.status, result.status)}")
        if result.status_reason:
            lines.append(f"- **사유**: {result.status_reason}")
        lines.append(f"- **목적**: {result.purpose}")
        if result.parameters:
            lines.append(f"- **파라미터**: `{result.parameters}`")
        if result.rationale:
            lines.append("")
            lines.append(result.rationale)

        if result.status != "SUCCESS":
            lines.append("")
            continue

        if result.findings:
            lines.append("")
            lines.append("**관찰 사실**")
            for finding in result.findings:
                lines.append(f"- {finding.message_ko}")

        for table in result.tables:
            if isinstance(table, pd.DataFrame) and not table.empty:
                lines.append("")
                lines.append(_table_to_markdown(table))

        lines.append("")

    path = os.path.join(output_dir, "context.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
