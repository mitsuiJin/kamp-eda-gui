"""00. Manifest — 파일을 어떻게 읽었고, 어떤 분석이 어떤 사유로 실행/생략됐는지 기록한다.

설계 원칙 §7: SUCCESS / SKIPPED / NOT_APPLICABLE / FAILED를 구분해 기록한다. 조건을 만족하지
못해 생략하는 것은 오류가 아니라 정상 동작이며, 사유가 남아야 추적할 수 있다.
"""

from __future__ import annotations

import pandas as pd

from eda_report.analyses.base import AnalysisResult
from eda_report.profiling.dataset_profile import DatasetProfile

_STATUS_LABEL = {
    "SUCCESS": "실행",
    "SKIPPED": "생략",
    "NOT_APPLICABLE": "해당 없음",
    "FAILED": "실패",
}
_TIER_LABEL = {"core": "Core EDA", "advanced": "Advanced/Optional EDA"}


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    manifest = profile.parse_manifest
    entries: list[dict] = params.get("entries", [])
    thresholds: dict = params.get("thresholds", {})

    parse_table = pd.DataFrame(
        {
            "항목": ["파일 경로", "인코딩", "구분자", "헤더", "컬럼명", "행 수", "열 수"],
            "값": [
                manifest.file_path,
                manifest.encoding,
                repr(manifest.delimiter),
                "없음(headerless)" if manifest.header_row is None else f"{manifest.header_row}행",
                "자동 생성" if manifest.generated_column_names else "원본 사용",
                f"{manifest.n_rows:,}",
                f"{manifest.n_cols:,}",
            ],
        }
    )

    ordered_entries = sorted(entries, key=lambda e: int(e["section_id"].split("_")[0]))
    status_table = pd.DataFrame(
        [
            {
                "section": entry["section_id"],
                "tier": _TIER_LABEL.get(entry.get("tier", "core"), entry.get("tier", "core")),
                "status": _STATUS_LABEL.get(entry["status"], entry["status"]),
                "reason": entry.get("reason") or "-",
            }
            for entry in ordered_entries
        ]
    )

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    summary = ", ".join(f"{_STATUS_LABEL.get(k, k)} {v}건" for k, v in counts.items())

    rationale_parts = [f"분석 상태 요약: {summary}."]
    rationale_parts.append(
        "Core EDA는 데이터 구조만으로 항상 시도되는 기초 분석이고, Advanced/Optional EDA는 "
        "target·시간축처럼 조건부로만 성립하는 분석입니다 — Advanced가 덜 중요하다는 뜻이 "
        "아니라, 실행 조건이 데이터 구조가 아니라 사용자 지정이나 특정 변수 종류에 달려 있다는 "
        "뜻입니다."
    )
    if manifest.warnings:
        rationale_parts.append("파싱 경고: " + " / ".join(manifest.warnings))
    rationale_parts.append(
        "생략(SKIPPED)과 해당 없음(NOT_APPLICABLE)은 오류가 아니라, 해당 분석이 이 데이터에서 "
        "성립하지 않거나 수행할 필요가 없다고 판정된 정상 결과입니다."
    )

    return AnalysisResult(
        section_id="00_manifest",
        title="Manifest (실행 기록)",
        purpose="이 리포트가 데이터를 어떻게 읽었고 각 분석이 어떤 상태로 처리됐는지 기록합니다.",
        rationale="\n\n".join(rationale_parts),
        parameters={"thresholds": thresholds},
        tables=[parse_table, status_table],
        ai_context={
            "parse_manifest": {
                "file_path": manifest.file_path,
                "encoding": manifest.encoding,
                "delimiter": manifest.delimiter,
                "header_row": manifest.header_row,
                "generated_column_names": manifest.generated_column_names,
                "n_rows": manifest.n_rows,
                "n_cols": manifest.n_cols,
                "warnings": manifest.warnings,
            },
            "analysis_status": entries,
            "status_counts": counts,
            "thresholds": thresholds,
        },
    )
