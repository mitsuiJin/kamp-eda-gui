"""AI Agent Context용 JSON 출력.

PDF에는 담지 않는 상세 수치(전체 상관행렬, 이상치 인덱스, 전체 범주 빈도 등)와 분석
상태·사유·파라미터를 모두 포함한다.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from eda_report.analyses.base import AnalysisResult


def _default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return str(obj)


def render(results: list[AnalysisResult], output_dir: str, profile=None, run_config=None, column_glossary=None) -> str:
    analyses = [
        {
            "section_id": r.section_id,
            "title": r.title,
            "tier": r.tier,
            "purpose": r.purpose,
            "rationale": r.rationale,
            "status": r.status,
            "status_reason": r.status_reason,
            "input_columns": r.input_columns,
            "parameters": r.parameters,
            "duration_sec": r.duration_sec,
            "observations": [vars(f) for f in r.findings],
            "results": r.ai_context,
        }
        for r in results
    ]

    payload: dict[str, Any] = {
        "schema": "eda_context/v2",
        "note": (
            "이 파일은 EDA가 데이터에서 관찰한 객관적 사실과 분석 실행 상태를 담는다. "
            "도메인 해석·원인 판단·후속 조치 권고는 포함하지 않으며, 그 판단은 이 Context를 "
            "받는 AI Agent 또는 사용자의 몫이다."
        ),
        "analyses": analyses,
    }

    if profile is not None:
        payload["dataset"] = {
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "numeric_columns": profile.numeric_columns,
            "categorical_columns": profile.categorical_columns,
            "datetime_columns": profile.datetime_columns,
            "text_columns": profile.text_columns,
            "excluded_columns": profile.excluded_columns,
            "target_columns": profile.target_columns,
            "column_profiles": [c.to_dict() for c in profile.columns],
            "column_descriptions": column_glossary or {},
        }

    if run_config is not None:
        payload["run_config"] = {
            "input_path": run_config.input_path,
            "target_columns": run_config.target_columns,
            "column_glossary_path": run_config.column_glossary_path,
            "thresholds": run_config.thresholds.to_dict(),
        }

    path = os.path.join(output_dir, "context.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)
    return path
