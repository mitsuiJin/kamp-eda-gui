"""AI Agent Context용 JSON 출력.

PDF에는 담지 않는 상세 수치(전체 상관행렬, PCA loading, 이상치 인덱스, 전체 범주 빈도 등)와
분석 상태·사유·파라미터를 모두 포함한다. Dataset Guidebook Context(도메인 설명)와 EDA Context(관찰
사실)를 분리해 담아, 후속 AI Agent가 둘을 구분해 사용할 수 있게 한다.
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


def render(results: list[AnalysisResult], output_dir: str, profile=None, run_config=None) -> str:
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
        metadata = profile.metadata
        validation = profile.validation
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
        }
        payload["dataset_guidebook"] = {
            "source": metadata.source if metadata else "none",
            "dataset_name": metadata.dataset_name if metadata else None,
            "process": metadata.process if metadata else None,
            "columns": [c.to_dict() for c in metadata.columns] if metadata else [],
            "target_columns": metadata.target_columns if metadata else [],
            "datetime_column": metadata.datetime_column if metadata else None,
            "sampling_interval": metadata.sampling_interval if metadata else None,
            "collection_period": metadata.collection_period if metadata else None,
            # 도메인 문맥은 EDA가 사용하지 않고 AI Agent 해석용으로만 보존한다.
            "domain_context": metadata.domain_context if metadata else {},
        }
        payload["metadata_validation"] = validation.to_dict() if validation else None

    if run_config is not None:
        payload["run_config"] = {
            "input_path": run_config.input_path,
            "metadata_path": run_config.metadata_path,
            "guideline_pdf_path": run_config.guideline_pdf_path,
            "target_columns": run_config.target_columns,
            "thresholds": run_config.thresholds.to_dict(),
        }

    path = os.path.join(output_dir, "context.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)
    return path
