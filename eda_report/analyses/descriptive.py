"""04. Descriptive Statistics — 수치형 변수의 기술통계를 산출한다.

설계 원칙 §3-[2]: 왜도(skewness)는 객관적 통계량으로만 제시한다. 왜도가 크다는 이유로
로그 변환 등 특정 변환을 권장하지 않는다 — 제조 데이터에서는 clipping, saturation,
제어값, 운전 모드 때문에 큰 왜도가 정상적으로 나타날 수 있고, 변환 필요 여부는 후속
분석 단계(AI Agent/사용자)의 판단 영역이다.
"""

from __future__ import annotations

import pandas as pd

from eda_report.analyses.base import AnalysisResult, Finding, round_floats
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = profile.numeric_columns
    sub = df[columns]

    desc = sub.agg(["count", "mean", "std", "min", "median", "max"]).T
    desc.insert(4, "q1", sub.quantile(0.25))
    desc.insert(6, "q3", sub.quantile(0.75))
    desc["skew"] = sub.skew()
    desc["kurtosis"] = sub.kurtosis()
    desc["missing_rate"] = sub.isna().mean()
    desc = desc.reset_index().rename(columns={"index": "column"})

    findings = [
        Finding(
            flag_type="high_skew",
            columns=[row["column"]],
            metric=float(row["skew"]),
            severity="info",
            message_ko=(
                f"{row['column']}: 왜도 {row['skew']:.2f}"
                f"({'왼쪽' if row['skew'] < 0 else '오른쪽'}으로 긴 꼬리를 가진 비대칭 분포로 관찰됨)"
            ),
        )
        for _, row in desc.iterrows()
        if pd.notna(row["skew"]) and abs(row["skew"]) >= thresholds.high_skew_threshold
    ]

    return AnalysisResult(
        section_id="04_descriptive",
        title="Descriptive Statistics (기술통계)",
        purpose=(
            "기술통계는 변수의 값 분포를 몇 가지 숫자로 요약한 것입니다. 평균·중앙값·표준편차·"
            "사분위수로 값의 중심과 퍼짐 정도를, 왜도·첨도로 분포의 치우침과 꼬리 형태를 확인할 "
            "수 있습니다."
        ),
        rationale="왜도·첨도 값이 크다고 데이터 오류이거나 변환이 필요하다는 뜻은 아닙니다.",
        input_columns=columns,
        parameters={"skew_report_threshold": thresholds.high_skew_threshold},
        tables=[round_floats(desc)],
        findings=findings,
        ai_context={"descriptive_stats": desc.set_index("column").to_dict(orient="index")},
    )
