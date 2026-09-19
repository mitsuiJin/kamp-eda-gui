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
            "수치형 변수의 개수·평균·중앙값·표준편차·사분위수·최소/최대값과 분포 형태 지표"
            "(왜도, 첨도)를 산출합니다."
        ),
        rationale=(
            "q1/q3은 하위 25%·75% 지점의 값이고, skew(왜도)는 분포가 좌우 대칭에서 벗어난 정도, "
            "kurtosis(첨도)는 꼬리가 두꺼운 정도입니다. 값이 크다고 해서 데이터 오류이거나 특정 "
            "변환이 필요하다는 뜻은 아니며, 여기서는 관찰된 수치만 제시합니다."
        ),
        input_columns=columns,
        parameters={"skew_report_threshold": thresholds.high_skew_threshold},
        tables=[round_floats(desc)],
        findings=findings,
        ai_context={"descriptive_stats": desc.set_index("column").to_dict(orient="index")},
    )
