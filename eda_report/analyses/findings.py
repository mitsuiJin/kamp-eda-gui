"""18. Key Observations — 각 분석에서 기록된 관찰 사실을 한곳에 모은다.

이전 구현에는 19번 "Feature Engineering Hints"(제거/결합/파생/변환 권장) 섹션이 있었으나,
EDA가 특정 Feature Engineering이나 변환을 권장하지 않는다는 설계 원칙(§4-7)에 따라 삭제했다.
후속 판단(변환·변수 선택·모델 선택)은 AI Agent 또는 사용자의 몫이며, 이 섹션은 그 판단에
필요한 "관찰된 사실"만 제공한다.
"""

from __future__ import annotations

from eda_report.analyses.base import AnalysisResult, Finding

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

_CATEGORY_BY_FLAG = {
    "excluded_column": "데이터 구조",
    "conditional_missing": "결측",
    "high_skew": "분포",
    "iqr_outlier": "이상치(통계적)",
    "iqr_not_defined": "이상치(분석 불가)",
    "high_correlation": "변수 간 관계",
    "class_ratio": "target 구성",
}

MAX_DISPLAYED = 12


def _collect(results: list[AnalysisResult]) -> list[Finding]:
    findings: list[Finding] = []
    for result in results:
        findings.extend(result.findings)
    return findings


def build_key_observations(results: list[AnalysisResult]) -> AnalysisResult:
    findings = _collect(results)
    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 3))

    grouped: dict[str, list[str]] = {}
    for finding in findings:
        category = _CATEGORY_BY_FLAG.get(finding.flag_type, "기타")
        grouped.setdefault(category, []).append(finding.message_ko)

    return AnalysisResult(
        section_id="18_key_observations",
        title="Key Observations (관찰 사실 종합)",
        purpose="앞선 분석들에서 기록된 관찰 사실을 한곳에 모아 확인합니다.",
        rationale=(
            "여기 적힌 내용은 데이터에서 관찰된 사실이며, 원인 해석이나 후속 조치 권고가 아닙니다. "
            f"전체 {len(findings)}건 중 상위 {min(len(findings), MAX_DISPLAYED)}건을 표시했고 전체는 "
            "AI Context(JSON)에 기록했습니다."
        ),
        parameters={"total_findings": len(findings), "displayed": min(len(findings), MAX_DISPLAYED)},
        findings=findings[:MAX_DISPLAYED],
        ai_context={
            "all_observations": [vars(f) for f in findings],
            "observations_by_category": grouped,
        },
    )
