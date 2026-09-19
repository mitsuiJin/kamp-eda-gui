"""17. Association Analysis — 범주 값 조합의 동시 출현 패턴을 산출한다.

실행 조건은 selection/rules.py에서 판정한다(범주형 2개 이상 + 카디널리티 제한 + 조합 수 제한).
조건을 만족하지 못해 생략되는 것은 정상 동작이며, 사유는 Manifest에 기록된다.
"""

from __future__ import annotations

import pandas as pd

from eda_report.analyses.base import AnalysisResult, round_floats, skipped
from eda_report.config import AnalysisThresholds
from eda_report.profiling.dataset_profile import DatasetProfile

_SECTION_ID = "17_association"
_TITLE = "Association Analysis (연관규칙)"
_PURPOSE = (
    "여러 범주형 변수의 값 조합 중 함께 나타나는 빈도가 높은 패턴을 지지도·신뢰도·향상도로 "
    "산출합니다."
)
TOP_N_RULES = 10


def run(df: pd.DataFrame, profile: DatasetProfile, params: dict) -> AnalysisResult:
    thresholds: AnalysisThresholds = params["thresholds"]
    columns = params.get("columns") or profile.categorical_columns

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        return skipped(_SECTION_ID, _TITLE, _PURPOSE, "mlxtend 패키지가 설치되어 있지 않아 수행하지 않음")

    transactions = (
        df[columns].astype(str).apply(lambda row: [f"{c}={v}" for c, v in zip(columns, row)], axis=1).tolist()
    )
    encoder = TransactionEncoder()
    onehot = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
    frequent = apriori(onehot, min_support=thresholds.association_min_support, use_colnames=True)
    if frequent.empty:
        return skipped(
            _SECTION_ID, _TITLE, _PURPOSE,
            f"지지도 {thresholds.association_min_support:.0%} 이상인 빈발 조합이 없어 규칙을 산출하지 못함",
        )

    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    if rules.empty:
        return skipped(_SECTION_ID, _TITLE, _PURPOSE, "향상도(lift) 1.0 이상인 규칙이 없음")

    rules = rules.sort_values("lift", ascending=False).head(TOP_N_RULES)
    display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    display["antecedents"] = display["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    display["consequents"] = display["consequents"].apply(lambda s: ", ".join(sorted(s)))

    max_lift = float(display["lift"].max())
    rationale = (
        f"대상 컬럼: {columns} (고유값 {thresholds.association_max_cardinality}개 이하인 범주형만 사용). "
        "support는 그 조합이 전체에서 관찰된 비율, confidence는 좌변이 나타났을 때 우변도 나타난 "
        "비율, lift는 두 항목이 독립일 때 대비 몇 배로 함께 나타나는지를 뜻합니다(1이면 독립과 동일). "
        f"이번 결과의 최대 lift는 {max_lift:.3f}로 관찰됐습니다."
    )

    return AnalysisResult(
        section_id=_SECTION_ID,
        title=_TITLE,
        purpose=_PURPOSE,
        rationale=rationale,
        input_columns=list(columns),
        parameters={
            "min_support": thresholds.association_min_support,
            "metric": "lift",
            "min_lift": 1.0,
            "rules_shown": int(len(display)),
            "combinations": params.get("combinations"),
        },
        tables=[round_floats(display)],
        ai_context={"rules": display.to_dict(orient="records"), "columns_used": list(columns), "max_lift": max_lift},
    )
