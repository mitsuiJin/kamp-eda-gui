"""실행 설정과 분석 임계값.

설계 원칙(§10): 임계값은 도메인 규칙이 아니라 "분석이 성립하는 최소 조건"과 "지면 제약"만
담는다. 모든 값은 여기 한 곳에 모으고 실행 시 Manifest/AI Context에 그대로 기록되므로,
어떤 기준으로 분석이 실행/생략됐는지 사후에 추적할 수 있다.

각 값의 근거는 주석에 남긴다. 근거를 댈 수 없는 임의 임계값은 두지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AnalysisThresholds:
    # --- 데이터 구조 판정 ---
    # 문자열 컬럼을 datetime으로 인정할 파싱 성공률. 거의 모든 행이 날짜로 해석돼야만
    # "시간 변수"로 취급한다(추정이 아니라 검증에 가깝게 보수적으로 설정).
    datetime_parse_min_rate: float = 0.99
    # 범주형으로 "집계·시각화"할 수 있는 최대 고유값 수. 통계적 기준이 아니라 표/그래프
    # 가독성 한계이며, 초과 시 해당 분석은 NOT_APPLICABLE로 기록된다.
    categorical_max_cardinality: int = 50

    # --- 분석 성립 최소 조건 ---
    min_rows_for_analysis: int = 30  # 표본이 이보다 적으면 분포/관계 통계의 의미가 희박
    correlation_min_numeric: int = 2  # 상관계수는 변수 2개 이상에서만 정의됨
    max_crosstab_cells: int = 10_000  # 교차표 칸 수(범주 수의 곱)가 이를 넘으면 해당 쌍은 건너뜀

    # --- 표시(지면) 제약 ---
    categorical_display_top_n: int = 20  # 범주가 이보다 많으면 Top-N만 표시(전체는 Context에 기록)
    top_correlation_pairs: int = 8
    scatter_max_pairs: int = 8
    timeseries_max_series: int = 4

    # --- 보고 기준(판단이 아니라 표시 임계값) ---
    high_correlation_threshold: float = 0.7  # 상관계수를 "높음"으로 표시할 기준
    high_skew_threshold: float = 1.0  # 왜도를 관찰 사실로 기록할 기준

    # --- 시계열 ---
    # rolling window를 고정값으로 두지 않고 데이터 길이에서 산출한다(기록됨).
    rolling_window_ratio: float = 0.01  # 전체 길이의 1%
    rolling_window_min: int = 5
    rolling_window_max: int = 500
    # 연속된 타임스탬프 간격 중 이 비율 이상이 동일해야 "수집 주기가 검증됨"으로 인정하고
    # 이동평균 구간을 시간 단위로도 표시한다(datetime_parse_min_rate와 같은 원칙 — 추정이
    # 아니라 데이터 자체에서 검증). 간격이 이보다 덜 일정하면 관측치 개수로만 표현한다.
    timeseries_interval_uniform_min_rate: float = 0.95

    def to_dict(self) -> dict:
        return asdict(self)

    def rolling_window(self, n_rows: int) -> int:
        window = int(n_rows * self.rolling_window_ratio)
        return max(self.rolling_window_min, min(window, self.rolling_window_max))


@dataclass
class RunConfig:
    input_path: str
    output_dir: str
    target_columns: list[str] | None = None
    # 사람이 검수한 컬럼 설명 JSON(선택). 순수 표시용이며 type/target/시간축 판단에는 쓰이지
    # 않는다 — eda_report.column_glossary.extract_glossary_draft()로 초안을 만든 뒤 검수한다.
    column_glossary_path: str | None = None
    formats: list[str] = field(default_factory=lambda: ["pdf", "markdown", "json"])
    thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)
