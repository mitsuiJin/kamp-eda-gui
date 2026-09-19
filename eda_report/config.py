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
    pca_min_numeric: int = 3  # 3개 미만이면 차원 축약의 실익이 없음
    clustering_min_numeric: int = 2
    clustering_min_rows: int = 100
    clustering_k_min: int = 2
    clustering_k_max: int = 6  # 실루엣 점수로 k를 고르는 탐색 범위(결과에 k와 점수를 기록)
    multivariate_outlier_min_numeric: int = 3  # 변수 조합 이상치는 3개 이상에서만 의미
    multivariate_outlier_min_rows: int = 500  # 밀도 추정이 불안정해지는 하한
    association_max_cardinality: int = 20  # 조합 폭발 방지
    association_min_support: float = 0.05
    association_max_combinations: int = 10_000  # 카디널리티 곱이 이를 넘으면 SKIP

    # --- 표시(지면) 제약 ---
    max_display_columns: int = 16  # 개별 그래프로 그릴 변수 수 상한
    categorical_display_top_n: int = 20  # 범주가 이보다 많으면 Top-N만 표시(전체는 Context에 기록)
    top_correlation_pairs: int = 8
    scatter_max_pairs: int = 8
    timeseries_max_series: int = 4
    max_displayed_clusters: int = 8
    max_table_rows: int = 30

    # --- 보고 기준(판단이 아니라 표시 임계값) ---
    high_correlation_threshold: float = 0.7  # 상관계수를 "높음"으로 표시할 기준
    high_skew_threshold: float = 1.0  # 왜도를 관찰 사실로 기록할 기준

    # --- 시계열 ---
    # rolling window를 고정값으로 두지 않고 데이터 길이에서 산출한다(기록됨).
    rolling_window_ratio: float = 0.01  # 전체 길이의 1%
    rolling_window_min: int = 5
    rolling_window_max: int = 500

    def to_dict(self) -> dict:
        return asdict(self)

    def rolling_window(self, n_rows: int) -> int:
        window = int(n_rows * self.rolling_window_ratio)
        return max(self.rolling_window_min, min(window, self.rolling_window_max))


@dataclass
class RunConfig:
    input_path: str
    output_dir: str
    metadata_path: str | None = None
    guideline_pdf_path: str | None = None
    target_columns: list[str] | None = None
    formats: list[str] = field(default_factory=lambda: ["pdf", "markdown", "json"])
    thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)
