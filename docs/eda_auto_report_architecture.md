# EDA 자동화 & 리포트 생성 모듈 — 아키텍처 (v2, 구현 반영)

> 전제: [`eda_auto_report_design.md`](eda_auto_report_design.md)의 **v2 개정 원칙**을 구현한 실제 코드 구조를 기술한다. 입력은 CSV/TXT 표 파일 1개 + (선택) Guideline PDF 또는 메타데이터 JSON. 이미지·파형·ARFF는 범위 밖이다.

---

## 1. 설계 원칙

1. **EDA는 해석하지 않고 관찰한다.** 통계량·분포·관계라는 사실을 산출하고, 도메인 판단·변환 권장·알고리즘 추천은 하지 않는다.
2. **Guideline 메타데이터 우선, EDA는 검증만.** 변수 타입·시간축·target 같은 "무엇을 분석할지"는 Guideline에서 받고, 실제 데이터와 일치가 확인된 정보만 사용한다. 불일치는 고치지 않고 상태로 기록한다.
3. **분석 엔진 하나, 렌더러 둘.** 계산은 한 번만 하고 PDF(사람용)와 Markdown/JSON(AI Context용)은 같은 결과를 다르게 렌더링한다.
4. **모든 분석은 상태를 남긴다.** SUCCESS / SKIPPED / NOT_APPLICABLE / FAILED + 사유 + 파라미터 + 입력 컬럼 + 소요 시간.
5. **임계값은 한 곳에.** `AnalysisThresholds`에 모으고 실행 시 Manifest/Context에 기록한다.

---

## 2. 디렉터리 구조

```
eda_report/
├── cli.py                        # 진입점(--input/--metadata/--guideline-pdf/--target/--formats)
├── config.py                     # RunConfig, AnalysisThresholds(모든 임계값 + 근거 주석)
├── pipeline.py                   # load → metadata → validate → profile → decide → run → render
│
├── io/
│   ├── loader.py                 # 인코딩·구분자·헤더 자동 감지, 표 구조 검증(행 0개/단일 라인 차단)
│   └── manifest.py               # ParseManifest
│
├── metadata/                     # [v2 신규] Guideline → 구조화 메타데이터 계층
│   ├── schema.py                 # DatasetMetadata / ColumnMetadata (+ JSON 입출력)
│   ├── validator.py              # 실제 데이터와 대조 검증(confirmed/type_mismatch/missing_in_data/...)
│   └── pdf_extract.py            # PDF 초안 추출(실제 컬럼명 매칭만, 타입 토큰이 있을 때만 선언 타입 채택)
│
├── profiling/
│   ├── column_profile.py         # role = 검증된 메타데이터 > dtype (고유값 수로 추론하지 않음)
│   └── dataset_profile.py        # numeric/categorical/datetime/text 집합, target 제외
│
├── selection/
│   └── rules.py                  # 분석별 실행 조건 → AnalysisDecision(run/status/reason/params)
│
├── analyses/                     # 항목당 파일 1개, 모두 같은 시그니처
│   ├── base.py                   # AnalysisResult/Figure/Finding + 상태 헬퍼 + 표시용 유틸
│   ├── manifest_page.py(00) glossary.py(00) overview.py(01) column_profile_page.py(02)
│   ├── missing.py(03) descriptive.py(04) distribution_numeric.py(05) distribution_categorical.py(06)
│   ├── outlier.py(07: run_iqr / run_multivariate) correlation.py(08) relationship.py(09)
│   ├── cat_numeric.py(10) cat_categorical.py(11) pca.py(12) clustering.py(13)
│   ├── target.py(15) timeseries.py(16) association.py(17)
│   └── findings.py(18: Key Observations)   ※ 19번 FE Hints는 v2에서 삭제
│
└── render/
    ├── mpl_style.py              # 한글 폰트 전역 설정
    ├── pdf/{builder,layout}.py   # ReportLab 조립(상태 표시, 표 줄바꿈, 섹션 자동 배치)
    └── context/{markdown,json}_builder.py

tests/
├── test_eda_loader.py     # 인코딩/구분자/헤더/비표형 파일
├── test_eda_profiling.py  # role 결정, 메타데이터 검증, target 제외
└── test_eda_analyses.py   # IQR=0, 왜도 문구, 군집 결정, 분석별 skip 사유
```

---

## 3. 핵심 데이터 계약

```python
@dataclass
class AnalysisResult:
    section_id: str
    title: str
    purpose: str
    rationale: str | None
    status: Literal["SUCCESS", "SKIPPED", "NOT_APPLICABLE", "FAILED"]
    status_reason: str | None
    input_columns: list[str]
    parameters: dict          # 사용한 기법·임계값·선택 기준(재현 가능하도록)
    duration_sec: float | None
    figures: list[Figure]
    tables: list[pd.DataFrame]
    findings: list[Finding]   # 관찰 사실(권장·판단 아님)
    ai_context: dict          # PDF엔 없는 상세 수치 전부

@dataclass
class ColumnProfile:
    name: str
    role: Literal["numeric","categorical","datetime","text","constant","empty"]
    role_source: Literal["metadata","dtype","data_quality"]   # 판단 출처를 항상 기록
    observed_dtype: str; declared_type: str
    missing_rate: float; n_unique: int; unique_ratio: float

@dataclass
class ColumnValidation:   # metadata/validator.py
    name: str
    status: Literal["confirmed","type_mismatch","missing_in_data","not_in_metadata","unverifiable"]
    declared_type: str; observed_dtype: str; detail: str | None
```

분석 모듈 시그니처: `run(df, profile: DatasetProfile, params: dict) -> AnalysisResult`

---

## 4. 실행 흐름

```
1. io.loader.load_table(input)
      → 인코딩/구분자/헤더 감지 + 표 구조 검증 → (df, ParseManifest)
2. metadata 확보 (우선순위: --metadata JSON > --guideline-pdf 초안 > 없음)
3. metadata.validator.validate_metadata(metadata, df)
      → 컬럼별 confirmed/type_mismatch/..., target·datetime 사용 가능 여부
4. profiling.build_dataset_profile(df, manifest, metadata, validation, thresholds, targets)
      → role은 confirmed된 메타데이터 우선, 없으면 dtype
5. selection.rules.decide_analyses(profile, run_config)
      → 분석별 run/skip 결정 + 사유 + 파라미터
6. 각 분석 실행(예외는 FAILED로 격리) → AnalysisResult + duration
7. findings.build_key_observations → manifest_page → glossary → 섹션번호 정렬
8. render: pdf / markdown / json
```

**target 확정 규칙**: `--target` > Guideline 메타데이터의 `target_columns` 중 실제 데이터에 존재하는 것. EDA는 target을 추측하지 않으며, 없으면 15번은 NOT_APPLICABLE로 기록된다.

---

## 5. CLI

```bash
python -m eda_report.cli --input data.csv --output-dir out/
python -m eda_report.cli --input data.csv --output-dir out/ --guideline-pdf guide.pdf --target passorfail
python -m eda_report.cli --input data.csv --output-dir out/ --metadata meta.json
```

| 파라미터 | 설명 |
|---|---|
| `--input` | CSV/TXT 표 파일(필수) |
| `--output-dir` | report.pdf / context.md / context.json 저장 위치(필수) |
| `--metadata` | 검증된 메타데이터 JSON(스키마: `metadata/schema.py`) |
| `--guideline-pdf` | Guideline PDF — 컬럼명 매칭 기반 초안 추출 |
| `--target` | target 컬럼(콤마 구분). 지정 없으면 Target Analysis는 NOT_APPLICABLE |
| `--formats` | 기본 `pdf,markdown,json` |

표로 해석할 수 없는 입력은 종료 코드 2와 사유를 출력한다.

---

## 6. 기술 스택

| 영역 | 선택 | 비고 |
|---|---|---|
| 차트 | matplotlib | 정적 PNG → PDF 임베드, 한글 폰트 전역 설정 |
| PDF | ReportLab | Windows에서 네이티브 의존성 없이 pip만으로 설치 |
| 통계/ML | pandas, numpy, scipy, scikit-learn | PCA/KMeans/IsolationForest/StandardScaler |
| 연관규칙 | mlxtend | 선택적 — 미설치 시 해당 분석만 SKIPPED |
| PDF 텍스트 | pymupdf | Guideline 추출용 — 미설치 시 메타데이터 없이 진행 |

---

## 7. 테스트 전략

- **단위**: 로더(인코딩/구분자/headerless/2행 헤더/비표형), 프로파일링(role 출처, 메타데이터 검증), 분석(IQR=0, 왜도 문구, 군집 결정, skip 사유)
- **회귀 방지 포인트**: ① 저카디널리티 numeric이 categorical로 바뀌지 않을 것 ② IQR=0 변수가 이상치로 집계되지 않을 것 ③ 왜도 문구에 변환 권장이 없을 것 ④ 군집 결정이 target 불균형에 영향받지 않을 것
- **엔드투엔드**: 실제 KAMP 데이터셋 6종(headerless, 2행 헤더, 시계열, 범주형 중심, 소규모, 비표형)으로 상태·파싱·역할 분포를 확인
