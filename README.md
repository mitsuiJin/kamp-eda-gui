# KAMP EDA 자동화 & 리포트 생성 모듈

KAMP 제조데이터 분석 경진대회 당일, 데이터셋(CSV/TXT)과 Dataset Guidebook PDF를 입력하면 (1) 사람이 읽는 시각화 중심 PDF 리포트와 (2) AI Agent에 넘길 구조화된 EDA Context(Markdown/JSON)를 자동 생성합니다.

```
Dataset Guidebook PDF ─→ Dataset Metadata ─→ 검증 ─┐
                                          ├─→ EDA(객관적 사실 산출) ─→ PDF Report + AI Context
CSV / TXT ────────────────────────────────┘
```

- 설계 원칙·실측 검증: [`docs/eda_auto_report_design.md`](docs/eda_auto_report_design.md)
- 아키텍처: [`docs/eda_auto_report_architecture.md`](docs/eda_auto_report_architecture.md)

## 설계 철학

**EDA Module은 데이터를 해석하는 모듈이 아니라, 객관적으로 관찰하고 구조화하는 모듈입니다.**

| 하는 것 | 하지 않는 것 |
|---|---|
| 통계량·분포·결측·중복 산출 | 공정 이상/센서 고장 판단 |
| 변수 간 통계적 관계 확인 | 변수의 물리적 의미 추론 |
| 조건이 충족되는 분석만 수행 | Feature Engineering·변환 권장 |
| 분석 상태와 사유 기록 | ML 알고리즘 추천 |
| 결과를 구조화해 저장 | 통계적 이상치를 공정 이상으로 해석 |

도메인 해석과 후속 분석 판단은 이 Context를 받는 AI Agent 또는 사용자의 몫입니다.

## 설치

프로젝트 전용 가상환경을 사용합니다(전역 인터프리터에 설치하지 않습니다).

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell / cmd
pip install -r requirements.txt
```

## 사용법

```bash
# target 없이 관찰 가능한 항목만 실행 (target 관련 섹션은 NOT_APPLICABLE로 기록됨)
python -m eda_report.cli --input data.csv --output-dir out/

# Dataset Guidebook PDF로 컬럼 타입을 초안 추출 → 데이터와 자동 대조·검증
python -m eda_report.cli --input data.csv --output-dir out/ --guideline-pdf guide.pdf --target passorfail

# Guidebook을 JSON으로 미리 정리해둔 경우 (PDF 추출 단계 생략)
python -m eda_report.cli --input data.csv --output-dir out/ --metadata meta.json

# target이 여러 컬럼에 나뉜 multilabel 데이터셋 (콤마로 구분)
python -m eda_report.cli --input data.csv --output-dir out/ --target "Crack_1,Crack_2,Stain_1"
```

| 옵션 | 필수 | 설명 |
|---|---|---|
| `--input` | ✅ | CSV/TXT 경로 |
| `--output-dir` | ✅ | 산출물 저장 폴더 |
| `--guideline-pdf` | | Dataset Guidebook PDF (메타데이터 초안 추출 + 검증) |
| `--metadata` | | 이미 정리된 Guidebook 메타데이터 JSON (검증만 수행) |
| `--target` | | target 컬럼명, 콤마로 다중 지정 가능. **자동 추측하지 않음** — 안 주면 target 관련 분석은 skip |
| `--formats` | | 기본 `pdf,markdown,json` |

출력(`--output-dir` 하위): `report.pdf`(사람용), `context.md`/`context.json`(AI Agent용, 전체 세부 결과 보존), `figures/`(PDF에 쓰인 원본 이미지).

## Core EDA vs Advanced/Optional EDA

모든 분석 결과에는 `tier`가 표시됩니다.

- **Core EDA**: 데이터 구조(dtype/행 개수)만으로 항상 시도되는 기초 분석 — Overview, Column Profile, Missing, Descriptive, Distribution, Outlier(IQR), Correlation, Relationship, Categorical×Numerical/Categorical
- **Advanced/Optional EDA**: 통계적 모델(Isolation Forest·PCA·KMeans)을 쓰거나 target·시간축처럼 데이터에 따라서만 성립하는 분석 — Outlier(다변량), PCA, Clustering, Target, Time Series, Association

Advanced가 덜 중요하다는 뜻이 아니라, 결과를 해석할 때 사용된 알고리즘·파라미터를 함께 감안해야 한다는 의미입니다(예: PCA의 PC1/PC2는 단위가 없는 투영 좌표이고, Clustering의 k=2~6 탐색 범위는 "공정이 그만큼의 상태를 가진다"는 도메인 규칙이 아니라 현재 구현의 탐색 범위일 뿐입니다).

## 분석 항목과 실행 조건

조건을 만족하지 못하면 **생략하는 것이 정상 동작**이며, 모든 분석의 상태(SUCCESS / SKIPPED / NOT_APPLICABLE / FAILED)와 사유가 Manifest에 기록됩니다. SKIPPED(조건은 맞지만 이번 데이터에서 건너뜀, 예: IQR=0)와 FAILED(실행 중 오류)는 서로 다른 상태입니다.

| 항목 | 구분 | 실행 조건 |
|---|---|---|
| Overview, Column Profile | Core | 항상 |
| Missing Values | Core | 결측이 있는 컬럼 존재 |
| Descriptive / Distribution / Outlier(IQR) | Core | 수치형 ≥ 1 (IQR=0인 변수는 해당 변수만 제외) |
| Correlation, Feature Relationship | Core | 수치형 ≥ 2 |
| Categorical Distribution | Core | 범주형 ≥ 1 |
| Categorical × Numerical | Core | 범주형 ≥ 1 AND 수치형 ≥ 1 |
| Categorical × Categorical | Core | 범주형 ≥ 2 |
| Outlier(다변량, Isolation Forest) | Advanced | 수치형 ≥ 3, 행 ≥ 500 |
| PCA | Advanced | 수치형 ≥ 3 |
| Clustering(KMeans) | Advanced | 수치형 ≥ 2, 행 ≥ 100 |
| Time Series | Advanced | Dataset Guidebook 지정 또는 날짜 파싱이 확인된 시간 변수 존재. **전체 수치형 변수를 다 계산하고, PDF에는 변동계수(CV) 기준 대표 변수만 시각화** — 전체 결과는 AI Context에 그대로 보존 |
| Target Analysis | Advanced | `--target` 또는 메타데이터로 target이 지정된 경우 (추측하지 않음). 이진/다중클래스/연속형/multilabel을 자동 구분 |
| Association | Advanced | 카디널리티 ≤ 20인 범주형 ≥ 2 AND 조합 수 제한 이내 |

## 지원 입력

CSV, TXT(구분자 기반 표). 이미지·파형(오디오)·ARFF는 범위 밖입니다(공식 50개 KAMP 데이터셋 중 01_OCR, 12_XrayInspection, 38_전자부품_음향기기, 47_Scene-Text, 48_안전관리 5개는 이미지/오디오 전용이라 대상 외). 인코딩(UTF-8/CP949/EUC-KR), headerless, 2행 헤더는 자동 감지하며 감지 결과를 Manifest에 기록합니다. `dataset/` 폴더는 용량 문제로 저장소에 포함하지 않습니다(`.gitignore`) — 각자 로컬에 KAMP 데이터셋을 받아서 사용하세요.

## 테스트

```bash
pytest tests/
```

실측 KAMP 데이터셋(`33_소성가공_품질보증`, `37_주조_품질보증` 등)으로 end-to-end 동작을 검증했습니다.

## 이전 버전(아카이브)

Streamlit 기반 대화형 GUI 도구는 이 프로젝트의 v1 시도였으며 [`archive/streamlit_gui/`](archive/streamlit_gui/)로 옮겨 더 이상 유지보수하지 않습니다.
