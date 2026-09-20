# KAMP EDA 자동화 & 리포트 생성 모듈

CSV/TXT 제조 데이터를 입력하면 (1) 사람이 보는 시각화 중심 PDF 리포트와 (2) AI Agent용 구조화 Context(Markdown/JSON)를 자동 생성합니다.

## 설치

프로젝트 전용 가상환경을 사용합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행 방법

### GUI

```bash
python -m eda_report.gui
```

또는 `run_gui.bat` 더블클릭.

### CLI

```bash
python -m eda_report.cli --input data.csv --output-dir out/ --target passorfail
```

| 옵션 | 필수 | 설명 |
|---|---|---|
| `--input` | ✅ | CSV/TXT 경로 |
| `--output-dir` | ✅ | 산출물 저장 폴더 |
| `--target` | | target 컬럼명(콤마로 다중 지정 가능, 자동 추측하지 않음) |
| `--column-glossary` | | 사람이 검수한 컬럼 설명 JSON(표시용) |
| `--formats` | | 기본 `pdf,markdown,json` |

출력(`--output-dir` 하위): `report.pdf`, `context.md`, `context.json`, `figures/`

### 컬럼 설명(Column Glossary) 생성

Dataset Guidebook PDF에서 컬럼 설명 초안을 뽑습니다. 자동 추출 결과는 검증되지 않은 초안이므로 검수 후 `--column-glossary`로 사용하세요.

```bash
python -m eda_report.column_glossary --pdf guide.pdf --input data.csv --output glossary.json
```

## 테스트

```bash
pytest tests/
```

## 지원 입력

CSV, TXT(구분자 기반 표 데이터). `dataset/` 폴더는 용량 문제로 저장소에 포함하지 않습니다 — 각자 로컬에 KAMP 데이터셋을 받아서 사용하세요.
