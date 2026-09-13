# KAMP EDA GUI

KAMP 제조 CSV를 업로드하면 데이터 구조를 파악하고, 사용자가 선택한 분석 목적과 변수에 따라 기초 기술통계·시각화를 보여주는 Streamlit 도구입니다.

전체 설계 명세(단일 진실 공급원, SSOT)는 [`docs/eda_gui_spec.md`](docs/eda_gui_spec.md)를 따릅니다.

## 현재 상태

Phase 0 — 프로젝트 초기 설정 단계입니다. 분석 기능은 아직 구현되지 않았습니다.

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

```bash
streamlit run app.py
```

## 지원 CSV / 분석 목적

CSV 정형 데이터만 지원합니다(이미지·오디오 제외). 분석 목적, 분석 모듈, MVP 범위 등 전체 계획은 [`docs/eda_gui_spec.md`](docs/eda_gui_spec.md)에 정리되어 있습니다.

## 테스트 데이터

`dataset/` 폴더의 KAMP 50개 데이터셋(저장소에는 포함되지 않음 — `.gitignore` 처리)을 사용해 기능을 검증합니다.

## 향후 확장

`docs/eda_gui_spec.md`의 "향후 확장" 절 참고.
