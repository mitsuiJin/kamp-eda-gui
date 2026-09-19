# [보관됨] KAMP EDA GUI (Streamlit)

이 폴더는 프로젝트 v1 시도였던 Streamlit 기반 대화형 EDA GUI 도구입니다. 더 이상 유지보수하지 않습니다.

- 폐기 이유: 대회 당일 "자동으로 PDF 리포트 + AI Agent용 컨텍스트 파일을 생성한다"는 새 목표에는 사람이 매번 클릭하며 목적을 고르는 대화형 GUI 구조가 맞지 않아, 완전 자동화된 조건부 파이프라인으로 새로 설계했습니다.
- 설계 기록: [`../../docs/eda_gui_spec.md`](../../docs/eda_gui_spec.md)
- 후속 모듈: [`../../eda_report/`](../../eda_report/), 설계 문서 [`../../docs/eda_auto_report_design.md`](../../docs/eda_auto_report_design.md)

## 실행 (참고용)

```bash
pip install -r requirements.txt
streamlit run app.py
```
