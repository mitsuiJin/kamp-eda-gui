"""Streamlit 엔트리포인트 — CSV 업로드부터 분석 결과 표시까지의 전체 UI 흐름을 구성한다."""

import pandas as pd
import streamlit as st

from src.analyses.distribution import numeric_column_stats
from src.analyses.relationship import correlation_matrix
from src.data_loader import CANDIDATE_ENCODINGS, detect_encoding, load_csv, preview_lines
from src.profiler import categorical_summary, column_profile, dataset_overview, numeric_summary
from src.visualization.charts import bar_chart, boxplot, correlation_heatmap, histogram, scatter_plot

st.set_page_config(page_title="KAMP EDA GUI", layout="wide")
st.title("KAMP 제조 CSV 기초 분석 GUI")
st.caption("설계 명세: docs/eda_gui_spec.md")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 기본 데이터 정보를 확인할 수 있습니다.")
else:
    raw_bytes = uploaded_file.getvalue()

    detected_encoding = detect_encoding(raw_bytes)
    if detected_encoding is None:
        st.warning("인코딩을 자동으로 판별하지 못했습니다. 직접 선택해주세요.")
    default_index = CANDIDATE_ENCODINGS.index(detected_encoding) if detected_encoding else 0
    encoding = st.selectbox("인코딩", CANDIDATE_ENCODINGS, index=default_index)

    st.text("원본 미리보기 (처음 5줄) — 인코딩·헤더 행이 올바른지 이 미리보기로 확인하세요")
    st.code("\n".join(preview_lines(raw_bytes, encoding)))

    no_header = st.checkbox("이 파일은 헤더가 없습니다")
    header_row = None
    if not no_header:
        header_row = st.number_input("헤더로 사용할 행 번호 (1부터 시작)", min_value=1, value=1, step=1) - 1

    try:
        df = load_csv(raw_bytes, encoding, header_row)
    except Exception as e:
        st.error(f"CSV를 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    st.subheader("데이터 개요")
    st.caption(f"파일명: {uploaded_file.name}")
    overview = dataset_overview(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("행 수", f"{overview['행 수']:,}")
    col2.metric("열 수", f"{overview['열 수']:,}")
    col3.metric("메모리 사용량", f"{overview['메모리 사용량(MB)']:.2f} MB")
    col4.metric("중복행 수", f"{overview['중복행 수']:,}")

    st.subheader("컬럼 프로파일")
    st.dataframe(column_profile(df), use_container_width=True)

    st.subheader("기술통계")
    numeric_stats = numeric_summary(df)
    if not numeric_stats.empty:
        st.caption("수치형 컬럼")
        st.dataframe(numeric_stats, use_container_width=True)

    categorical_stats = categorical_summary(df)
    if not categorical_stats.empty:
        st.caption("범주형 컬럼")
        st.dataframe(categorical_stats, use_container_width=True)

    st.divider()
    st.subheader("분석 목적")
    analysis_purpose = st.radio(
        "분석 목적을 선택하세요",
        ["변수 분포 확인", "변수 간 관계 확인"],
        horizontal=True,
    )

    if analysis_purpose == "변수 분포 확인":
        column = st.selectbox("분석할 컬럼을 선택하세요", df.columns.astype(str))
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            st.dataframe(pd.DataFrame([numeric_column_stats(series)]), use_container_width=True)
            chart_col1, chart_col2 = st.columns(2)
            chart_col1.plotly_chart(histogram(series, title=f"{column} 히스토그램"), use_container_width=True)
            chart_col2.plotly_chart(boxplot(series, title=f"{column} 박스플롯"), use_container_width=True)
        else:
            st.plotly_chart(bar_chart(series, title=f"{column} 빈도"), use_container_width=True)

    elif analysis_purpose == "변수 간 관계 확인":
        numeric_columns = df.select_dtypes(include="number").columns.astype(str).tolist()
        if len(numeric_columns) < 2:
            st.warning("수치형 변수가 2개 미만이라 변수 간 관계를 확인할 수 없습니다.")
        else:
            selected_columns = st.multiselect(
                "상관관계를 확인할 수치형 변수를 선택하세요",
                numeric_columns,
                default=numeric_columns,
            )
            if len(selected_columns) < 2:
                st.info("2개 이상의 변수를 선택해주세요.")
            else:
                corr = correlation_matrix(df, selected_columns)
                st.plotly_chart(correlation_heatmap(corr, title="상관관계 히트맵"), use_container_width=True)

                scatter_col1, scatter_col2 = st.columns(2)
                x_col = scatter_col1.selectbox("X축 변수", selected_columns, index=0)
                y_col = scatter_col2.selectbox(
                    "Y축 변수", selected_columns, index=1 if len(selected_columns) > 1 else 0
                )
                st.plotly_chart(scatter_plot(df, x_col, y_col, title=f"{x_col} vs {y_col}"), use_container_width=True)
