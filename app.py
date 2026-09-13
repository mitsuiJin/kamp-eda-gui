"""Streamlit 엔트리포인트 — CSV 업로드부터 분석 결과 표시까지의 전체 UI 흐름을 구성한다."""

import pandas as pd
import streamlit as st

from src.analyses.distribution import numeric_column_stats
from src.analyses.group_comparison import group_counts, group_numeric_summary
from src.analyses.outlier import outlier_mask, outlier_summary
from src.analyses.relationship import correlation_matrix
from src.analyses.time_series import datetime_parseable_columns, sort_by_time
from src.data_loader import CANDIDATE_ENCODINGS, detect_encoding, load_csv, preview_lines
from src.profiler import categorical_summary, column_profile, dataset_overview, numeric_summary
from src.visualization.charts import (
    bar_chart,
    boxplot,
    correlation_heatmap,
    grouped_boxplot,
    grouped_histogram,
    histogram,
    outlier_scatter_over_time,
    scatter_plot,
    time_series_line,
)

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
        ["변수 분포 확인", "변수 간 관계 확인", "그룹별 비교", "시간에 따른 변화 확인", "통계적 이상치 탐색"],
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

    elif analysis_purpose == "그룹별 비교":
        group_column = st.selectbox("그룹으로 사용할 컬럼을 선택하세요", df.columns.astype(str))

        st.caption("그룹별 count / 비율")
        st.dataframe(group_counts(df, group_column), use_container_width=True)

        numeric_columns = [
            c for c in df.select_dtypes(include="number").columns.astype(str) if c != group_column
        ]
        if not numeric_columns:
            st.warning("비교할 수치형 변수가 없습니다.")
        else:
            analysis_variables = st.multiselect(
                "비교할 수치형 변수를 선택하세요",
                numeric_columns,
                default=numeric_columns[:1],
            )
            if not analysis_variables:
                st.info("비교할 변수를 1개 이상 선택해주세요.")
            else:
                st.caption("그룹별 mean / std")
                st.dataframe(
                    group_numeric_summary(df, group_column, analysis_variables),
                    use_container_width=True,
                )

                for variable in analysis_variables:
                    chart_col1, chart_col2 = st.columns(2)
                    chart_col1.plotly_chart(
                        grouped_boxplot(df, group_column, variable, title=f"{group_column}별 {variable} 박스플롯"),
                        use_container_width=True,
                    )
                    chart_col2.plotly_chart(
                        grouped_histogram(df, group_column, variable, title=f"{group_column}별 {variable} 히스토그램"),
                        use_container_width=True,
                    )

    elif analysis_purpose == "시간에 따른 변화 확인":
        time_candidates = datetime_parseable_columns(df)
        numeric_columns = df.select_dtypes(include="number").columns.astype(str).tolist()

        if not time_candidates:
            st.warning("시간축으로 사용할 수 있는(날짜/시간 형식) 컬럼이 없습니다.")
        elif not numeric_columns:
            st.warning("관찰할 수치형 변수가 없습니다.")
        else:
            time_column = st.selectbox("시간축으로 사용할 컬럼을 선택하세요", time_candidates)
            value_column = st.selectbox("관찰할 수치형 변수를 선택하세요", numeric_columns)

            use_rolling = st.checkbox("rolling mean 표시")
            rolling_window = None
            if use_rolling:
                rolling_window = st.number_input("rolling window 크기", min_value=2, value=5, step=1)

            sorted_df = sort_by_time(df, time_column, value_column)
            st.plotly_chart(
                time_series_line(
                    sorted_df,
                    time_column,
                    value_column,
                    rolling_window,
                    title=f"{value_column} 추세 ({time_column} 기준)",
                ),
                use_container_width=True,
            )
            st.caption("시간에 따른 변화를 관찰하는 차트입니다 — 설비 이상 여부를 판단하지 않습니다.")

            missing_count = int(sorted_df[value_column].isna().sum())
            if missing_count > 0:
                st.caption(f"{value_column} 결측 {missing_count}건 — 위 그래프에서 선이 끊긴 구간이 결측 위치입니다.")

    elif analysis_purpose == "통계적 이상치 탐색":
        numeric_columns = df.select_dtypes(include="number").columns.astype(str).tolist()
        if not numeric_columns:
            st.warning("이상치를 확인할 수치형 변수가 없습니다.")
        else:
            column = st.selectbox("이상치를 확인할 컬럼을 선택하세요", numeric_columns)
            series = df[column]

            st.dataframe(pd.DataFrame([outlier_summary(series)]), use_container_width=True)
            st.plotly_chart(boxplot(series, title=f"{column} 박스플롯 (이상치 표시)"), use_container_width=True)
            st.warning("IQR 기준의 통계적 이상치이며, 제조공정의 실제 이상 또는 설비 고장을 의미하지 않습니다.")

            time_candidates = datetime_parseable_columns(df)
            if time_candidates:
                show_time_view = st.checkbox("시간축 기준 이상치 위치 표시")
                if show_time_view:
                    time_column = st.selectbox(
                        "시간축으로 사용할 컬럼을 선택하세요",
                        time_candidates,
                        key="outlier_time_column",
                    )
                    sorted_df = sort_by_time(df, time_column, column)
                    mask = outlier_mask(sorted_df[column])
                    st.plotly_chart(
                        outlier_scatter_over_time(
                            sorted_df, time_column, column, mask, title=f"{column} 이상치 위치 ({time_column} 기준)"
                        ),
                        use_container_width=True,
                    )
