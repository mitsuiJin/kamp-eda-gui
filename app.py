import pandas as pd
import streamlit as st

from src.data_loader import CANDIDATE_ENCODINGS, detect_encoding, load_csv, preview_lines

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

    st.subheader("기본 데이터 정보")
    st.caption(f"파일명: {uploaded_file.name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("행 수", f"{df.shape[0]:,}")
    col2.metric("열 수", f"{df.shape[1]:,}")
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    col3.metric("메모리 사용량", f"{memory_mb:.2f} MB")

    st.subheader("컬럼 목록 / dtype")
    dtype_table = pd.DataFrame(
        {
            "컬럼명": df.columns.astype(str),
            "dtype": df.dtypes.astype(str).values,
        }
    )
    st.dataframe(dtype_table, use_container_width=True)
