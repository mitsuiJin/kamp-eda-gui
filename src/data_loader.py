import io

import pandas as pd

CANDIDATE_ENCODINGS = ["utf-8", "cp949", "euc-kr"]


def detect_encoding(raw_bytes: bytes) -> str | None:
    for encoding in CANDIDATE_ENCODINGS:
        try:
            raw_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return None


def preview_lines(raw_bytes: bytes, encoding: str, n: int = 5) -> list[str]:
    text = raw_bytes.decode(encoding, errors="replace")
    return text.splitlines()[:n]


def load_csv(raw_bytes: bytes, encoding: str, header_row: int | None) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, header=header_row)
