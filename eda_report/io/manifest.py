"""load_table()이 실제로 어떻게 파일을 해석했는지 투명하게 남기는 기록.

design.md의 "00. 표지/실행 매니페스트" 항목이 사람/AI 에이전트에게 보여줄 원본 그대로다 —
헤더 없음·2행 헤더·인코딩 문제가 45개 데이터셋 중 다수에서 실제로 발생했기 때문에,
컬럼명이 원본 그대로인지 자동 생성됐는지를 감추지 않는 것이 이 모듈의 핵심 원칙이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParseManifest:
    file_path: str
    encoding: str
    delimiter: str
    header_row: int | None
    generated_column_names: bool
    n_rows: int
    n_cols: int
    warnings: list[str] = field(default_factory=list)
