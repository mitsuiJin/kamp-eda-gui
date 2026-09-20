"""CLI 진입점.

사용 예:
    python -m eda_report.cli --input data.csv --output-dir out/
    python -m eda_report.cli --input data.csv --output-dir out/ --target passorfail
    python -m eda_report.cli --input data.csv --output-dir out/ --column-glossary glossary.json

target은 EDA가 추론하지 않는다 — `--target`으로 명시된 경우에만 분석한다.
--column-glossary는 컬럼명에 대한 사람이 검수한 설명 텍스트를 표시용으로만 덧붙인다(초안은
`python -m eda_report.column_glossary`로 PDF에서 추출).
"""

from __future__ import annotations

import argparse

import sys

from eda_report.config import AnalysisThresholds, RunConfig
from eda_report.io.loader import NotTabularDataError
from eda_report.pipeline import run


def main() -> None:
    # Windows 콘솔(cp949)에서 일부 문자(—, ≥ 등)가 인코딩되지 않아 출력 중 죽는 것을 막는다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="KAMP 제조 데이터 EDA 자동 리포트 생성")
    parser.add_argument("--input", required=True, help="CSV/TXT 표 파일 경로")
    parser.add_argument("--output-dir", required=True, help="report.pdf / context.md / context.json 저장 위치")
    parser.add_argument("--target", default=None, help="target 컬럼명(콤마로 여러 개)")
    parser.add_argument(
        "--column-glossary", default=None,
        help="사람이 검수한 컬럼 설명 JSON 경로(표시용, type/target 판단에는 쓰이지 않음)",
    )
    parser.add_argument("--formats", default="pdf,markdown,json", help="생성할 산출물")
    args = parser.parse_args()

    run_config = RunConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        target_columns=[c.strip() for c in args.target.split(",")] if args.target else None,
        column_glossary_path=args.column_glossary,
        formats=[f.strip() for f in args.formats.split(",")],
        thresholds=AnalysisThresholds(),
    )

    try:
        results = run(run_config)
    except NotTabularDataError as exc:
        print(f"[입력 오류] 이 파일은 표 데이터로 해석할 수 없습니다: {exc}")
        sys.exit(2)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    summary = ", ".join(f"{status} {count}" for status, count in counts.items())
    print(f"완료: {summary} | 출력 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
