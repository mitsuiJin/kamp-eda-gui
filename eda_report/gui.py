"""로컬 데스크톱 GUI — 기존 eda_report.column_glossary / eda_report.pipeline 기능을 감싸는 창.

새 분석 로직은 만들지 않는다. 여기서 하는 일은 파일 대화상자로 고른 입력값을 그대로
`column_glossary.extract_glossary_draft()`와 `pipeline.run(RunConfig(...))`에 넘기고, 그
결과를 창에 보여주는 것뿐이다 — 각각 CLI 진입점인 eda_report.column_glossary.main()/
eda_report.cli.main()과 정확히 같은 함수를 호출한다. 수십 초~수 분 걸릴 수 있는 작업은
백그라운드 스레드에서 실행해 창이 멈추지 않게 한다(matplotlib은 mpl_style.py에서 이미
비-GUI 백엔드인 "Agg"로 고정되어 있어 백그라운드 스레드에서 그려도 tkinter와 충돌하지 않는다).

실행: python -m eda_report.gui
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from eda_report.column_glossary import GlossaryExtractionError, extract_glossary_draft
from eda_report.config import AnalysisThresholds, RunConfig
from eda_report.io.loader import NotTabularDataError, load_table
from eda_report.pipeline import run as run_pipeline

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class EdaReportApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KAMP EDA Report Generator")
        self.geometry("760x700")
        self.minsize(680, 600)

        self._log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._last_output_dir: str | None = None

        self._build_widgets()
        self.after(100, self._poll_log_queue)

    # ---------------------------------------------------------------- UI --
    def _build_widgets(self) -> None:
        pad = {"padx": 12, "pady": 6}
        self.grid_columnconfigure(1, weight=1)
        row = 0

        ctk.CTkLabel(self, text="데이터 파일 (CSV/TXT)", anchor="w").grid(row=row, column=0, sticky="w", **pad)
        self.data_entry = ctk.CTkEntry(self, placeholder_text="분석할 CSV/TXT 파일 경로")
        self.data_entry.grid(row=row, column=1, sticky="ew", **pad)
        ctk.CTkButton(self, text="찾아보기", width=90, command=self._browse_data_file).grid(row=row, column=2, **pad)
        row += 1

        ctk.CTkLabel(self, text="Guidebook PDF (선택)", anchor="w").grid(row=row, column=0, sticky="w", **pad)
        self.guidebook_entry = ctk.CTkEntry(self, placeholder_text="Glossary 생성에만 필요")
        self.guidebook_entry.grid(row=row, column=1, sticky="ew", **pad)
        ctk.CTkButton(self, text="찾아보기", width=90, command=self._browse_guidebook).grid(row=row, column=2, **pad)
        row += 1

        ctk.CTkLabel(self, text="Target 컬럼 (선택)", anchor="w").grid(row=row, column=0, sticky="w", **pad)
        self.target_combo = ctk.CTkComboBox(self, values=[])
        self.target_combo.set("")
        self.target_combo.grid(row=row, column=1, sticky="ew", **pad)
        ctk.CTkLabel(self, text="목록 자동표시·직접입력·쉼표로 다중선택", text_color="gray55").grid(
            row=row, column=2, sticky="w", **pad
        )
        row += 1

        ctk.CTkLabel(self, text="Column Glossary JSON (선택)", anchor="w").grid(row=row, column=0, sticky="w", **pad)
        self.glossary_entry = ctk.CTkEntry(self, placeholder_text="생성 결과 저장 경로 또는 기존 검수 파일")
        self.glossary_entry.grid(row=row, column=1, sticky="ew", **pad)
        ctk.CTkButton(self, text="찾아보기", width=90, command=self._browse_glossary).grid(row=row, column=2, **pad)
        row += 1

        ctk.CTkLabel(self, text="출력 폴더", anchor="w").grid(row=row, column=0, sticky="w", **pad)
        self.output_entry = ctk.CTkEntry(self, placeholder_text="report.pdf / context.md / context.json 저장 위치")
        self.output_entry.grid(row=row, column=1, sticky="ew", **pad)
        ctk.CTkButton(self, text="찾아보기", width=90, command=self._browse_output_dir).grid(row=row, column=2, **pad)
        row += 1

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(14, 6))
        button_row.grid_columnconfigure((0, 1), weight=1)
        self.glossary_button = ctk.CTkButton(
            button_row, text="① Glossary 생성 (Guidebook → 초안 JSON)", command=self._on_generate_glossary
        )
        self.glossary_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.run_button = ctk.CTkButton(
            button_row, text="② EDA 리포트 생성", command=self._on_run_eda, fg_color="#2E8B57", hover_color="#256F46"
        )
        self.run_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        row += 1

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 6))
        row += 1

        self.status_label = ctk.CTkLabel(self, text="대기 중", anchor="w", text_color="gray55")
        self.status_label.grid(row=row, column=0, columnspan=3, sticky="w", padx=12)
        row += 1

        self.log_box = ctk.CTkTextbox(self, height=280, state="disabled", wrap="word")
        self.log_box.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=12, pady=(6, 12))
        self.grid_rowconfigure(row, weight=1)
        row += 1

        self.open_output_button = ctk.CTkButton(
            self, text="출력 폴더 열기", command=self._open_output_folder, state="disabled"
        )
        self.open_output_button.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 12))

    # ------------------------------------------------------- 파일 선택 --
    def _browse_data_file(self) -> None:
        path = filedialog.askopenfilename(
            title="분석할 데이터 파일 선택",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        self._set_entry(self.data_entry, path)
        self._fill_default_paths(path)
        self._load_columns_async(path)

    def _browse_guidebook(self) -> None:
        path = filedialog.askopenfilename(title="Dataset Guidebook PDF 선택", filetypes=[("PDF", "*.pdf")])
        if path:
            self._set_entry(self.guidebook_entry, path)

    def _browse_glossary(self) -> None:
        path = filedialog.askopenfilename(
            title="Column Glossary JSON 선택(생성할 새 파일명도 직접 입력할 수 있습니다)",
            filetypes=[("JSON", "*.json"), ("모든 파일", "*.*")],
        )
        if path:
            self._set_entry(self.glossary_entry, path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="출력 폴더 선택")
        if path:
            self._set_entry(self.output_entry, path)

    @staticmethod
    def _set_entry(entry: ctk.CTkEntry, value: str) -> None:
        entry.delete(0, "end")
        entry.insert(0, value)

    def _fill_default_paths(self, data_path: str) -> None:
        # 편의 기본값일 뿐이다 — 사용자가 이미 채운 칸은 건드리지 않는다.
        stem = Path(data_path).stem
        parent = Path(data_path).parent
        if not self.output_entry.get().strip():
            self._set_entry(self.output_entry, str(parent / f"{stem}_eda_output"))
        if not self.glossary_entry.get().strip():
            self._set_entry(self.glossary_entry, str(parent / f"{stem}_glossary.json"))

    # --------------------------------------------------- 컬럼 자동 감지 --
    def _load_columns_async(self, path: str) -> None:
        def worker() -> None:
            try:
                df, _ = load_table(path)
                self._log_queue.put(("columns", list(df.columns)))
            except Exception as exc:  # 컬럼 미리보기 실패는 치명적이지 않다 — target은 직접 입력 가능
                self._log_queue.put(
                    ("log", f"[안내] 컬럼 목록을 미리 불러오지 못했습니다({exc}). target은 직접 입력할 수 있습니다.")
                )

        self._append_log(f"컬럼 목록을 불러오는 중: {path}")
        threading.Thread(target=worker, daemon=True).start()

    # ----------------------------------------------------------- 실행 --
    def _on_generate_glossary(self) -> None:
        if self._busy:
            return
        data_path = self.data_entry.get().strip()
        guidebook_path = self.guidebook_entry.get().strip()
        glossary_path = self.glossary_entry.get().strip()

        if not data_path:
            messagebox.showwarning("입력 필요", "데이터 파일을 먼저 선택하세요.")
            return
        if not guidebook_path:
            messagebox.showwarning("입력 필요", "Guidebook PDF를 선택하세요.")
            return
        if not glossary_path:
            messagebox.showwarning("입력 필요", "Glossary를 저장할 경로를 입력하세요.")
            return

        def worker() -> None:
            try:
                self._log_queue.put(("log", f"데이터 파일 읽는 중: {data_path}"))
                df, _ = load_table(data_path)
                self._log_queue.put(("log", f"Guidebook에서 컬럼 설명 초안 추출 중... (컬럼 {len(df.columns)}개)"))
                draft = extract_glossary_draft(guidebook_path, list(df.columns))
                os.makedirs(os.path.dirname(glossary_path) or ".", exist_ok=True)
                with open(glossary_path, "w", encoding="utf-8") as f:
                    json.dump(draft, f, ensure_ascii=False, indent=2)
                matched, unmatched = len(draft["descriptions"]), len(draft["unmatched_columns"])
                self._log_queue.put((
                    "log",
                    f"초안 저장 완료: {glossary_path}\n"
                    f"(매칭 {matched}개 / 미매칭 {unmatched}개)\n"
                    "이 초안은 검증되지 않았습니다 — 내용을 검수·수정한 뒤 EDA 실행에 사용하세요.",
                ))
                self._log_queue.put(("done", True))
            except (NotTabularDataError, GlossaryExtractionError) as exc:
                self._log_queue.put(("log", f"[오류] {exc}"))
                self._log_queue.put(("done", False))
            except Exception:
                self._log_queue.put(("log", f"[오류] 예상치 못한 오류가 발생했습니다:\n{traceback.format_exc()}"))
                self._log_queue.put(("done", False))

        self._start_task(worker, "Glossary 생성 중...")

    def _on_run_eda(self) -> None:
        if self._busy:
            return
        data_path = self.data_entry.get().strip()
        output_dir = self.output_entry.get().strip()
        target_text = self.target_combo.get().strip()
        glossary_path = self.glossary_entry.get().strip()

        if not data_path:
            messagebox.showwarning("입력 필요", "데이터 파일을 먼저 선택하세요.")
            return
        if not output_dir:
            messagebox.showwarning("입력 필요", "출력 폴더를 선택하세요.")
            return

        target_columns = [c.strip() for c in target_text.split(",") if c.strip()] or None
        run_config = RunConfig(
            input_path=data_path,
            output_dir=output_dir,
            target_columns=target_columns,
            column_glossary_path=glossary_path or None,
            thresholds=AnalysisThresholds(),
        )

        def worker() -> None:
            try:
                self._log_queue.put(("log", "EDA 리포트 생성 중... (데이터 크기에 따라 수 분 걸릴 수 있습니다)"))
                results = run_pipeline(run_config)
                counts: dict[str, int] = {}
                for r in results:
                    counts[r.status] = counts.get(r.status, 0) + 1
                summary = ", ".join(f"{status} {count}" for status, count in counts.items())
                self._log_queue.put(("log", f"완료: {summary}\n출력 위치: {output_dir}"))
                self._log_queue.put(("output_dir", output_dir))
                self._log_queue.put(("done", True))
            except NotTabularDataError as exc:
                self._log_queue.put(("log", f"[입력 오류] 이 파일은 표 데이터로 해석할 수 없습니다: {exc}"))
                self._log_queue.put(("done", False))
            except Exception:
                self._log_queue.put(("log", f"[오류] 예상치 못한 오류가 발생했습니다:\n{traceback.format_exc()}"))
                self._log_queue.put(("done", False))

        self._start_task(worker, "EDA 리포트 생성 중...")

    # --------------------------------------------------- 상태/로그 처리 --
    def _start_task(self, worker, status_text: str) -> None:
        self._busy = True
        self.glossary_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.open_output_button.configure(state="disabled")
        self.status_label.configure(text=status_text, text_color="gray55")
        self.progress.start()
        threading.Thread(target=worker, daemon=True).start()

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "columns":
                    columns = list(payload)
                    self.target_combo.configure(values=columns)
                    self._append_log(f"컬럼 {len(columns)}개를 불러왔습니다. target은 선택 사항입니다.")
                elif kind == "output_dir":
                    self._last_output_dir = str(payload)
                    self.open_output_button.configure(state="normal")
                elif kind == "done":
                    self._busy = False
                    self.glossary_button.configure(state="normal")
                    self.run_button.configure(state="normal")
                    self.progress.stop()
                    ok = bool(payload)
                    self.status_label.configure(
                        text="완료" if ok else "오류 발생 — 아래 로그를 확인하세요",
                        text_color="#2E8B57" if ok else "#C44E52",
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _open_output_folder(self) -> None:
        if self._last_output_dir and os.path.isdir(self._last_output_dir):
            os.startfile(self._last_output_dir)  # noqa: S606 — Windows 전용 데스크톱 GUI


def main() -> None:
    app = EdaReportApp()
    app.mainloop()


if __name__ == "__main__":
    main()
