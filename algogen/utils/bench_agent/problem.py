from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import os
from queue import Empty, Queue
import shutil
import subprocess
from threading import Lock
import zipfile

from .solution import Solution

logger = logging.getLogger(__name__)


class Problem:
    def __init__(
        self,
        name: str,
        bench_number: int = 5,
        threads: int = 1,
        sampler: str | Path | None = None,
        solver: Path = Path("./template/"),
        language: str | None = None,
        output_path: Path = Path("./output/"),
    ):
        self.name = name
        self.bench_number = bench_number
        self.threads = max(1, min(threads, os.cpu_count() or 1))
        self.sampler = Path(sampler) if sampler is not None else None
        self.problem_description = ""
        self.output_path = output_path.joinpath(name)
        self.input_and_output: list[tuple[Path, Path]] = []
        self.solver_path, self.language = self._resolve_solver_and_language(
            solver, language
        )
        self.solver = Solution(name)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._tmp_case_seq = 0
        self.crashed_reasons: list[str] = []

        self._accept_lock = Lock()
        self.problem_pool: Queue[Path | None] = Queue(maxsize=self.bench_number * 2)

    def set_sampler(self, sampler: str | Path | None):
        self.sampler = Path(sampler) if sampler is not None else None

    def set_problem_description(self, description: str) -> None:
        self.problem_description = description.strip()

    def describe(self, name: str = "description.md") -> Path:
        content = self.problem_description.strip()
        if not content:
            content = self.__doc__ or ""
        path = self.output_path.joinpath(name)
        path.write_text(f"{content.rstrip()}\n", encoding="utf-8")
        return path

    def archive(self) -> Path:
        description_path = self.output_path.joinpath("description.md")
        if not description_path.exists():
            self.describe()

        zip_file = self.output_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for input_file, output_file in sorted(
                self.input_and_output,
                key=lambda pair: pair[0].name,
            ):
                zipf.write(input_file, arcname=f"data/{input_file.name}")
                zipf.write(output_file, arcname=f"data/{output_file.name}")
            zipf.write(description_path, arcname="description.md")
        return zip_file

    def run_sample(self, output_file: Path, timeout_seconds: int = 10) -> None:
        if self.sampler is None:
            raise ValueError("Sampler is not set")
        if not self.sampler.exists():
            raise FileNotFoundError(f"Sampler script not found: {self.sampler}")

        python_bin = shutil.which("python3") or shutil.which("python")
        if python_bin is None:
            raise RuntimeError("Cannot find python interpreter for running sampler")

        with output_file.open("w", encoding="utf-8") as out:
            process = subprocess.run(
                [python_bin, str(self.sampler)],
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"Sampler failed: {self.sampler}\nstderr:\n{process.stderr}"
            )
        if output_file.stat().st_size == 0:
            raise RuntimeError(f"Sampler generated empty input file: {output_file}")

    def generate_case_group(
        self,
        group_size: int = 5,
    ) -> tuple[list[Path], list[str]]:
        if group_size <= 0:
            return [], []

        input_file_list = [
            self._next_temp_input_path(case_index)
            for case_index in range(1, group_size + 1)
        ]
        generation_failures: list[str] = []
        queued_cases = 0

        self.problem_pool.queue.clear()
        for input_file in input_file_list:
            error = self._generate_input(input_file)
            if error is not None:
                generation_failures.append(
                    f"case={input_file.name}, stage=sampler, error={error}"
                )
                continue
            self.problem_pool.put(input_file)
            queued_cases += 1

        if queued_cases == 0:
            return input_file_list, generation_failures

        worker_count = min(self.threads, queued_cases)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._solver_worker, generation_failures)
                for _ in range(worker_count)
            ]
            for future in futures:
                future.result()
        return input_file_list, generation_failures

    def cleanup_temp_group_files(self, input_files: list[Path]) -> None:
        for path in input_files:
            if not path.exists():
                continue
            if path.parent != self.output_path:
                continue
            if not path.name.startswith("tmp_"):
                continue
            path.unlink()

    def generate(self) -> bool:
        self.input_and_output.clear()
        self.crashed_reasons.clear()
        crashed = False
        while len(self.input_and_output) < self.bench_number:
            accepted_before = len(self.input_and_output)
            remaining = self.bench_number - accepted_before
            group_size = min(5, remaining)
            input_files, generation_failures = self.generate_case_group(
                group_size=group_size,
            )
            if generation_failures:
                crashed = True
                self.crashed_reasons.extend(generation_failures)
            self.cleanup_temp_group_files(input_files)

            if len(self.input_and_output) == accepted_before:
                break
        return crashed

    def _next_temp_input_path(self, case_index: int) -> Path:
        self._tmp_case_seq += 1
        return self.output_path.joinpath(
            f"tmp_c{case_index:02d}_{self._tmp_case_seq:04d}.in"
        )

    def _generate_input(self, input_file: Path) -> str | None:
        try:
            self.run_sample(input_file)
            return None
        except Exception as exc:
            logger.info(
                "Generate input failed (problem=%s, file=%s): %s",
                self.name,
                input_file.name,
                exc,
            )
            return str(exc)

    def _solver_worker(self, failures: list[str]) -> None:
        while True:
            try:
                input_file = self.problem_pool.get(timeout=0.1)
            except Empty:
                return
            try:
                if input_file is None:
                    return
                output_text = self.solver.run_solver(
                    solver=self.solver_path,
                    language=self.language,
                    input_file=input_file,
                )
                self._accept_passed_case(input_file, output_text)
            except Exception as exc:
                case_name = (
                    input_file.name if input_file is not None else "(queue_sentinel)"
                )
                failures.append(f"case={case_name}, stage=solver, error={exc}")
            finally:
                self.problem_pool.task_done()

    def _accept_passed_case(self, input_file: Path, output_text: str) -> None:
        with self._accept_lock:
            case_id = len(self.input_and_output) + 1
            final_input = self.output_path.joinpath(f"{case_id}.in")
            final_output = self.output_path.joinpath(f"{case_id}.out")
            final_input.write_text(
                input_file.read_text(encoding="utf-8"), encoding="utf-8"
            )
            final_output.write_text(output_text, encoding="utf-8")
            self.input_and_output.append((final_input, final_output))

    def _resolve_solver_and_language(
        self,
        solver: Path,
        language: str | None,
    ) -> tuple[Path, str]:
        normalized_language = self._normalize_language(language) if language else None

        if solver.is_file():
            suffix_language = self._language_from_suffix(solver.suffix)
            if suffix_language is None:
                raise ValueError(
                    f"Cannot infer language from solver file suffix: {solver.suffix}"
                )
            if normalized_language and normalized_language != suffix_language:
                raise ValueError(
                    f"Solver language mismatch: requested `{normalized_language}`, "
                    f"but file `{solver}` implies `{suffix_language}`"
                )
            return solver, normalized_language or suffix_language

        if normalized_language is not None:
            path = solver.joinpath(
                f"{self.name}{self._suffix_from_language(normalized_language)}"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Solver not found for language `{normalized_language}`: {path}"
                )
            return path, normalized_language

        detected = self._detect_solver_candidates(solver)
        if not detected:
            expected = ", ".join(
                f"{self.name}{suffix}" for suffix in [".py", ".cpp", ".c", ".java"]
            )
            raise FileNotFoundError(
                f"No solver found in `{solver}` for `{self.name}`. "
                f"Expected one of: {expected}"
            )
        if len(detected) > 1:
            choices = ", ".join(f"{path.name}({lang})" for path, lang in detected)
            raise RuntimeError(
                "Multiple solver files detected. "
                f"Please specify LANGUAGE explicitly. candidates={choices}"
            )
        return detected[0]

    def _detect_solver_candidates(self, solver_dir: Path) -> list[tuple[Path, str]]:
        candidates: list[tuple[Path, str]] = []
        for suffix, language in _SUFFIX_TO_LANGUAGE.items():
            candidate = solver_dir.joinpath(f"{self.name}{suffix}")
            if candidate.exists():
                candidates.append((candidate, language))
        return candidates

    def _normalize_language(self, language: str) -> str:
        raw = language.strip().lower()
        aliases = {"py": "python", "cc": "cpp"}
        normalized = aliases.get(raw, raw)
        if normalized not in _LANGUAGE_TO_SUFFIX:
            raise ValueError(f"Unsupported language: {language}")
        return normalized

    def _suffix_from_language(self, language: str) -> str:
        return _LANGUAGE_TO_SUFFIX[language]

    def _language_from_suffix(self, suffix: str) -> str | None:
        return _SUFFIX_TO_LANGUAGE.get(suffix.lower())


_LANGUAGE_TO_SUFFIX: dict[str, str] = {
    "python": ".py",
    "cpp": ".cpp",
    "c": ".c",
    "java": ".java",
}

_SUFFIX_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".cpp": "cpp",
    ".c": "c",
    ".java": "java",
}
