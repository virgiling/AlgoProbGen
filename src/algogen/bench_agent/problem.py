from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import os
import shutil
import subprocess
from typing import Any, Callable
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
        retry_times: int = 1,
    ):
        self.name = name
        self.bench_number = bench_number
        self.threads = threads
        self.pool: ThreadPoolExecutor | None = None
        self.sampler = Path(sampler) if sampler is not None else None
        self.problem_description = ""
        self.output_path = output_path.joinpath(name)
        self.input_and_output: list[tuple[Path, Path]] = []
        self.solver_path, self.language = self._resolve_solver_and_language(
            solver, language
        )
        self.retry_times = max(0, retry_times)
        self.solver = Solution(name)
        self.output_path.mkdir(parents=True, exist_ok=True)

        if threads > 1:
            workers = min(threads, os.cpu_count() or threads)
            self.pool = ThreadPoolExecutor(max_workers=workers)

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
                f"Sampler failed: {self.sampler}\n"
                f"stderr:\n{process.stderr}"
            )
        if output_file.stat().st_size == 0:
            raise RuntimeError(f"Sampler generated empty input file: {output_file}")

    def _generate_case(self, index: int) -> tuple[Path, Path]:
        input_file = self.output_path.joinpath(f"{index}.in")
        output_file = self.output_path.joinpath(f"{index}.out")
        last_error: Exception | None = None
        for attempt in range(1, self.retry_times + 2):
            try:
                self.run_sample(input_file)
                answer = self.solver.run_solver(
                    solver=self.solver_path,
                    language=self.language,
                    input_file=input_file,
                )
                output_file.write_text(answer, encoding="utf-8")
                return input_file, output_file
            except Exception as exc:  # noqa: PERF203 - explicit retry loop
                last_error = exc
                logger.warning(
                    "Generate case failed (problem=%s, case=%s, attempt=%s/%s): %s",
                    self.name,
                    index,
                    attempt,
                    self.retry_times + 1,
                    exc,
                )
        raise RuntimeError(f"Failed to generate case {index}") from last_error

    def _generate_single(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        self.input_and_output.clear()
        total = self.bench_number
        self._emit_subtask_progress(
            progress_callback,
            status="start",
            current=0,
            total=total,
        )
        for i in range(1, total + 1):
            try:
                self.input_and_output.append(self._generate_case(i))
            except Exception as exc:
                self._emit_subtask_progress(
                    progress_callback,
                    status="error",
                    current=i - 1,
                    total=total,
                    case_index=i,
                    message=str(exc),
                )
                raise
            self._emit_subtask_progress(
                progress_callback,
                status="update",
                current=i,
                total=total,
                case_index=i,
            )
        self._emit_subtask_progress(
            progress_callback,
            status="done",
            current=total,
            total=total,
        )
        return True

    def generate(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        self.input_and_output.clear()
        if self.pool is None:
            return self._generate_single(progress_callback=progress_callback)

        total = self.bench_number
        self._emit_subtask_progress(
            progress_callback,
            status="start",
            current=0,
            total=total,
        )
        results: list[tuple[Path, Path]] = []
        completed = 0
        futures = {
            self.pool.submit(self._generate_case, index): index
            for index in range(1, self.bench_number + 1)
        }
        try:
            for future in as_completed(futures):
                case_index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self._emit_subtask_progress(
                        progress_callback,
                        status="error",
                        current=completed,
                        total=total,
                        case_index=case_index,
                        message=str(exc),
                    )
                    raise RuntimeError(
                        f"Generate case failed for problem {self.name}, case={case_index}"
                    ) from exc
                results.append(result)
                completed += 1
                self._emit_subtask_progress(
                    progress_callback,
                    status="update",
                    current=completed,
                    total=total,
                    case_index=case_index,
                )
        except Exception as exc:
            raise RuntimeError(f"Generate data failed for problem {self.name}") from exc

        self.input_and_output.extend(sorted(results, key=lambda pair: pair[0].name))
        self._emit_subtask_progress(
            progress_callback,
            status="done",
            current=total,
            total=total,
        )
        return True

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

    def _emit_subtask_progress(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        *,
        status: str,
        current: int,
        total: int,
        case_index: int | None = None,
        message: str | None = None,
    ) -> None:
        if callback is None:
            return
        payload: dict[str, Any] = {
            "status": status,
            "current": current,
            "total": total,
        }
        if case_index is not None:
            payload["case_index"] = case_index
        if message:
            payload["message"] = message
        callback(payload)


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
