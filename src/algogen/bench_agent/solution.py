from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid


class Solution:
    def __init__(self, name: str):
        self.name = name

    def run_solver(
        self,
        solver: Path,
        language: str,
        input_file: Path,
        timeout_seconds: int = 10,
    ) -> str:
        if not solver.exists():
            raise FileNotFoundError(f"Solver not found: {solver}")
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        input_text = input_file.read_text(encoding="utf-8")
        lang = language.lower()

        match lang:
            case "python" | "py":
                return self._run_python_solver(solver, input_text, timeout_seconds)
            case "cpp" | "cc":
                return self._compile_and_run_native_solver(
                    compiler="g++",
                    compile_flags=["-O2", "-std=c++17"],
                    solver=solver,
                    input_text=input_text,
                    timeout_seconds=timeout_seconds,
                )
            case "c":
                return self._compile_and_run_native_solver(
                    compiler="gcc",
                    compile_flags=["-O2", "-std=c11"],
                    solver=solver,
                    input_text=input_text,
                    timeout_seconds=timeout_seconds,
                )
            case "java":
                raise NotImplementedError("Java solver execution is not supported yet")
            case _:
                raise ValueError(f"Unsupported language: {language}")

    def _run_python_solver(
        self,
        solver: Path,
        input_text: str,
        timeout_seconds: int,
    ) -> str:
        # Prefer in-process execution to support template style `Solution.solve(input: str)`.
        try:
            return self._run_python_solution_class(solver, input_text)
        except Exception:
            pass

        python_bin = self._require_binary("python3")
        process = subprocess.run(
            [python_bin, str(solver)],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Python solver failed:\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}"
            )
        return process.stdout.rstrip("\n")

    def _run_python_solution_class(self, solver: Path, input_text: str) -> str:
        module_name = f"_solver_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, solver)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import python solver: {solver}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        solution_cls = getattr(module, "Solution", None)
        if solution_cls is None:
            raise RuntimeError("Cannot find class `Solution` in solver module")
        instance = solution_cls()
        solve_fn = getattr(instance, "solve", None)
        if solve_fn is None:
            raise RuntimeError("Cannot find method `Solution.solve` in solver module")
        result = solve_fn(input_text.rstrip("\n"))
        return str(result).rstrip("\n")

    def _compile_and_run_native_solver(
        self,
        compiler: str,
        compile_flags: list[str],
        solver: Path,
        input_text: str,
        timeout_seconds: int,
    ) -> str:
        compiler_bin = self._require_binary(compiler)
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = Path(tmpdir).joinpath("solver_bin")
            compile_cmd = [compiler_bin, *compile_flags, str(solver), "-o", str(binary_path)]
            compile_process = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if compile_process.returncode != 0:
                raise RuntimeError(
                    f"Compile failed for {solver}:\n"
                    f"stdout:\n{compile_process.stdout}\n"
                    f"stderr:\n{compile_process.stderr}"
                )
            run_process = subprocess.run(
                [str(binary_path)],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if run_process.returncode != 0:
                raise RuntimeError(
                    f"Executable failed for {solver}:\n"
                    f"stdout:\n{run_process.stdout}\n"
                    f"stderr:\n{run_process.stderr}"
                )
            return run_process.stdout.rstrip("\n")

    def _require_binary(self, command: str) -> str:
        resolved = shutil.which(command)
        if resolved:
            return resolved
        raise RuntimeError(f"Required command not found in PATH: {command}")
