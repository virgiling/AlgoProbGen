from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
import tempfile
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..problem_spec import ProblemSpec
from .problem import Problem


@dataclass(slots=True)
class BenchArtifacts:
    sampler_path: Path
    description_path: Path
    zip_path: Path
    output_dir: Path


class BenchAgent:
    def __init__(
        self,
        name: str = "BenchAgent",
        llm: BaseChatModel | None = None,
        model_name: str | None = None,
    ):
        self.name = name
        self.llm = llm or self._build_llm(model_name)
        self.problem: Problem | None = None

    def build_package(
        self,
        spec: ProblemSpec,
        template_dir: Path = Path("./template"),
        output_dir: Path = Path("./output"),
        bench_number: int = 5,
        threads: int = 1,
        language: str | None = None,
    ) -> BenchArtifacts:
        target_dir = output_dir.joinpath(spec.problem_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        sampler_path = self.materialize_sampler(spec=spec, target_dir=target_dir)
        description_path = target_dir.joinpath("description.md")
        description_path.write_text(spec.to_problem_markdown(), encoding="utf-8")

        problem = self.prepare_problem(
            spec=spec,
            sampler_path=sampler_path,
            template_dir=template_dir,
            output_dir=output_dir,
            bench_number=bench_number,
            threads=threads,
            language=language,
        )
        self.generate_data(problem)
        zip_path = self.archive_problem(problem)
        return BenchArtifacts(
            sampler_path=sampler_path,
            description_path=description_path,
            zip_path=zip_path,
            output_dir=target_dir,
        )

    def materialize_sampler(self, spec: ProblemSpec, target_dir: Path) -> Path:
        sampler_code = self.generate_sampler_code(spec)
        sampler_path = target_dir.joinpath("sampler.py")
        sampler_path.write_text(sampler_code.rstrip() + "\n", encoding="utf-8")
        return sampler_path

    def prepare_problem(
        self,
        spec: ProblemSpec,
        sampler_path: Path,
        template_dir: Path = Path("./template"),
        output_dir: Path = Path("./output"),
        bench_number: int = 5,
        threads: int = 1,
        language: str | None = None,
    ) -> Problem:
        problem = Problem(
            name=spec.problem_id,
            bench_number=bench_number,
            threads=threads,
            sampler=sampler_path,
            solver=template_dir,
            language=language,
            output_path=output_dir,
        )
        problem.set_problem_description(spec.to_problem_markdown())
        problem.describe()
        self.problem = problem
        return problem

    def generate_data(
        self,
        problem: Problem,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        problem.generate(progress_callback=progress_callback)

    def archive_problem(self, problem: Problem) -> Path:
        return problem.archive()

    def generate_sampler_code(self, spec: ProblemSpec) -> str:
        llm_code = self._generate_sampler_with_llm(spec)
        valid, error_message = self._validate_sampler_code(llm_code)
        if valid:
            return llm_code

        repaired = self._repair_sampler_with_llm(spec, llm_code, error_message)
        valid, repaired_error = self._validate_sampler_code(repaired)
        if valid:
            return repaired
        raise RuntimeError(
            "BenchAgent failed to generate a valid sampler with LLM. "
            f"initial_error={error_message}; repaired_error={repaired_error}"
        )

    def _build_llm(self, model_name: str | None) -> BaseChatModel:
        if model_name is None:
            model_name = os.environ.get("BENCH_AGENT_MODEL")
        if not model_name:
            raise ValueError(
                "BenchAgent requires an LLM model. "
                "Pass `llm` explicitly or set `BENCH_AGENT_MODEL`."
            )
        return init_chat_model(model_name)

    def _generate_sampler_with_llm(self, spec: ProblemSpec) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You generate cyaron input generator scripts.\n"
                    "Return only python code.\n"
                    "The script should print one valid test case to stdout.\n"
                    "Do not generate outputs, only inputs.",
                ),
                (
                    "human",
                    "Problem ID: {problem_id}\n\n"
                    "Statement:\n{statement}\n\n"
                    "Input schema:\n{input_schema}\n\n"
                    "Output schema:\n{output_schema}\n\n"
                    "Constraints:\n{constraints}\n\n"
                    "Hints:\n{hints}\n\n"
                    "Samples:\n{samples}",
                ),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke(
            {
                "problem_id": spec.problem_id,
                "statement": spec.rewritten_statement_md,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "constraints": "\n".join(f"- {line}" for line in spec.constraints),
                "hints": "\n".join(f"- {hint}" for hint in spec.generator_hints),
                "samples": "\n\n".join(
                    f"Input:\n{s.input}\nOutput:\n{s.output}" for s in spec.sample_io
                )
                or "(none)",
            }
        )

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        if not isinstance(content, str):
            raise RuntimeError("BenchAgent LLM returned non-text sampler output")
        return self._strip_code_fence(content)

    def _repair_sampler_with_llm(
        self,
        spec: ProblemSpec,
        previous_code: str,
        error_message: str,
    ) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Fix the cyaron sampler code.\nReturn only corrected python code.",
                ),
                (
                    "human",
                    "Problem ID: {problem_id}\n\n"
                    "Input schema:\n{input_schema}\n\n"
                    "Constraints:\n{constraints}\n\n"
                    "Previous code:\n{previous_code}\n\n"
                    "Runtime error:\n{error_message}",
                ),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke(
            {
                "problem_id": spec.problem_id,
                "input_schema": spec.input_schema,
                "constraints": "\n".join(f"- {line}" for line in spec.constraints),
                "previous_code": previous_code,
                "error_message": error_message,
            }
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        if not isinstance(content, str):
            raise RuntimeError("BenchAgent LLM returned non-text repaired sampler")
        return self._strip_code_fence(content)

    def _validate_sampler_code(self, code: str) -> tuple[bool, str]:
        python_bin = "python3"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir).joinpath("sampler.py")
                script_path.write_text(code.rstrip() + "\n", encoding="utf-8")
                process = subprocess.run(
                    [python_bin, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
        except Exception as exc:
            return False, str(exc)
        if process.returncode != 0:
            return False, process.stderr.strip() or "sampler execution failed"
        if not process.stdout.strip():
            return False, "sampler output is empty"
        return True, ""

    def _strip_code_fence(self, content: str) -> str:
        content = content.strip()
        code_match = re.search(r"```(?:python)?\n(?P<body>[\s\S]*?)```", content)
        if code_match:
            return code_match.group("body").strip()
        return content
