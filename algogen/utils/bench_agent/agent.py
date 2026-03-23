from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Callable

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..problem_spec import ProblemSpec
from ..problem_spec.spec import SAMPLER_PROMPT, SAMPLER_TEMPLATE
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

    def ensure_sampler(self, target_dir: Path) -> Path:
        sampler_path = target_dir.joinpath("sampler.py")
        return sampler_path

    def ensure_solver_with_assert(self, solver_path: Path) -> Path:
        if not solver_path.exists():
            raise FileNotFoundError(f"Solver path not found: {solver_path}")
        if self.problem is None:
            raise ValueError("Problem is not set")
        solver_with_assert_path = solver_path.with_name(
            f"{self.problem.name}-assert"
        ).with_suffix(solver_path.suffix)
        self.solver_with_assert_code = solver_path.read_text(encoding="utf-8")
        self.solver_with_assert_path = solver_with_assert_path
        return solver_with_assert_path

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
    ) -> list[str] | None:
        has_crashed = problem.generate()
        if has_crashed:
            return problem.crashed_reasons
        return None

    def archive_problem(self, problem: Problem) -> Path:
        return problem.archive()

    def generate_sampler_code(
        self,
        spec: ProblemSpec,
        crashed_reasons: list[str] | None,
    ) -> str:
        llm_code = self._generate_sampler_with_llm(spec, crashed_reasons)
        return llm_code

    def merge_assert_to_solver(self, spec: ProblemSpec) -> None:
        self.solver_with_assert_code = self._generate_assert_to_solver_with_llm(spec)
        self.solver_with_assert_path.write_text(
            self.solver_with_assert_code, encoding="utf-8"
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

    def _generate_sampler_with_llm(
        self,
        spec: ProblemSpec,
        crashed_reasons: list[str] | None,
    ) -> str:
        constraints = "\n".join(f"- {line}" for line in spec.constraints) or "- (none)"
        crash_feedback = "\n".join(crashed_reasons) if crashed_reasons else "(none)"
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You write robust `cyaron` samplers for programming contest tasks.\n"
                    "Return Python code only.",
                ),
                ("human", SAMPLER_PROMPT),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke(
            {
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
                "constraints": constraints,
                "sampler_template": SAMPLER_TEMPLATE,
                "crashed_reason": crash_feedback,
            }
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        if not isinstance(content, str):
            raise RuntimeError("BenchAgent LLM returned non-text sampler output")
        return self._strip_code_fence(content)

    def _generate_assert_to_solver_with_llm(self, spec: ProblemSpec) -> str:
        constraints = "\n".join(f"- {line}" for line in spec.constraints) or "- (none)"
        std_code = self.solver_with_assert_code
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a program synthesis assistant.\n"
                    "Your task is to augment the given solver code by inserting runtime "
                    "assertions derived from problem constraints.\n"
                    "Map each constraint to concrete variables/expressions in the code, "
                    "even when names differ from the statement wording.\n"
                    "Carefully read the full code to identify where those variables are parsed "
                    "or computed, then place asserts at the earliest safe point after values are available.\n"
                    "Use assert syntax idiomatic to the solver language.\n"
                    "Preserve original algorithm logic and output behavior.\n"
                    "Do not remove existing code; only add necessary assertions (and tiny helper code if required).\n"
                    "If a constraint cannot be mapped with high confidence, keep code unchanged for that part.\n"
                    "Return code only.",
                ),
                (
                    "human",
                    "Standard solver code with {language}:\n{std_code}\n\n"
                    "Constraints:\n{constraints}\n\n"
                    "Return code only.",
                ),
            ]
        )
        chain = prompt | self.llm
        response = chain.invoke(
            {
                "language": self.problem.language if self.problem else None,
                "std_code": std_code,
                "constraints": constraints,
            }
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        if not isinstance(content, str):
            raise RuntimeError("BenchAgent LLM returned non-text assert solver output")
        return self._strip_code_fence(content)

    def _strip_code_fence(self, content: str) -> str:
        content = content.strip()
        code_match = re.search(r"```(?:\w+)?\n(?P<body>[\s\S]*?)```", content)
        if code_match:
            return code_match.group("body").strip()
        return content
