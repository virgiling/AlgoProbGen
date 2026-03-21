from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

from pydantic import BaseModel, Field

from ..problem_spec import (
    ProblemSpec,
    SampleIO,
    extract_constraints,
    normalize_statement_markdown,
    parse_problem_markdown,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class _RewriteResult(BaseModel):
    rewritten_statement_md: str = Field(description="Rewritten statement only.")
    generator_hints: list[str] = Field(
        default_factory=list,
        description="Hints for random input generation.",
    )


@dataclass(slots=True)
class RewriteArtifacts:
    spec: ProblemSpec
    description_path: Path
    spec_path: Path


class DescAgent:
    """Rewrite problem statements and extract structured spec."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        model_name: str | None = None,
    ) -> None:
        self.llm = llm or self._build_llm(model_name)

    def rewrite_problem(
        self,
        problem_id: str,
        db_dir: Path = Path("./db"),
        output_dir: Path = Path("./output"),
    ) -> RewriteArtifacts:
        problem_path = db_dir.joinpath(f"{problem_id}.md")
        if not problem_path.exists():
            raise FileNotFoundError(f"Problem markdown not found: {problem_path}")
        return self.rewrite_problem_file(problem_path=problem_path, output_dir=output_dir)

    def rewrite_problem_file(
        self,
        problem_path: Path,
        output_dir: Path = Path("./output"),
    ) -> RewriteArtifacts:
        problem_id = problem_path.stem
        markdown_text = problem_path.read_text(encoding="utf-8")
        parsed = parse_problem_markdown(markdown_text)
        if not parsed.statement:
            raise ValueError("Cannot find `题目描述` section in source markdown")
        if not parsed.input_format or not parsed.output_format:
            raise ValueError("Input/output sections are required")

        constraints = extract_constraints(parsed.input_format)
        default_hints = self._build_default_hints(
            statement=parsed.statement,
            input_schema=parsed.input_format,
            constraints=constraints,
        )
        rewrite_result = self._llm_rewrite(
            statement=parsed.statement,
            input_schema=parsed.input_format,
            output_schema=parsed.output_format,
            constraints=constraints,
            sample_io=parsed.samples,
            default_hints=default_hints,
        )
        rewritten_statement = normalize_statement_markdown(
            rewrite_result.rewritten_statement_md
        )
        if not rewritten_statement:
            raise RuntimeError("DescAgent LLM returned empty rewritten statement")
        hint_list = self._deduplicate(rewrite_result.generator_hints + default_hints)

        sample_io = [
            SampleIO(input=sample_input, output=sample_output)
            for sample_input, sample_output in parsed.samples
        ]
        spec = ProblemSpec(
            problem_id=problem_id,
            rewritten_statement_md=rewritten_statement,
            input_schema=parsed.input_format,
            output_schema=parsed.output_format,
            constraints=constraints,
            sample_io=sample_io,
            generator_hints=hint_list,
        )

        target_dir = output_dir.joinpath(problem_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        description_path = target_dir.joinpath("description.md")
        spec_path = target_dir.joinpath("spec.json")

        description_path.write_text(spec.to_problem_markdown(), encoding="utf-8")
        spec_path.write_text(
            spec.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return RewriteArtifacts(
            spec=spec,
            description_path=description_path,
            spec_path=spec_path,
        )

    def _build_llm(self, model_name: str | None) -> BaseChatModel:
        if model_name is None:
            model_name = os.environ.get("DESC_AGENT_MODEL")
        if not model_name:
            raise ValueError(
                "DescAgent requires an LLM model. "
                "Pass `llm` explicitly or set `DESC_AGENT_MODEL`."
            )
        return init_chat_model(model_name)

    def _llm_rewrite(
        self,
        statement: str,
        input_schema: str,
        output_schema: str,
        constraints: list[str],
        sample_io: list[tuple[str, str]],
        default_hints: list[str],
    ) -> _RewriteResult:
        structured_model = self.llm.with_structured_output(_RewriteResult)
        sample_text = "\n\n".join(
            f"[Sample {idx} Input]\n{sample_input}\n[Sample {idx} Output]\n{sample_output}"
            for idx, (sample_input, sample_output) in enumerate(sample_io, start=1)
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a contest problem rewrite assistant.\n"
                    "Rewrite only wording of the statement. Keep semantics unchanged.\n"
                    "Do not alter input/output meanings, constraints, or sample behavior.\n"
                    "Only return the statement section content.\n"
                    "Do NOT include sections like 输入格式/输出格式/样例/数据范围.\n"
                    "Use concise Chinese markdown text.",
                ),
                (
                    "human",
                    "Original statement:\n{statement}\n\n"
                    "Input format:\n{input_schema}\n\n"
                    "Output format:\n{output_schema}\n\n"
                    "Constraints:\n{constraints}\n\n"
                    "Samples:\n{sample_text}\n\n"
                    "Default generator hints:\n{default_hints}",
                ),
            ]
        )
        chain = prompt | structured_model
        response = chain.invoke(
            {
                "statement": statement,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "constraints": "\n".join(f"- {item}" for item in constraints),
                "sample_text": sample_text or "(none)",
                "default_hints": "\n".join(f"- {item}" for item in default_hints),
            }
        )
        if not isinstance(response, _RewriteResult):
            raise RuntimeError("DescAgent LLM output has unexpected schema")
        if not response.rewritten_statement_md.strip():
            raise RuntimeError("DescAgent LLM returned empty rewritten statement")
        return response

    def _build_default_hints(
        self,
        statement: str,
        input_schema: str,
        constraints: list[str],
    ) -> list[str]:
        hints: list[str] = [
            "Use cyaron to generate random but valid inputs.",
            "Cover both small and boundary cases.",
            "Generated input must strictly follow input schema.",
        ]
        if constraints:
            hints.append("Respect all numeric ranges mentioned in constraints.")
        if "进制" in statement or "进制" in input_schema:
            hints.extend(
                [
                    "Pick base uniformly in [2, 16].",
                    "Ensure digits are valid under the selected source base.",
                    "Keep decimal value <= 1e9 when converting between bases.",
                ]
            )
        return self._deduplicate(hints)

    def _deduplicate(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            clean_item = re.sub(r"\s+", " ", item).strip()
            if not clean_item or clean_item in seen:
                continue
            seen.add(clean_item)
            result.append(clean_item)
        return result
