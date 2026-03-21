from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from pydantic import BaseModel, Field

_TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_CONSTRAINT_PATTERN = re.compile(r"\\(?:leq|geq|le|ge|lt|gt|in|neq)\b")
_CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\n(?P<body>[\s\S]*?)```")
_LEADING_STATEMENT_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s*题目描述\s*$")
_NON_STATEMENT_HEADING_PATTERN = re.compile(
    r"^\s*#{1,6}\s*(?:输入格式|输出格式|输入样例|输出样例|样例(?:输入|输出)?|样例|数据范围)\s*$",
    flags=re.MULTILINE,
)


@dataclass(slots=True)
class ParsedProblemMarkdown:
    statement: str
    input_format: str
    output_format: str
    samples: list[tuple[str, str]]


class SampleIO(BaseModel):
    input: str = Field(description="单组样例输入")
    output: str = Field(description="对应的样例输出")


class ProblemSpec(BaseModel):
    problem_id: str = Field(description="题目唯一 ID，通常是文件名（不含后缀）")
    rewritten_statement_md: str = Field(
        description="改写后的题目描述（Markdown），仅包含描述部分，无需标题"
    )
    input_schema: str = Field(description="输入格式描述")
    output_schema: str = Field(description="输出格式描述")
    constraints: list[str] = Field(
        default_factory=list,
        description="用于数据生成的约束条件",
    )
    sample_io: list[SampleIO] = Field(
        default_factory=list,
        description="样例输入输出",
    )
    generator_hints: list[str] = Field(
        default_factory=list,
        description="给数据生成器的提示",
    )

    def to_problem_markdown(self) -> str:
        statement_md = normalize_statement_markdown(self.rewritten_statement_md)
        lines: list[str] = [
            "# 题目描述",
            "",
            statement_md,
            "",
            "# 输入格式",
            "",
            self.input_schema.strip(),
            "",
            "# 输出格式",
            "",
            self.output_schema.strip(),
            "",
        ]

        if self.sample_io:
            first = self.sample_io[0]
            lines.extend(
                [
                    "# 输入样例",
                    "",
                    "```txt",
                    first.input.strip(),
                    "```",
                    "",
                    "# 输出样例",
                    "",
                    "```txt",
                    first.output.strip(),
                    "```",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"


def parse_markdown_sections(markdown_text: str) -> dict[str, str]:
    matches = list(_TITLE_PATTERN.finditer(markdown_text))
    sections: dict[str, str] = {}
    if not matches:
        return sections

    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(markdown_text)
        )
        sections[title] = markdown_text[start:end].strip()
    return sections


def extract_constraints(text: str) -> list[str]:
    candidates: list[str] = []
    for raw in re.split(r"[\n。；;]", text):
        line = raw.strip()
        if line and _CONSTRAINT_PATTERN.search(line):
            candidates.append(line)
    return _deduplicate(candidates)


def extract_code_block(text: str) -> str:
    match = _CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group("body").strip()
    return text.strip()


def parse_problem_markdown(markdown_text: str) -> ParsedProblemMarkdown:
    sections = parse_markdown_sections(markdown_text)
    statement = sections.get("题目描述", "").strip()
    input_format = sections.get("输入格式", "").strip()
    output_format = sections.get("输出格式", "").strip()

    input_sample = extract_code_block(sections.get("输入样例", "").strip())
    output_sample = extract_code_block(sections.get("输出样例", "").strip())
    samples: list[tuple[str, str]] = []
    if input_sample and output_sample:
        samples.append((input_sample, output_sample))

    return ParsedProblemMarkdown(
        statement=statement,
        input_format=input_format,
        output_format=output_format,
        samples=samples,
    )


def _deduplicate(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def normalize_statement_markdown(statement: str) -> str:
    normalized = statement.replace("\r\n", "\n").strip()
    if not normalized:
        return ""

    lines = normalized.splitlines()
    if lines and _LEADING_STATEMENT_HEADING_PATTERN.fullmatch(lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        normalized = "\n".join(lines).strip()

    stop_match = _NON_STATEMENT_HEADING_PATTERN.search(normalized)
    if stop_match:
        normalized = normalized[: stop_match.start()].rstrip()

    return normalized.strip()
