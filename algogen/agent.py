from __future__ import annotations

import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from algogen.utils.bench_agent import BenchAgent, Problem
from algogen.utils.desc_agent import DescAgent
from algogen.utils.problem_spec import ProblemSpec

_DEFAULT_GROUP_SIZE = 5
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineState:
    problem_id: str
    db_dir: Path
    template_dir: Path
    output_dir: Path
    bench_number: int
    threads: int
    language: str | None
    max_group_rounds: int
    rewrite_artifacts: Any | None = None
    spec: ProblemSpec | None = None
    sampler_path: Path | None = None
    assert_solver_path: Path | None = None
    problem: Problem | None = None
    accepted_cases: int = 0
    group_round: int = 0
    zip_path: Path | None = None
    crashed_reasons: list[str] | None = None


class MultiAgentWorkflow:
    """State-machine workflow for desc->bench package generation."""

    def __init__(
        self,
        desc_agent: DescAgent | None = None,
        bench_agent: BenchAgent | None = None,
        desc_model_name: str | None = None,
        bench_model_name: str | None = None,
    ) -> None:
        self.desc_agent = desc_agent or DescAgent(model_name=desc_model_name)
        self.bench_agent = bench_agent or BenchAgent(model_name=bench_model_name)
        self.graph = self._build_graph()
        self.garbage: list[Path] = []

    def run(
        self,
        problem_id: str,
        db_dir: Path = Path("./db"),
        template_dir: Path = Path("./template"),
        output_dir: Path = Path("./output"),
        bench_number: int = 5,
        threads: int = 1,
        language: str | None = None,
        max_group_rounds: int = 50,
    ) -> PipelineState:
        self.garbage = []
        state = PipelineState(
            problem_id=problem_id,
            db_dir=db_dir,
            template_dir=template_dir,
            output_dir=output_dir,
            bench_number=bench_number,
            threads=threads,
            language=language,
            max_group_rounds=max_group_rounds,
        )

        logger.info(
            "✍️ 重写问题 problem_id=%s db_dir=%s output_dir=%s",
            state.problem_id,
            state.db_dir,
            state.output_dir,
        )
        rewrite_updates = self._rewrite_node(state)
        self._apply_updates(state, rewrite_updates)

        try:
            while True:
                if state.group_round >= state.max_group_rounds:
                    reasons = state.crashed_reasons or []
                    reason_preview = "; ".join(reasons[:3]) if reasons else "unknown"
                    raise RuntimeError(
                        "Exceeded max CEGIS group rounds "
                        f"({state.max_group_rounds}). Last failures: {reason_preview}"
                    )

                state.group_round += 1
                logger.info(
                    "🛠️ 修复/生成采样器 group=%s/%s",
                    state.group_round,
                    state.max_group_rounds,
                )

                write_updates = self._write_sampler_node(state)
                self._apply_updates(state, write_updates)
                logger.info(
                    "🧠 合成断言求解器/生成数据 group=%s/%s",
                    state.group_round,
                    state.max_group_rounds,
                )
                data_updates = self._generate_data_node(state)
                self._apply_updates(state, data_updates)
                status = self._validate_data_node(state)
                if status == "accepted":
                    logger.info(
                        "✅ 完成: 合成断言求解器/生成数据 group=%s accepted=%s",
                        state.group_round,
                        state.accepted_cases,
                    )
                    break

                logger.warning(
                    "🔄 重试: 修复/生成采样器 group=%s reason=sampler regeneration required",
                    state.group_round,
                )

            logger.info("🏁 阶段开始: 归档问题")
            archive_updates = self._archive_node(state)
            self._apply_updates(state, archive_updates)
            return state
        finally:
            logger.info("🧹 阶段开始: 清理临时文件")
            cleanup_updates = self._cleanup_node(state)
            self._apply_updates(state, cleanup_updates)

    def _build_graph(self):
        graph_builder = StateGraph(PipelineState)
        graph_builder.add_node("rewrite", self._rewrite_node)
        graph_builder.add_node("write_sampler", self._write_sampler_node)
        graph_builder.add_node("generate_data", self._generate_data_node)
        graph_builder.add_node("archive", self._archive_node)
        graph_builder.add_node("cleanup", self._cleanup_node)

        graph_builder.add_edge(START, "rewrite")
        graph_builder.add_edge("rewrite", "write_sampler")
        graph_builder.add_edge("write_sampler", "generate_data")
        graph_builder.add_conditional_edges(
            "generate_data",
            self._validate_data_node,
            {
                "accepted": "archive",
                "crashed": "write_sampler",
            },
        )
        graph_builder.add_edge("archive", "cleanup")
        graph_builder.add_edge("cleanup", END)
        return graph_builder.compile()

    def _rewrite_node(self, state: PipelineState) -> dict[str, Any]:
        artifacts = self.desc_agent.rewrite_problem(
            problem_id=state.problem_id,
            db_dir=state.db_dir,
            output_dir=state.output_dir,
        )
        target_dir = state.output_dir.joinpath(artifacts.spec.problem_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        sampler_path = self.bench_agent.ensure_sampler(target_dir=target_dir)
        problem = self.bench_agent.prepare_problem(
            spec=artifacts.spec,
            sampler_path=sampler_path,
            template_dir=state.template_dir,
            output_dir=state.output_dir,
            bench_number=state.bench_number,
            threads=state.threads,
            language=state.language,
        )
        assert_solver_path = self.bench_agent.ensure_solver_with_assert(
            solver_path=problem.solver_path
        )
        self.garbage.append(artifacts.spec_path)
        self.garbage.append(sampler_path)
        return {
            "spec": artifacts.spec,
            "problem": problem,
            "sampler_path": sampler_path,
            "assert_solver_path": assert_solver_path,
        }

    def _write_sampler_node(self, state: PipelineState) -> dict[str, Any]:
        if state.spec is None:
            raise RuntimeError("Missing `spec` in workflow state")
        if state.assert_solver_path is None:
            raise RuntimeError("Missing `solver_with_assert_path` in workflow state")
        if state.sampler_path is None:
            raise RuntimeError("Missing `sampler_path` in workflow state")
        sampler_code = self.bench_agent.generate_sampler_code(
            spec=state.spec,
            crashed_reasons=state.crashed_reasons,
        )
        state.sampler_path.write_text(sampler_code.rstrip() + "\n", encoding="utf-8")
        return {"sampler_path": state.sampler_path}

    def _generate_data_node(self, state: PipelineState) -> dict[str, Any]:
        if state.spec is None:
            raise RuntimeError("Missing `spec` in workflow state")
        if state.problem is None:
            raise RuntimeError("Missing `problem` in workflow state")
        logger.debug("任务开始: merge_assert_to_solver")
        self.bench_agent.merge_assert_to_solver(spec=state.spec)
        logger.debug("任务开始: generate_data")
        crashed_reasons = self.bench_agent.generate_data(problem=state.problem)
        accepted_cases = len(state.problem.input_and_output)
        if crashed_reasons:
            logger.warning(
                "任务结果: generate_data crashed accepted=%s failed=%s",
                accepted_cases,
                len(crashed_reasons),
            )
        else:
            logger.debug("任务完成: generate_data accepted=%s", accepted_cases)
        return {"crashed_reasons": crashed_reasons, "accepted_cases": accepted_cases}

    def _validate_data_node(self, state: PipelineState) -> str:
        if state.crashed_reasons is not None:
            return "crashed"
        return "accepted"

    def _archive_node(self, state: PipelineState) -> dict[str, Any]:
        if state.problem is None:
            raise RuntimeError("Missing `problem` in workflow state")
        zip_path = self.bench_agent.archive_problem(state.problem)
        return {"zip_path": zip_path}

    def _cleanup_node(self, state: PipelineState) -> dict[str, Any]:
        removed = 0
        for path in self.garbage:
            if path.exists() and path.is_file():
                os.remove(path)
                removed += 1
        if state.problem is None:
            raise RuntimeError("Missing `problem` in workflow state")
        if state.assert_solver_path is not None and state.assert_solver_path.exists():
            os.remove(state.problem.solver_path)
            logger.debug("替换 solver 为 solver-assert")
        logger.debug("任务完成: cleanup removed_files=%s", removed)
        return {}

    def _apply_updates(self, state: PipelineState, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if not key:
                continue
            setattr(state, key, value)
