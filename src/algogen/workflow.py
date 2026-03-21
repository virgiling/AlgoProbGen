from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from .bench_agent import BenchAgent
from .desc_agent import DescAgent
from .problem_spec import ProblemSpec


@dataclass(slots=True)
class PipelineState:
    problem_id: str
    db_dir: Path
    template_dir: Path
    output_dir: Path
    bench_number: int
    threads: int
    language: str | None
    rewrite_artifacts: Any | None = None
    spec: ProblemSpec | None = None
    sampler_path: Path | None = None
    problem: Any | None = None
    zip_path: Path | None = None


@dataclass(slots=True)
class ProgressEvent:
    level: Literal["stage", "subtask"]
    name: str
    status: Literal["start", "update", "done", "error"]
    current: int | None = None
    total: int | None = None
    message: str | None = None
    case_index: int | None = None


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

    def run(
        self,
        problem_id: str,
        db_dir: Path = Path("./db"),
        template_dir: Path = Path("./template"),
        output_dir: Path = Path("./output"),
        bench_number: int = 5,
        threads: int = 1,
        language: str | None = None,
    ) -> PipelineState:
        state = PipelineState(
            problem_id=problem_id,
            db_dir=db_dir,
            template_dir=template_dir,
            output_dir=output_dir,
            bench_number=bench_number,
            threads=threads,
            language=language,
        )
        return self.graph.invoke(state)

    def run_with_progress(
        self,
        problem_id: str,
        db_dir: Path = Path("./db"),
        template_dir: Path = Path("./template"),
        output_dir: Path = Path("./output"),
        bench_number: int = 5,
        threads: int = 1,
        language: str | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> PipelineState:
        state = PipelineState(
            problem_id=problem_id,
            db_dir=db_dir,
            template_dir=template_dir,
            output_dir=output_dir,
            bench_number=bench_number,
            threads=threads,
            language=language,
        )
        return self._run_sequential(state, on_progress=on_progress)

    def _build_graph(self):
        graph_builder = StateGraph(PipelineState)
        graph_builder.add_node("rewrite", self._rewrite_node)
        graph_builder.add_node("extract_spec", self._extract_spec_node)
        graph_builder.add_node("gen_sampler", self._gen_sampler_node)
        graph_builder.add_node("generate_data", self._generate_data_node)
        graph_builder.add_node("solve_data", self._solve_data_node)
        graph_builder.add_node("archive", self._archive_node)

        graph_builder.add_edge(START, "rewrite")
        graph_builder.add_edge("rewrite", "extract_spec")
        graph_builder.add_edge("extract_spec", "gen_sampler")
        graph_builder.add_edge("gen_sampler", "generate_data")
        graph_builder.add_edge("generate_data", "solve_data")
        graph_builder.add_edge("solve_data", "archive")
        graph_builder.add_edge("archive", END)
        return graph_builder.compile()

    def _run_sequential(
        self,
        state: PipelineState,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> PipelineState:
        steps: list[tuple[str, Callable[[PipelineState], dict[str, Any]]]] = [
            ("rewrite", self._rewrite_node),
            ("extract_spec", self._extract_spec_node),
            ("gen_sampler", self._gen_sampler_node),
            ("generate_data", self._generate_data_node),
            ("solve_data", self._solve_data_node),
            ("archive", self._archive_node),
        ]
        total_steps = len(steps)

        for index, (name, handler) in enumerate(steps, start=1):
            self._emit_progress(
                on_progress,
                ProgressEvent(
                    level="stage",
                    name=name,
                    status="start",
                    current=index,
                    total=total_steps,
                ),
            )
            try:
                if name == "generate_data":
                    updates = self._generate_data_node(state, on_progress=on_progress)
                else:
                    updates = handler(state)
            except Exception as exc:
                self._emit_progress(
                    on_progress,
                    ProgressEvent(
                        level="stage",
                        name=name,
                        status="error",
                        current=index,
                        total=total_steps,
                        message=str(exc),
                    ),
                )
                raise
            self._apply_updates(state, updates)
            self._emit_progress(
                on_progress,
                ProgressEvent(
                    level="stage",
                    name=name,
                    status="done",
                    current=index,
                    total=total_steps,
                ),
            )
        return state

    def _rewrite_node(self, state: PipelineState) -> dict[str, Any]:
        artifacts = self.desc_agent.rewrite_problem(
            problem_id=state.problem_id,
            db_dir=state.db_dir,
            output_dir=state.output_dir,
        )
        return {
            "rewrite_artifacts": artifacts,
            "spec": artifacts.spec,
        }

    def _extract_spec_node(self, state: PipelineState) -> dict[str, Any]:
        if state.spec is None:
            raise RuntimeError("Missing `spec` in workflow state")
        return {"spec": state.spec}

    def _gen_sampler_node(self, state: PipelineState) -> dict[str, Any]:
        if state.spec is None:
            raise RuntimeError("Missing `spec` in workflow state")
        spec = state.spec
        target_dir = state.output_dir.joinpath(spec.problem_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        sampler_path = self.bench_agent.materialize_sampler(spec, target_dir=target_dir)
        return {"sampler_path": sampler_path}

    def _generate_data_node(
        self,
        state: PipelineState,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> dict[str, Any]:
        if state.spec is None:
            raise RuntimeError("Missing `spec` in workflow state")
        if state.sampler_path is None:
            raise RuntimeError("Missing `sampler_path` in workflow state")
        problem = self.bench_agent.prepare_problem(
            spec=state.spec,
            sampler_path=state.sampler_path,
            template_dir=state.template_dir,
            output_dir=state.output_dir,
            bench_number=state.bench_number,
            threads=state.threads,
            language=state.language,
        )

        def _subtask_progress_callback(payload: dict[str, Any]) -> None:
            self._emit_progress(
                on_progress,
                ProgressEvent(
                    level="subtask",
                    name="data_generation",
                    status=payload.get("status", "update"),
                    current=payload.get("current"),
                    total=payload.get("total"),
                    message=payload.get("message"),
                    case_index=payload.get("case_index"),
                ),
            )

        progress_callback = _subtask_progress_callback if on_progress else None
        self.bench_agent.generate_data(problem, progress_callback=progress_callback)
        return {"problem": problem}

    def _solve_data_node(self, state: PipelineState) -> dict[str, Any]:
        if state.problem is None:
            raise RuntimeError("Missing `problem` in workflow state")
        expected_count = state.bench_number
        actual_count = len(state.problem.input_and_output)
        if actual_count != expected_count:
            raise RuntimeError(f"Expected {expected_count} data pairs, got {actual_count}")
        return {}

    def _archive_node(self, state: PipelineState) -> dict[str, Any]:
        if state.problem is None:
            raise RuntimeError("Missing `problem` in workflow state")
        zip_path = self.bench_agent.archive_problem(state.problem)
        return {"zip_path": zip_path}

    def _apply_updates(self, state: PipelineState, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            setattr(state, key, value)

    def _emit_progress(
        self,
        callback: Callable[[ProgressEvent], None] | None,
        event: ProgressEvent,
    ) -> None:
        if callback is not None:
            callback(event)
