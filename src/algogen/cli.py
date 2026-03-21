from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TextIO

from .config import AppConfig, ConfigError
from .workflow import MultiAgentWorkflow, ProgressEvent

_STAGE_LABELS: dict[str, str] = {
    "rewrite": "改写题面",
    "extract_spec": "抽取结构化信息",
    "gen_sampler": "生成数据脚本",
    "generate_data": "生成输入输出数据",
    "solve_data": "校验数据完整性",
    "archive": "打包压缩文件",
}

_STATUS_LABELS: dict[str, str] = {
    "start": "开始",
    "update": "进行中",
    "done": "完成",
    "error": "失败",
}


class _ProgressRenderer:
    def __init__(self, stream: TextIO = sys.stderr) -> None:
        self.stream = stream

    def __call__(self, event: ProgressEvent) -> None:
        if event.level == "stage":
            self._render_stage(event)
        elif event.level == "subtask":
            self._render_subtask(event)

    def _render_stage(self, event: ProgressEvent) -> None:
        total = event.total or 0
        current = event.current or 0
        done_count = max(current - 1, 0) if event.status == "start" else current
        label = _STAGE_LABELS.get(event.name, event.name)
        status_text = _STATUS_LABELS.get(event.status, event.status)
        bar = self._bar(done_count, total)
        message = (
            f"[stage] {bar} {done_count}/{total} {status_text}: {label}"
            if total > 0
            else f"[stage] {status_text}: {label}"
        )
        if event.message:
            message = f"{message} ({event.message})"
        print(message, file=self.stream, flush=True)

    def _render_subtask(self, event: ProgressEvent) -> None:
        total = event.total or 0
        current = event.current or 0
        status_text = _STATUS_LABELS.get(event.status, event.status)
        bar = self._bar(current, total)
        case_text = (
            f" case={event.case_index}"
            if event.case_index is not None and event.status in {"update", "error"}
            else ""
        )
        message = (
            f"\t[subtask] {bar} {current}/{total} {status_text}: 生成数据{case_text}"
            if total > 0
            else f"\t[subtask] {status_text}: 生成数据{case_text}"
        )
        if event.message:
            message = f"{message} ({event.message})"
        print(message, file=self.stream, flush=True)

    def _bar(self, current: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return "[" + ("-" * width) + "]"
        ratio = min(max(current / total, 0.0), 1.0)
        filled = int(ratio * width)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algogen",
        description="Multi-agent algorithm package generator CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run workflow for one problem id",
    )
    run_parser.add_argument("problem_id", help="Problem id, e.g. 1000")
    run_parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    run_parser.add_argument("--desc-model", help="Override DESC_AGENT_MODEL")
    run_parser.add_argument("--bench-model", help="Override BENCH_AGENT_MODEL")
    run_parser.add_argument("--db-dir", help="Override DB_DIR")
    run_parser.add_argument("--template-dir", help="Override TEMPLATE_DIR")
    run_parser.add_argument("--output-dir", help="Override OUTPUT_DIR")
    run_parser.add_argument(
        "--bench-number",
        type=int,
        help="Override BENCH_NUMBER (must be > 0)",
    )
    run_parser.add_argument(
        "--threads",
        type=int,
        help="Override THREADS (must be > 0)",
    )
    run_parser.add_argument("--language", help="Override LANGUAGE")
    run_parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show stage/subtask progress output (default: enabled)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _handle_run(parser, args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


def _handle_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.bench_number is not None and args.bench_number <= 0:
        parser.error("--bench-number must be > 0")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be > 0")

    overrides: dict[str, str | int | Path | None] = {
        "DESC_AGENT_MODEL": args.desc_model,
        "BENCH_AGENT_MODEL": args.bench_model,
        "DB_DIR": args.db_dir,
        "TEMPLATE_DIR": args.template_dir,
        "OUTPUT_DIR": args.output_dir,
        "BENCH_NUMBER": args.bench_number,
        "THREADS": args.threads,
        "LANGUAGE": args.language,
    }
    env_file = Path(args.env_file)
    try:
        config = AppConfig.from_env(env_file=env_file, overrides=overrides)
    except ConfigError as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        return 2

    workflow = MultiAgentWorkflow(
        desc_model_name=config.desc_agent_model,
        bench_model_name=config.bench_agent_model,
    )
    progress_renderer = _ProgressRenderer(stream=sys.stderr) if args.progress else None

    try:
        if progress_renderer is None:
            state = workflow.run(
                problem_id=args.problem_id,
                db_dir=config.db_dir,
                template_dir=config.template_dir,
                output_dir=config.output_dir,
                bench_number=config.bench_number,
                threads=config.threads,
                language=config.language,
            )
        else:
            state = workflow.run_with_progress(
                problem_id=args.problem_id,
                db_dir=config.db_dir,
                template_dir=config.template_dir,
                output_dir=config.output_dir,
                bench_number=config.bench_number,
                threads=config.threads,
                language=config.language,
                on_progress=progress_renderer,
            )
    except Exception as exc:
        print(f"[run-error] {exc}", file=sys.stderr)
        return 1

    if state.zip_path is None:
        print("[run-error] workflow completed without zip_path", file=sys.stderr)
        return 1

    print(f"problem_id={args.problem_id}")
    print(f"zip_path={state.zip_path}")
    print(f"output_dir={config.output_dir.joinpath(args.problem_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
