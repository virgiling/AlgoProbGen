from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .agent import MultiAgentWorkflow
from .config import AppConfig, ConfigError

_LOG_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_LOG_RESET = "\033[0m"


class _PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, datefmt="%H:%M:%S")
        level_name = record.levelname
        color = _LOG_COLORS.get(level_name, "")
        colored_level = f"{color}{level_name:<8}{_LOG_RESET}" if color else level_name
        return f"{timestamp} | {colored_level} | {record.name:.<16} | {record.getMessage()}"


class _AlgogenOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("algogen")


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_PrettyFormatter())
    handler.addFilter(_AlgogenOnlyFilter())
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("algogen").setLevel(level)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algogen",
        description="Multi-agent algorithm package generator CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run workflow for one problem id")
    run_parser.add_argument("problem_id", help="Problem id, e.g. 1000")
    run_parser.add_argument("--env-file", default=".env", help="Path to .env file")
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
        "--threads", type=int, help="Override THREADS (must be > 0)"
    )
    run_parser.add_argument(
        "--cegis-max-group-rounds",
        type=int,
        help="Override CEGIS_MAX_GROUP_ROUNDS (must be > 0)",
    )
    run_parser.add_argument("--language", help="Override LANGUAGE")
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logger verbosity (default: INFO)",
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
    _setup_logging(args.log_level)
    logger = logging.getLogger("algogen.cli")

    _validate_positive(parser, "--bench-number", args.bench_number)
    _validate_positive(parser, "--threads", args.threads)
    _validate_positive(parser, "--cegis-max-group-rounds", args.cegis_max_group_rounds)

    overrides: dict[str, str | int | Path | None] = {
        "DESC_AGENT_MODEL": args.desc_model,
        "BENCH_AGENT_MODEL": args.bench_model,
        "DB_DIR": args.db_dir,
        "TEMPLATE_DIR": args.template_dir,
        "OUTPUT_DIR": args.output_dir,
        "BENCH_NUMBER": args.bench_number,
        "THREADS": args.threads,
        "CEGIS_MAX_GROUP_ROUNDS": args.cegis_max_group_rounds,
        "LANGUAGE": args.language,
    }
    env_file = Path(args.env_file)
    try:
        config = AppConfig.from_env(env_file=env_file, overrides=overrides)
    except ConfigError as exc:
        logger.error("[config-error] %s", exc)
        return 2

    workflow = MultiAgentWorkflow(
        desc_model_name=config.desc_agent_model,
        bench_model_name=config.bench_agent_model,
    )
    logger.info(
        "🚀 启动任务: problem_id=%s, db_dir=%s, template_dir=%s, output_dir=%s, bench=%s, threads=%s, language=%s",
        args.problem_id,
        config.db_dir,
        config.template_dir,
        config.output_dir,
        config.bench_number,
        config.threads,
        config.language or "auto",
    )

    try:
        state = workflow.run(
            problem_id=args.problem_id,
            db_dir=config.db_dir,
            template_dir=config.template_dir,
            output_dir=config.output_dir,
            bench_number=config.bench_number,
            threads=config.threads,
            language=config.language,
            max_group_rounds=config.cegis_max_group_rounds,
        )
    except Exception as exc:
        logger.exception("[run-error] %s", exc)
        return 1

    if state.zip_path is None:
        logger.error("[run-error] workflow completed without zip_path")
        return 1

    logger.info("✅ 完成: problem_id=%s, zip_path=%s", args.problem_id, state.zip_path)
    print(f"problem_id={args.problem_id}")
    print(f"zip_path={state.zip_path}")
    print(f"output_dir={config.output_dir.joinpath(args.problem_id)}")
    return 0


def _validate_positive(
    parser: argparse.ArgumentParser,
    name: str,
    value: int | None,
) -> None:
    if value is not None and value <= 0:
        parser.error(f"{name} must be > 0")


if __name__ == "__main__":
    raise SystemExit(main())
