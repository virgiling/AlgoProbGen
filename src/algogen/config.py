from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Mapping

from dotenv import load_dotenv


class ConfigError(ValueError):
    """Configuration related errors."""


_PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "xai": "XAI_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ollama": None,
}


@dataclass(frozen=True, slots=True)
class AppConfig:
    desc_agent_model: str
    bench_agent_model: str
    db_dir: Path
    template_dir: Path
    output_dir: Path
    bench_number: int
    threads: int
    language: str | None
    env_file: Path

    @classmethod
    def from_env(
        cls,
        env_file: Path = Path(".env"),
        overrides: Mapping[str, str | int | Path | None] | None = None,
    ) -> "AppConfig":
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)

        merged_env = dict(os.environ)
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            merged_env[key] = str(value)

        desc_agent_model = _read_required(
            merged_env,
            "DESC_AGENT_MODEL",
            env_file=env_file,
        )
        bench_agent_model = _read_required(
            merged_env,
            "BENCH_AGENT_MODEL",
            env_file=env_file,
        )

        config = cls(
            desc_agent_model=desc_agent_model,
            bench_agent_model=bench_agent_model,
            db_dir=Path(merged_env.get("DB_DIR", "./db")),
            template_dir=Path(merged_env.get("TEMPLATE_DIR", "./template")),
            output_dir=Path(merged_env.get("OUTPUT_DIR", "./output")),
            bench_number=_read_positive_int(merged_env, "BENCH_NUMBER", default=5),
            threads=_read_positive_int(merged_env, "THREADS", default=1),
            language=_read_optional_language(merged_env, "LANGUAGE"),
            env_file=env_file,
        )
        config.validate_provider_api_keys(merged_env)
        return config

    def validate_provider_api_keys(self, env: Mapping[str, str]) -> None:
        missing: list[str] = []
        for model_name in (self.desc_agent_model, self.bench_agent_model):
            provider = _extract_provider(model_name)
            if provider is None:
                continue
            key_env = _PROVIDER_API_KEY_ENV.get(provider)
            if key_env is None:
                continue
            if not (env.get(key_env) or "").strip():
                missing.append(f"{key_env} (required by model `{model_name}`)")

        if missing:
            deduplicated = list(dict.fromkeys(missing))
            raise ConfigError(
                "Missing API key(s) for configured model providers: "
                + ", ".join(deduplicated)
            )


def _read_required(env: Mapping[str, str], key: str, env_file: Path) -> str:
    value = (env.get(key) or "").strip()
    if value:
        return value
    raise ConfigError(
        f"Missing required config `{key}`. "
        f"Please set it in environment or `{env_file}`."
    )


def _read_positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"Config `{key}` must be an integer, got `{raw}`") from exc
    if value <= 0:
        raise ConfigError(f"Config `{key}` must be > 0, got `{value}`")
    return value


def _extract_provider(model_name: str) -> str | None:
    cleaned = model_name.strip()
    if ":" not in cleaned:
        return None
    provider = cleaned.split(":", maxsplit=1)[0].strip().lower()
    return provider or None


def _read_optional_language(env: Mapping[str, str], key: str) -> str | None:
    raw = (env.get(key) or "").strip().lower()
    if not raw or raw == "auto":
        return None
    aliases = {
        "py": "python",
        "cc": "cpp",
    }
    value = aliases.get(raw, raw)
    allowed = {"python", "cpp", "c", "java"}
    if value not in allowed:
        raise ConfigError(
            f"Config `{key}` must be one of auto/python/cpp/c/java, got `{raw}`"
        )
    return value
