"""Utilities to load run and scenario definitions from YAML files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

from pydantic import BaseModel, ValidationError
from dotenv import dotenv_values
import yaml
from code_agent_experiments.domain.models import RunConfig, Scenario
__all__ = [
    "LoadError",
    "LoadResult",
    "load_environment",
    "load_run_configs",
    "load_scenarios",
]

YamlScalar = str | int | float | bool | None
YamlValue = YamlScalar | list["YamlValue"] | dict[str, "YamlValue"]
YamlMapping = dict[str, YamlValue]
YamlPayload = list[YamlMapping] | YamlMapping

class LoadError(RuntimeError):
    """Raised when configuration payloads fail to load."""

@dataclass(slots=True)
class LoadResult[T: BaseModel]:
    """Represents the result of loading configuration items from disk."""

    items: list[T]
    source: Path

    def __iter__(self) -> Iterator[T]:
        """Allow iterating directly over the loaded items."""
        return iter(self.items)

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of loaded items."""
        return len(self.items)

def load_environment(
    env_files: Sequence[str | Path] | None = None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load environment values from one or more ``.env`` style files."""
    combined: dict[str, str] = dict(base or {})
    if env_files is None:
        return combined

    for env_file in env_files:
        path = Path(env_file)
        if not path.exists():
            message = f"Environment file not found: {path}"
            raise LoadError(message)
        values = dotenv_values(path, encoding="utf-8")
        combined.update({k: v for k, v in values.items() if v is not None})
    return combined

def load_run_configs(
    yaml_path: str | Path,
    *,
    env_files: Sequence[str | Path] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> LoadResult[RunConfig]:
    """Load ``RunConfig`` definitions from a YAML document."""
    payload = _load_yaml(yaml_path)
    env_values = load_environment(env_files, base=overrides)
    resolved = _apply_env(payload, env_values)
    items = [
        _validate_item(RunConfig, mapping)
        for mapping in _select_sequence(resolved, key="runs", context="Run configuration")
    ]
    return LoadResult(items=items, source=Path(yaml_path))

def load_scenarios(
    yaml_path: str | Path,
    *,
    env_files: Sequence[str | Path] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> LoadResult[Scenario]:
    """Load ``Scenario`` definitions from a YAML document."""
    payload = _load_yaml(yaml_path)
    env_values = load_environment(env_files, base=overrides)
    resolved = _apply_env(payload, env_values)
    items = [
        _validate_item(Scenario, mapping)
        for mapping in _select_sequence(resolved, key="scenarios", context="Scenario")
    ]
    return LoadResult(items=items, source=Path(yaml_path))

def _load_yaml(yaml_path: str | Path) -> YamlPayload:
    path = Path(yaml_path)
    if not path.exists():
        message = f"Configuration file not found: {path}"
        raise LoadError(message)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem errors are rare
        message = f"Failed to read configuration file: {path}"
        raise LoadError(message) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - bubbled as LoadError
        message = f"Invalid YAML in configuration file: {path}"
        raise LoadError(message) from exc
    if raw is None:
        return []
    if isinstance(raw, list):
        return _coerce_root_list(cast("list[Any]", raw), path)
    if isinstance(raw, dict):
        mapping = _ensure_str_key_mapping(cast("dict[Any, Any]", raw), path)
        return _coerce_mapping(mapping, path)
    message = f"Unsupported YAML structure in {path}"
    raise LoadError(message)


def _apply_env(payload: YamlPayload, env: Mapping[str, str]) -> YamlPayload:
    if isinstance(payload, list):
        return [_apply_env_mapping(item, env) for item in payload]
    return _apply_env_mapping(payload, env)


def _coerce_root_list(items: list[Any], source: Path) -> list[YamlMapping]:
    entries: list[YamlMapping] = []
    for item in items:
        if not isinstance(item, dict):
            message = f"Configuration entries must be mappings in {source}"
            raise LoadError(message)
        mapping = _ensure_str_key_mapping(cast("dict[Any, Any]", item), source)
        entries.append(_coerce_mapping(mapping, source))
    return entries



def _coerce_mapping(value: dict[str, Any], source: Path) -> YamlMapping:
    coerced: YamlMapping = {}
    for key, raw_value in value.items():
        coerced[key] = _coerce_value(raw_value, source)
    return coerced



def _coerce_value(value: Any, source: Path) -> YamlValue:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        value_list = cast("list[Any]", value)
        return [_coerce_value(item, source) for item in value_list]
    if isinstance(value, dict):
        mapping = _ensure_str_key_mapping(cast("dict[Any, Any]", value), source)
        return _coerce_mapping(mapping, source)
    message = f"Unsupported YAML value {value!r} in {source}"
    raise LoadError(message)


def _ensure_str_key_mapping(value: dict[Any, Any], source: Path) -> dict[str, Any]:
    for key in value:
        if not isinstance(key, str):
            message = f"Configuration keys must be strings in {source}"
            raise LoadError(message)
    return cast("dict[str, Any]", value)


def _apply_env_mapping(mapping: YamlMapping, env: Mapping[str, str]) -> YamlMapping:
    return {key: _apply_env_value(value, env) for key, value in mapping.items()}


def _apply_env_value(value: YamlValue, env: Mapping[str, str]) -> YamlValue:
    if isinstance(value, str):
        return _substitute_env(value, env)
    if isinstance(value, list):
        return [_apply_env_value(item, env) for item in value]
    if isinstance(value, dict):
        return _apply_env_mapping(value, env)
    return value


def _select_sequence(payload: YamlPayload, *, key: str, context: str) -> list[YamlMapping]:
    if isinstance(payload, list):
        return [_expect_mapping(item, context) for item in payload]
    if key not in payload:
        message = f"{context} payload must be a list of items or contain a '{key}' key"
        raise LoadError(message)
    value = payload[key]
    if not isinstance(value, list):
        message = f"{context} payload '{key}' must be a list"
        raise LoadError(message)
    return [_expect_mapping(item, context) for item in value]


def _expect_mapping(value: YamlValue, context: str) -> YamlMapping:
    if not isinstance(value, dict):
        message = f"{context} entries must be mappings"
        raise LoadError(message)
    return value


def _substitute_env(value: str, env: Mapping[str, str]) -> str:
    result: list[str] = []
    idx = 0
    while idx < len(value):
        start = value.find("${", idx)
        if start == -1:
            result.append(value[idx:])
            break
        result.append(value[idx:start])
        end = value.find("}", start + 2)
        if end == -1:
            message = f"Malformed environment placeholder in '{value}'"
            raise LoadError(message)
        placeholder = value[start + 2 : end]
        key, default = _split_placeholder(placeholder)
        if key in env:
            result.append(env[key])
        elif default is not None:
            result.append(default)
        else:
            message = f"Missing environment variable '{key}' referenced in configuration"
            raise LoadError(message)
        idx = end + 1
    return "".join(result)
def _split_placeholder(placeholder: str) -> tuple[str, str | None]:
    if ":-" in placeholder:
        key, default = placeholder.split(":-", 1)
        return key, default
    return placeholder, None


def _validate_item[T: BaseModel](model_cls: type[T], payload: YamlMapping) -> T:
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - validation errors mapped to LoadError
        message = str(exc)
        raise LoadError(message) from exc
