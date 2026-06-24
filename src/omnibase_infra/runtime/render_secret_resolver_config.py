# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render the deployed secret resolver config for runtime boot.

Lane overlays own the mapping from logical secret references to concrete secret
source descriptors. This renderer materializes that mapping as a typed
``ModelSecretResolverConfig`` artifact so runtime effects resolve logical
references through ``SecretResolver`` rather than treating environment variable
names as route authority.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)

_DEFAULT_TARGET_PATH = Path("/app/data/delegation/secret_resolver.yaml")


def _load_source_data(
    *,
    environ: Mapping[str, str],
    source_path: Path | None,
) -> dict[str, object]:
    inline_config = environ.get("ONEX_SECRET_RESOLVER_CONFIG_JSON", "").strip()
    if inline_config:
        try:
            data: object = json.loads(inline_config)
        except json.JSONDecodeError as exc:
            raise ProtocolConfigurationError(
                "ONEX_SECRET_RESOLVER_CONFIG_JSON must contain valid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise ProtocolConfigurationError(
                "ONEX_SECRET_RESOLVER_CONFIG_JSON must contain a mapping root"
            )
        return {str(key): value for key, value in data.items()}

    configured_source = environ.get(
        "ONEX_SECRET_RESOLVER_SOURCE_CONFIG_PATH", ""
    ).strip()
    source = source_path or (Path(configured_source) if configured_source else None)
    if source is None:
        raise ProtocolConfigurationError(
            "Secret resolver render requires ONEX_SECRET_RESOLVER_CONFIG_JSON or "
            "ONEX_SECRET_RESOLVER_SOURCE_CONFIG_PATH"
        )
    try:
        raw: object = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolConfigurationError(
            f"Secret resolver source config could not be loaded: {source}"
        ) from exc
    if not isinstance(raw, dict):
        raise ProtocolConfigurationError(
            f"Secret resolver source config must have a mapping root: {source}"
        )
    return {str(key): value for key, value in raw.items()}


def _validate_config_data(
    data: dict[str, object],
    *,
    path: Path,
) -> ModelSecretResolverConfig:
    try:
        config = ModelSecretResolverConfig.model_validate(data)
    except ValidationError as exc:
        raise ProtocolConfigurationError(
            f"Secret resolver config failed validation: {path}"
        ) from exc

    logical_names = [mapping.logical_name for mapping in config.mappings]
    duplicates = sorted(
        {
            logical_name
            for logical_name in logical_names
            if logical_names.count(logical_name) > 1
        }
    )
    if duplicates:
        raise ProtocolConfigurationError(
            f"Secret resolver config contains duplicate logical mappings: {duplicates}"
        )
    return config


def _resolve_target_path(
    *,
    target_path: Path | None,
    environ: Mapping[str, str],
) -> Path | None:
    if target_path is not None:
        return target_path
    configured_path = environ.get("ONEX_SECRET_RESOLVER_CONFIG_PATH")
    if configured_path is None:
        return _DEFAULT_TARGET_PATH
    stripped_path = configured_path.strip()
    if not stripped_path:
        return None
    return Path(stripped_path)


def render_secret_resolver_config(
    *,
    source_path: Path | None = None,
    target_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Render the deployed secret resolver config and return its path."""
    env = environ if environ is not None else os.environ
    target = _resolve_target_path(target_path=target_path, environ=env)
    if target is None:
        return None

    data = _load_source_data(environ=env, source_path=source_path)
    config = _validate_config_data(data, path=target)

    target.parent.mkdir(parents=True, exist_ok=True)
    staged_target = target.with_suffix(f"{target.suffix}.tmp")
    staged_target.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json", exclude_defaults=True),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    staged_target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    _validate_config_data(_load_yaml(staged_target), path=staged_target)
    staged_target.replace(target)
    target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return target


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolConfigurationError(
            f"Secret resolver config could not be loaded: {path}"
        ) from exc
    if not isinstance(raw, dict):
        raise ProtocolConfigurationError(
            f"Secret resolver config must have a mapping root: {path}"
        )
    return {str(key): value for key, value in raw.items()}


def main() -> int:
    rendered = render_secret_resolver_config()
    if rendered is None:
        sys.stdout.write(
            "[entrypoint] Secret resolver config rendering disabled: "
            "ONEX_SECRET_RESOLVER_CONFIG_PATH is empty\n"
        )
        return 0
    sys.stdout.write(f"[entrypoint] Secret resolver config ready: {rendered}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
