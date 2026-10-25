# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve deployment data through the existing workspace configuration owner."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import yaml

RELATIVE_PATH = Path("config/onex/overlays/deploy_lane_routing")
CONTRACT_PATH = Path(__file__).resolve().parents[3] / "config/deploy_lane_routing.yaml"


def contract_digest(text: str) -> str:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PrivateRoutingConfigError(
            "deployment routing contract is not YAML"
        ) from exc
    digest = raw.get("overlay_sha256") if isinstance(raw, dict) else None
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PrivateRoutingConfigError(
            "deployment routing contract requires overlay_sha256; deployment values must come from ONEX_WORKSPACE_CONFIG_ROOT"
        )
    return digest


class PrivateRoutingConfigError(RuntimeError):
    """The deployment routing overlay cannot be resolved or read."""


def routing_table_path(
    digest: str | None = None, *, environ: Mapping[str, str]
) -> Path:
    digest = digest or contract_digest(CONTRACT_PATH.read_text(encoding="utf-8"))
    configured = environ.get("ONEX_WORKSPACE_CONFIG_ROOT", "").strip()
    if configured:
        root = Path(configured).resolve()
    else:
        registry = environ.get("OMNI_HOME", "").strip()
        if not registry:
            raise PrivateRoutingConfigError(
                "set ONEX_WORKSPACE_CONFIG_ROOT to the owner of config/lab/deploy_lane_routing.yaml; no deployment routing overlay is configured"
            )
        try:
            from omnibase_infra.runtime.service_kernel import (
                workspace_runtime_config_root,
            )

            root = workspace_runtime_config_root(Path(registry))
        except ImportError as exc:
            raise PrivateRoutingConfigError(
                "set ONEX_WORKSPACE_CONFIG_ROOT; this interpreter cannot resolve the configuration owner from OMNI_HOME"
            ) from exc
    path = root / RELATIVE_PATH / f"{digest}.yaml"
    if not path.is_file():
        raise PrivateRoutingConfigError(
            f"deployment routing overlay missing at {path}; set ONEX_WORKSPACE_CONFIG_ROOT to its owner"
        )
    return path


def read_routing_text(digest: str | None = None, *, environ: Mapping[str, str]) -> str:
    digest = digest or contract_digest(CONTRACT_PATH.read_text(encoding="utf-8"))
    path = routing_table_path(digest, environ=environ)
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise PrivateRoutingConfigError(
                f"deployment routing overlay digest mismatch at {path}; check ONEX_WORKSPACE_CONFIG_ROOT"
            )
        return data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PrivateRoutingConfigError(
            f"deployment routing overlay unreadable at {path}; check ONEX_WORKSPACE_CONFIG_ROOT"
        ) from exc
