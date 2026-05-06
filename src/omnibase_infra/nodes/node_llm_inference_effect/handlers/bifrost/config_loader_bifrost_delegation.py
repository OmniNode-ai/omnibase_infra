# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Loader for bifrost_delegation.yaml delegation routing config.

Reads and validates the delegation routing config from disk.
The config maps Claude Code task classes to bifrost backend policies.

Related:
    - OMN-10637: Bifrost routing rules for delegation task classes
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_delegation_config import (
    ModelBifrostDelegationConfig,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "configs"
    / "bifrost_delegation.yaml"
)


def load_bifrost_delegation_config(
    config_path: Path | None = None,
) -> ModelBifrostDelegationConfig:
    """Load and validate the bifrost delegation routing config from disk.

    Args:
        config_path: Path to the YAML config file. Defaults to the
            canonical ``src/omnibase_infra/configs/bifrost_delegation.yaml``.

    Returns:
        A validated ``ModelBifrostDelegationConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML cannot be parsed or fails schema validation.
    """
    resolved = config_path or _DEFAULT_CONFIG_PATH

    if not resolved.exists():
        msg = f"Bifrost delegation config not found at {resolved}"
        raise FileNotFoundError(msg)

    raw = resolved.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    if not isinstance(data, dict):
        msg = f"Expected YAML mapping at root, got {type(data).__name__}"
        raise ValueError(msg)

    config = ModelBifrostDelegationConfig.model_validate(data)

    logger.info(
        "Loaded bifrost delegation config v%s: %d backends, %d rules",
        config.config_version,
        len(config.backends),
        len(config.routing_rules),
    )
    return config


__all__: list[str] = ["load_bifrost_delegation_config"]
