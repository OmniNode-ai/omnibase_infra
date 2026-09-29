# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Report environment overrides of delegation configuration paths."""

from __future__ import annotations

from collections.abc import Mapping

from omnibase_infra.cli.model_delegate_env_config_override import (
    ModelDelegateEnvConfigOverride,
)

DELEGATION_ENV_CONFIG_KEYS: tuple[str, ...] = (
    "BIFROST_CONTRACT_PATH",
    "BIFROST_OVERLAY_PATH",
    "DELEGATION_ROUTING_TIERS_PATH",
)


def env_config_overrides(
    env: Mapping[str, str],
) -> tuple[ModelDelegateEnvConfigOverride, ...]:
    """Return non-blank overrides in the declared configuration key order."""
    return tuple(
        ModelDelegateEnvConfigOverride(config_key=key, resolved_path=path)
        for key in DELEGATION_ENV_CONFIG_KEYS
        if (path := env.get(key, "").strip())
    )


def format_override_line(o: ModelDelegateEnvConfigOverride) -> str:
    """Describe an override for the CLI's stderr output."""
    return (
        f"config override: {o.config_key} resolved from environment "
        f"(source={o.source}) path={o.resolved_path}"
    )
