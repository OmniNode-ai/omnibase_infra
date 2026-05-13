# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Config loader for delegation runtime profile contracts (OMN-10923).

Reads a delegation-runtime-profile YAML and returns typed Pydantic models.
Zero env var reads — all configuration is sourced from the contract file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any  # ONEX_EXCLUDE: any_type

import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from omnibase_core.models.contracts.model_delegation_runtime_profile import (
        ModelDelegationRuntimeProfile,
    )


class DelegationProfileNotFoundError(Exception):
    """Raised when the delegation profile contract is missing or invalid."""


class DelegationProfileConfigLoader:
    """Reads a delegation runtime profile from a contract YAML file.

    Caches the parsed result after the first successful load.
    Raises DelegationProfileNotFoundError on missing file or schema violation.
    Never reads env vars for delegation configuration — contract is the sole source.

    Return types use Any because omnibase_core's delegation models are gated behind
    PR #1072 (OMN-10919) and not yet in the published package. Once merged, these
    signatures will be narrowed to ModelDelegationRuntimeProfile and sub-models.
    """

    def __init__(self, contract_path: Path) -> None:
        self._contract_path = contract_path
        # ONEX_EXCLUDE: any_type — narrowed to ModelDelegationRuntimeProfile after OMN-10919 merges
        self._profile: Any = None

    # ONEX_EXCLUDE: any_type — return narrowed to ModelDelegationRuntimeProfile after OMN-10919 merges
    def load(self) -> Any:
        """Load and validate the delegation runtime profile.

        Returns the cached profile on subsequent calls.
        """
        if self._profile is not None:
            return self._profile

        if not self._contract_path.exists():
            raise DelegationProfileNotFoundError(
                f"Delegation profile contract not found: {self._contract_path}"
            )

        raw = yaml.safe_load(self._contract_path.read_text(encoding="utf-8"))

        from omnibase_core.models.contracts.model_delegation_runtime_profile import (
            ModelDelegationRuntimeProfile,
        )

        try:
            self._profile = ModelDelegationRuntimeProfile.model_validate(raw)
        except ValidationError as exc:
            raise DelegationProfileNotFoundError(
                f"Invalid delegation profile contract at {self._contract_path}: {exc}"
            ) from exc

        return self._profile

    # ONEX_EXCLUDE: any_type — return narrowed to ModelDelegationEventBusEndpoint after OMN-10919 merges
    def event_bus_config(self) -> Any:
        """Return the event bus endpoint config from the loaded profile."""
        return self.load().event_bus

    # ONEX_EXCLUDE: any_type — return narrowed to dict[str, ModelDelegationLlmBackend] after OMN-10919 merges
    def llm_backend_config(self) -> Any:
        """Return the LLM backends dict from the loaded profile."""
        return self.load().llm_backends


__all__ = [
    "DelegationProfileConfigLoader",
    "DelegationProfileNotFoundError",
]
