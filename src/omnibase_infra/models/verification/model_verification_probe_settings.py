# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Contract-declared timeouts for the verification probes (OMN-21623).

The values live in the ``config.probe_settings`` block of the co-located
``verification/contract.yaml``. The model has no field defaults, so a block
that omits a timeout fails to load rather than falling back to a literal.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.verification.probes.util_rpk_env import rpk_env

_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "verification" / "contract.yaml"


class ModelVerificationProbeSettings(BaseModel):
    """Timeouts, in seconds, that the verification probes run under.

    Attributes:
        rpk_command_timeout_seconds: Wait for one ``rpk`` describe or list call.
        rpk_discovery_timeout_seconds: Wait for the ``rpk`` group listing that
            discovers a contract's node identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    rpk_command_timeout_seconds: float = Field(gt=0)
    rpk_discovery_timeout_seconds: float = Field(gt=0)

    @classmethod
    def from_contract(cls, contract_path: Path) -> ModelVerificationProbeSettings:
        """Load the settings from ``config.probe_settings`` of a contract file.

        Raises:
            ProtocolConfigurationError: If the contract has no readable
                ``config.probe_settings`` mapping.
        """
        with contract_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        config = document.get("config") if isinstance(document, dict) else None
        block = config.get("probe_settings") if isinstance(config, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_probe_settings",
            )
            raise ProtocolConfigurationError(
                f"{contract_path.name} has no config.probe_settings mapping",
                context=context,
            )
        return cls.model_validate(block)

    def run_rpk(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run ``rpk`` under the command timeout, with the broker environment.

        Raises:
            FileNotFoundError: If ``rpk`` is not on ``PATH``.
        """
        return subprocess.run(
            ["rpk", *args],
            capture_output=True,
            text=True,
            timeout=self.rpk_command_timeout_seconds,
            check=False,
            env=rpk_env(),
        )

    def run_rpk_discovery(
        self, args: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        """Run ``rpk`` under the discovery timeout, in the process environment."""
        return subprocess.run(
            ["rpk", *args],
            capture_output=True,
            text=True,
            timeout=self.rpk_discovery_timeout_seconds,
            check=False,
        )


@cache
def load_probe_settings() -> ModelVerificationProbeSettings:
    """Return the probe settings declared by the verification contract."""
    return ModelVerificationProbeSettings.from_contract(_CONTRACT_PATH)


__all__: list[str] = ["ModelVerificationProbeSettings", "load_probe_settings"]
