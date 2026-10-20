# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The nodes a deployment loads, read from its overlay (OMN-17427).

``ONEX_RUNTIME_NODE_SELECTION`` is a comma-separated list of contract names set
by a deployment overlay. Unset or blank is no selection: every discovered
contract is eligible, which is the lab default. Set, only the named contracts
are attached by the auto-wiring ownership filter and only their topics are
provisioned by the topic provisioner, so a plane that runs a handful of nodes
does not create and subscribe the topics of every installed node.

Read by
:func:`omnibase_infra.runtime.auto_wiring.profile_ownership.filter_manifest_for_runtime_profile`
and :class:`omnibase_infra.event_bus.service_topic_manager.TopicProvisioner`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

ENV_RUNTIME_NODE_SELECTION = "ONEX_RUNTIME_NODE_SELECTION"


class ModelRuntimeNodeSelection(BaseModel):
    """The contract names a deployment selects; empty selects every contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contract names the deployment loads, in declared order, deduplicated. "
            "Empty means no selection: every discovered contract is eligible."
        ),
    )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> ModelRuntimeNodeSelection:
        """Read the selection from ``ONEX_RUNTIME_NODE_SELECTION``."""
        source: Mapping[str, str] = os.environ if environ is None else environ
        raw = source.get(ENV_RUNTIME_NODE_SELECTION) or ""
        return cls(
            contracts=tuple(
                dict.fromkeys(part.strip() for part in raw.split(",") if part.strip())
            )
        )

    @property
    def is_active(self) -> bool:
        """Whether the deployment named any contract."""
        return bool(self.contracts)

    def admits(self, contract_name: str) -> bool:
        """Return whether a contract is loaded under this selection."""
        return not self.contracts or contract_name in self.contracts


__all__ = ["ENV_RUNTIME_NODE_SELECTION", "ModelRuntimeNodeSelection"]
