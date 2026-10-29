# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The nodes a deployment loads, read from its overlay (OMN-17427).

``ONEX_RUNTIME_NODE_SELECTION`` is a comma-separated list of contract names set
by a deployment overlay. Unset or blank is no selection: every discovered
contract is eligible, which is the lab default. Set, only the named contracts
are attached by the auto-wiring ownership filter and only their topics are
provisioned by the topic provisioner, so a plane that runs a handful of nodes
does not create and subscribe the topics of every installed node.

``ONEX_RUNTIME_NODE_EXCLUSION`` is the inverse list (OMN-19513): contract names
a deployment hands to another process. Unset or blank excludes nothing. Set, the
named contracts are left unattached by this process whatever the profile and
selection say, so a node that a dedicated container carries has exactly one
consumer-group owner. The two variables compose: a contract is loaded when the
selection admits it and the exclusion does not name it. Exclusion does not make
the selection "active", so topic provisioning still builds every topic.

Read by
:func:`omnibase_infra.runtime.auto_wiring.profile_ownership.filter_manifest_for_runtime_profile`
and :class:`omnibase_infra.event_bus.service_topic_manager.TopicProvisioner`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

ENV_RUNTIME_NODE_SELECTION = "ONEX_RUNTIME_NODE_SELECTION"
ENV_RUNTIME_NODE_EXCLUSION = "ONEX_RUNTIME_NODE_EXCLUSION"


def _parse_names(raw: str) -> tuple[str, ...]:
    """Split a comma-separated name list, dropping blanks and duplicates."""
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


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

    excluded: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contract names the deployment hands to another process, in declared "
            "order, deduplicated. Empty excludes nothing."
        ),
    )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> ModelRuntimeNodeSelection:
        """Read the selection and exclusion from the environment."""
        source: Mapping[str, str] = os.environ if environ is None else environ
        return cls(
            contracts=_parse_names(source.get(ENV_RUNTIME_NODE_SELECTION) or ""),
            excluded=_parse_names(source.get(ENV_RUNTIME_NODE_EXCLUSION) or ""),
        )

    @property
    def is_active(self) -> bool:
        """Whether the deployment named any contract."""
        return bool(self.contracts)

    def admits(self, contract_name: str) -> bool:
        """Return whether a contract is loaded under this selection and exclusion."""
        if contract_name in self.excluded:
            return False
        return not self.contracts or contract_name in self.contracts


__all__ = [
    "ENV_RUNTIME_NODE_EXCLUSION",
    "ENV_RUNTIME_NODE_SELECTION",
    "ModelRuntimeNodeSelection",
]
