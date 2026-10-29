# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Kafka ACL calls the broker principal effect needs (OMN-20822)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from omnibase_infra.models.developer_principal.model_broker_acl_binding import (
    ModelBrokerAclBinding,
)


@runtime_checkable
class ProtocolBrokerPrincipalAcls(Protocol):
    """Read and create one principal's ALLOW ACLs."""

    def list_acls(self, principal: str) -> frozenset[ModelBrokerAclBinding]:
        """Every ALLOW binding the broker holds for ``User:<principal>``."""
        ...

    def create_acls(
        self, principal: str, bindings: Iterable[ModelBrokerAclBinding]
    ) -> None:
        """Create the bindings; raise unless every one was created."""
        ...
