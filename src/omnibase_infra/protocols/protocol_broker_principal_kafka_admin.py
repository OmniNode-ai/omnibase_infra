# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The confluent_kafka ``AdminClient`` calls the broker principal clients use (OMN-20822).

Typed so the clients take a real ``AdminClient`` or a test double without ``Any``.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Protocol

from confluent_kafka.admin import AclBinding, AclBindingFilter


class ProtocolBrokerPrincipalKafkaAdmin(Protocol):
    """``create_acls`` and ``describe_acls`` of a confluent admin client."""

    def create_acls(
        self, acls: list[AclBinding], **kwargs: object
    ) -> dict[AclBinding, Future[object]]:
        """One future per binding."""
        ...

    def describe_acls(
        self, acl_binding_filter: AclBindingFilter, **kwargs: object
    ) -> Future[object]:
        """A future whose result is the list of bindings matching the filter."""
        ...
