# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The confluent client call the broker principal session uses (OMN-20822)."""

from __future__ import annotations

from typing import Protocol


class ProtocolBrokerPrincipalTopicLister(Protocol):
    """``list_topics`` of a confluent client logged in as one principal."""

    def list_topics(self, *args: object, **kwargs: object) -> object:
        """Cluster metadata whose ``topics`` maps topic names the login can see."""
        ...
