# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The host operations the stability integration refresh takes (OMN-26715)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ProtocolStabilityIntegrationHost(Protocol):
    """Git, Docker and HTTP reads and writes of the stability host."""

    def resolve_head(self, repository: str, branch: str) -> str:
        """Commit of ``origin/<branch>`` in the repository's canonical clone; ``""`` when absent."""
        ...

    def cut_deploy_tags(
        self, heads: Sequence[tuple[str, str]]
    ) -> tuple[tuple[str, str], ...]:
        """Cut ``deploy-<UTC stamp>-<sha12>`` on each ``(repository, commit)`` in the host's own clone; return ``(repository, tag)``.

        The tags are never pushed. Raises ``RuntimeError`` when a tag cannot be cut.
        """
        ...

    def build_image(
        self, heads: Sequence[tuple[str, str]], tags: Sequence[tuple[str, str]]
    ) -> str:
        """Build the runtime image from the deploy ``tags`` of the ``heads`` with stage_workspace.sh; return its image id.

        Raises ``RuntimeError`` with the build's refusal text when it fails.
        """
        ...

    def service_image_id(self, compose_project: str, service: str) -> str:
        """Image id the running service container uses; ``""`` when unreadable."""
        ...

    def recreate(
        self, compose_project: str, services: Sequence[tuple[str, str]]
    ) -> None:
        """Recreate each ``(service, image_id)`` on that image, keeping every volume."""
        ...

    def health_status(self, url: str) -> int:
        """HTTP status of ``url``; 0 when unreachable."""
        ...
