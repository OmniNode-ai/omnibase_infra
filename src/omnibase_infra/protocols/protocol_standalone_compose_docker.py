# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The docker calls the standalone compose recreate effect needs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class StandaloneComposeDockerError(RuntimeError):
    """Compose exited non-zero or timed out while recreating the service."""


@runtime_checkable
class ProtocolStandaloneComposeDocker(Protocol):
    """Read every container on the host, and recreate one compose service."""

    def list_containers(self) -> dict[str, str]:
        """Every container on the host, running or not: name -> full container id."""
        ...

    def recreate_service(self, compose_file: str, env_file: str, service: str) -> None:
        """Recreate ``service`` alone: no dependencies, forced, from ``compose_file``.

        Raises ``StandaloneComposeDockerError`` when compose exits non-zero or times out.
        """
        ...

    def resolve_env_file(self, variable: str) -> str | None:
        """The compose env file path the host declares under ``variable``, or None."""
        ...
