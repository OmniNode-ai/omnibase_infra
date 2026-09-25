# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Duplicate host-port check for a resolved catalog stack (OMN-19497).

docker compose publishes each ``ports`` entry on the Docker host. Two services
in one project that publish the same host port cannot both start: the second
fails with "port is already allocated" at ``up`` time, after images are built
and the stack is half up. Until OMN-19497 nothing checked this, and the
resolved ``runtime`` bundle mapped host port 8091 to two services and 8097 to
three. This check runs over the resolved manifests, before any compose file is
written, so a clash is refused at render time instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class HostPortClashResult:
    """Every host port claimed by more than one service, with its claimants."""

    clashes: dict[int, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.clashes

    def messages(self) -> list[str]:
        return [
            f"host port {port} is published by {', '.join(names)}"
            for port, names in sorted(self.clashes.items())
        ]


def find_duplicate_host_ports(
    manifests: Mapping[str, CatalogManifest],
) -> HostPortClashResult:
    """Return every host port that more than one resolved entry publishes."""
    claimants: dict[int, list[str]] = {}
    for name, manifest in manifests.items():
        if manifest.ports is None:
            continue
        claimants.setdefault(manifest.ports.external, []).append(name)
    return HostPortClashResult(
        clashes={
            port: sorted(names) for port, names in claimants.items() if len(names) > 1
        }
    )


__all__ = ["HostPortClashResult", "find_duplicate_host_ports"]
