# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a prebuilt node manifest was built from, and the comparison that decides it is still valid (OMN-26361)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPrebuiltManifestStamp(BaseModel):
    """The installed distributions, their versions and every contract's content hash.

    ``fingerprint`` is the ``onex.nodes`` entry-point listing as
    ``(entry point, distribution, version)``, active or not. ``contract_hashes`` is
    ``(contract path, canonical content hash)`` for every contract the build read.
    ``active_packages`` and ``gateway_cloud_mirroring`` are the two environment
    switches the discovery skip predicates read, and ``workload`` names the announcer
    identity the catalog carries, so a manifest built under one setting is never
    served under another.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: tuple[tuple[str, str, str], ...] = Field(default=())
    contract_hashes: tuple[tuple[str, str], ...] = Field(default=())
    active_packages: str = Field(
        default="", description="Sorted comma-joined active package names; empty = all"
    )
    gateway_cloud_mirroring: bool = Field(default=False)
    workload: str = Field(
        default="main",
        description="Announcement workload the catalog's announcer identity was built for",
    )

    def differing_fields(self, current: ModelPrebuiltManifestStamp) -> tuple[str, ...]:
        """Name every field of this (stored) stamp that ``current`` no longer matches.

        Each entry names the field and the value that moved, so the boot log says
        which stamp field differed and which distribution or contract moved it.
        """
        out: list[str] = []
        stored = {(ep, dist): ver for ep, dist, ver in self.fingerprint}
        now = {(ep, dist): ver for ep, dist, ver in current.fingerprint}
        for key in sorted(stored.keys() | now.keys()):
            if key not in now:
                out.append(f"distribution_removed:{key[1]}:{key[0]}")
            elif key not in stored:
                out.append(f"distribution_added:{key[1]}:{key[0]}")
            elif stored[key] != now[key]:
                out.append(
                    f"distribution_version:{key[1]}:{key[0]}:{stored[key]}->{now[key]}"
                )
        have = dict(self.contract_hashes)
        seen = dict(current.contract_hashes)
        for path in sorted(have):
            if seen.get(path) != have[path]:
                out.append(f"contract_hash:{path}")
        if self.active_packages != current.active_packages:
            out.append(
                f"active_packages:{self.active_packages!r}->{current.active_packages!r}"
            )
        if self.gateway_cloud_mirroring != current.gateway_cloud_mirroring:
            out.append("gateway_cloud_mirroring")
        if self.workload != current.workload:
            out.append(f"workload:{self.workload!r}->{current.workload!r}")
        return tuple(out)
