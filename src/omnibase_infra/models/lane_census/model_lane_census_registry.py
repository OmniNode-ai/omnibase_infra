# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lab host registry the lane census reads (OMN-22152).

Which hosts exist for the census, the names each one reports and the lanes each
may host are overlay data of the lane census node, in the ``registry:`` key of
the overlay file the collector's ``LANE_CENSUS_REGISTRY`` names. The manifest
declares lanes and services only.

This module is standard library only. The collector (``scripts/lane_census_plan.py``)
runs under a bare interpreter on a lab host, where the runtime package is not
importable, so it loads this file by path; nothing here may import a third-party
or ``omnibase_infra`` module. Methods raise ``ValueError``; the collector maps
them to its own error classes and exit codes.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ModelLaneCensusRegistry", "registry_path"]

#: The node overlay a registry default is looked up in: the first
#: ``<root>/node_lane_census_effect/overlay.yaml`` under the overlay roots, the
#: lookup the lane census node itself makes.
_NODE_OVERLAY = Path("node_lane_census_effect") / "overlay.yaml"

#: The keys of one registry row. A row carries exactly these.
_ROW_KEYS = frozenset({"name", "aliases", "lanes"})

_Host = tuple[str, tuple[str, ...], tuple[str, ...]]


def registry_path(environ: Mapping[str, str]) -> str:
    """The registry file the collector reads (OMN-22152), from ``environ``.

    ``LANE_CENSUS_REGISTRY`` when it is set, taken as given: a path that is not a
    file is the caller's error to report, never silently replaced. Unset, the
    lane census node's own overlay, the first ``node_lane_census_effect/overlay.yaml``
    under ``ONEX_SKILL_OVERLAY_ROOTS``, so a caller that carries the overlay
    roots needs no second variable. Raises ``ValueError`` when neither names a file.
    """
    explicit = environ.get("LANE_CENSUS_REGISTRY", "").strip()
    if explicit:
        return explicit
    for root in environ.get("ONEX_SKILL_OVERLAY_ROOTS", "").split(os.pathsep):
        if root and (Path(root) / _NODE_OVERLAY).is_file():
            return str(Path(root) / _NODE_OVERLAY)
    raise ValueError(
        "LANE_CENSUS_REGISTRY is unset and no root of ONEX_SKILL_OVERLAY_ROOTS "
        f"holds {_NODE_OVERLAY}"
    )


def _short_name(name: str) -> str:
    """Lowercase first DNS label: ``Box.local`` and ``box.example.net`` -> ``box``."""
    return name.strip().lower().rstrip(".").split(".", 1)[0]


def _strings(value: object, where: str, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(i, str) for i in value):
        raise ValueError(f"{where}: {key} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class ModelLaneCensusRegistry:
    """Each host as ``(name, aliases, lanes)``; the host's lanes are what it may host."""

    hosts: tuple[_Host, ...]

    @classmethod
    def from_overlay(cls, data: object) -> ModelLaneCensusRegistry:
        """The registry of a parsed overlay file, from its ``registry:`` key."""
        if not isinstance(data, dict):
            raise ValueError("the file is not a mapping")
        return cls.from_rows(data.get("registry"))

    @classmethod
    def from_rows(cls, rows: object) -> ModelLaneCensusRegistry:
        """The registry of a parsed ``registry:`` list of name/aliases/lanes rows."""
        if not isinstance(rows, list):
            raise ValueError("the file has no `registry:` list")
        if not rows:
            raise ValueError("`registry:` declares no host")
        hosts: list[_Host] = []
        for position, row in enumerate(rows):
            where = f"`registry:` row {position}"
            if not isinstance(row, dict):
                raise ValueError(f"{where} is not a mapping")
            if set(row) != _ROW_KEYS:
                raise ValueError(
                    f"{where} must carry exactly {sorted(_ROW_KEYS)}, got "
                    f"{sorted(str(k) for k in row)}"
                )
            if not isinstance(row["name"], str):
                raise ValueError(f"{where}: name must be a string")
            hosts.append(
                (
                    row["name"],
                    _strings(row["aliases"], where, "aliases"),
                    _strings(row["lanes"], where, "lanes"),
                )
            )
        return cls(tuple(hosts))

    def as_rows(self) -> list[dict[str, object]]:
        """The ``registry:`` rows this registry was read from."""
        return [
            {"name": h, "aliases": list(a), "lanes": list(ls)}
            for h, a, ls in self.hosts
        ]

    def validate(
        self, lane_names: Iterable[str], ambiguous_names: Iterable[str]
    ) -> None:
        """Refuse a registry that cannot attribute a census to one host.

        Every name and alias is claimed by exactly one host, none is empty or
        identifies no single host, every manifest lane is listed by at least one
        host (a lane no host lists is evaluated nowhere: a silent clean), and
        every listed lane is a manifest lane. A host that lists no lane is valid.
        ``ambiguous_names`` are names that identify no single host (every Docker
        Desktop daemon reports the same one), which no host may declare.
        """
        lanes, ambiguous = set(lane_names), set(ambiguous_names)
        claimed: dict[str, str] = {}
        listed: set[str] = set()
        for position, (host_id, aliases, host_lanes) in enumerate(self.hosts):
            if any(host_id == other[0] for other in self.hosts[:position]):
                raise ValueError(f"host {host_id!r} is declared twice in the registry")
            for name in (host_id, *aliases):
                key = _short_name(name)
                if not key:
                    raise ValueError(f"host {host_id!r} declares an empty name")
                if key in ambiguous:
                    raise ValueError(
                        f"host {host_id!r} declares {name!r}, which identifies no "
                        "single host (every Docker Desktop daemon reports "
                        "'docker-desktop')"
                    )
                if claimed.setdefault(key, host_id) != host_id:
                    raise ValueError(
                        f"name {name!r} is claimed by both host {claimed[key]!r} "
                        f"and host {host_id!r}; a census there cannot be attributed"
                    )
            for lane in host_lanes:
                if lane not in lanes:
                    raise ValueError(
                        f"host {host_id!r} lists lane {lane!r}, which the lane "
                        "manifest does not declare"
                    )
            listed.update(host_lanes)
        for lane in sorted(lanes - listed):
            raise ValueError(
                f"lane {lane!r} is listed by no host of the registry; every lane "
                "must name the host(s) it may run on"
            )

    def resolve(self, raw: str) -> str:
        """The one host id a hostname, daemon name or host id names."""
        key = _short_name(raw or "")
        if not key:
            raise ValueError(
                "no host given: the census must be told which host it runs on "
                "(LANE_CENSUS_HOST, default `hostname`)"
            )
        for host_id, aliases, _ in self.hosts:
            if key in {_short_name(n) for n in (host_id, *aliases)}:
                return host_id
        raise ValueError(
            f"host {raw!r} is declared by no entry of the lane census registry. "
            "Declare it (with the lanes it may run) before running the census "
            "there; evaluating every lane on an undeclared host is the OMN-19088 "
            "defect"
        )

    def lanes_of(self, host_id: str) -> frozenset[str]:
        """The lanes ``host_id`` may host."""
        return frozenset(lane for h, _, ls in self.hosts if h == host_id for lane in ls)

    def hosts_of(self, lane: str) -> list[str]:
        """The hosts that list ``lane``."""
        return [h for h, _, ls in self.hosts if lane in ls]
