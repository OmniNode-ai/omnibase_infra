# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The services the runtime policy folds into the main runtime (OMN-23513).

``docker/runtime-policy.env`` (rendered from the runtime policy contract) names, in
``DEV_RUNTIME_MAIN_FOLDED_PROFILES``, the profiles whose services the main runtime
carries, and in ``DEV_RUNTIME_MAIN_FOLDABLE_SERVICES`` the service each profile
maps to. A lane of the policy's compose project that still declares such a service
``kind: service`` would read it as required and report ``container_absent`` for a
container that is absent by design (the one-runtime ruling of 2026-11-02T15:02:01Z).
:func:`fold_manifest` reads that declaration as ``profile_gated``: absent is clean,
running is ``profile_gated_present``.

This module is standard library only. The collector (``scripts/lane_census_plan.py``)
runs under a bare interpreter on a lab host, where the runtime package is not
importable, so it loads this file by path. Functions raise ``ValueError``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

__all__ = ["ENV_RUNTIME_POLICY", "fold_manifest", "folded_service_names"]

#: The variable that names the runtime policy file; unset, the repository's own.
ENV_RUNTIME_POLICY = "LANE_CENSUS_RUNTIME_POLICY"
_DEFAULT_POLICY = Path(__file__).resolve().parents[4] / "docker" / "runtime-policy.env"
_PROJECT_KEY = "DEV_COMPOSE_PROJECT"
_FOLDED_KEY = "DEV_RUNTIME_MAIN_FOLDED_PROFILES"
_FOLDABLE_KEY = "DEV_RUNTIME_MAIN_FOLDABLE_SERVICES"


def _policy_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key.strip()] = value
    return values


def folded_service_names(path: Path) -> tuple[str, frozenset[str]]:
    """The policy's compose project and the short names of its folded services.

    A missing file folds nothing. A foldable map that is not a JSON object of
    strings, or a folded profile the map does not name, raises ``ValueError``.
    """
    if not path.is_file():
        return "", frozenset()
    values = _policy_values(path)
    project = values.get(_PROJECT_KEY, "")
    profiles = [p.strip() for p in values.get(_FOLDED_KEY, "").split(",") if p.strip()]
    foldable_raw = values.get(_FOLDABLE_KEY, "")
    try:
        foldable: object = json.loads(foldable_raw) if foldable_raw else {}
    except ValueError as exc:
        raise ValueError(f"{_FOLDABLE_KEY} is not valid JSON: {exc}") from exc
    if not isinstance(foldable, dict) or not all(
        isinstance(v, str) for v in foldable.values()
    ):
        raise ValueError(f"{_FOLDABLE_KEY} must be a JSON object of strings")
    unknown = [p for p in profiles if p not in foldable]
    if unknown:
        raise ValueError(
            f"{_FOLDED_KEY} names profile {unknown[0]!r} absent from {_FOLDABLE_KEY}"
        )
    return project, frozenset(foldable[p] for p in profiles)


def fold_manifest(
    manifest: dict[str, object], environ: Mapping[str, str]
) -> dict[str, object]:
    """A copy of ``manifest`` with the policy's folded services read as profile_gated."""
    explicit = environ.get(ENV_RUNTIME_POLICY, "").strip()
    project, shorts = folded_service_names(
        Path(explicit) if explicit else _DEFAULT_POLICY
    )
    folded = copy.deepcopy(manifest)
    lanes = folded.get("lanes")
    if not project or not isinstance(lanes, dict):
        return folded
    for spec in lanes.values():
        if not isinstance(spec, dict) or spec.get("compose_project") != project:
            continue
        services = spec.get("services", [])
        for svc in services if isinstance(services, list) else []:
            if (
                isinstance(svc, dict)
                and svc.get("kind", "service") == "service"
                and any(str(svc.get("name", "")).endswith(f"-{s}") for s in shorts)
            ):
                svc["kind"] = "profile_gated"
    return folded
