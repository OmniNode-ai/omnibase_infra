# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: a dependency may not require a newer omnibase-core than this repo pins.

Incident: omnimarket v0.4.316 declared ``omnibase-core>=0.47.46,<0.48.0`` while
this repository pinned ``omnibase-core==0.47.44`` in ``[tool.uv]
override-dependencies`` (commit 44542e2a0, OMN-21146). The runtime image
installs omnimarket from its dev branch on top of the locked core, so every
merge-queue head failed ``runtime-boot`` for about five hours while each
repository's own tests passed.

The check is a pure function over recorded requirement text. It is run on:

* the real recorded pair (omnimarket v0.4.316 requirement, this repository's
  pin at 44542e2a0), where it must refuse, as the positive control;
* this repository's current ``pyproject.toml`` pin and ``uv.lock`` entry,
  against every downstream requirement recorded under
  ``tests/fixtures/pin_compat/``.

Raising a downstream floor above this pin is refused here; the order that
works is core release, then this repository's pin and lock, then the
downstream floor.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "pin_compat"

# Downstream requirements recorded from a real release, one fixture each.
DOWNSTREAM_REQUIREMENT_FIXTURES = ("omnimarket_v0.4.316_core_requirement.toml",)
INCIDENT_DOWNSTREAM = FIXTURES / "omnimarket_v0.4.316_core_requirement.toml"
INCIDENT_PIN = FIXTURES / "omnibase_infra_44542e2a0_core_pin.toml"

_CORE = "omnibase-core"
_SPEC = re.compile(r"^\s*omnibase[-_]core\s*(?P<op>==|>=)\s*(?P<version>\d+(?:\.\d+)*)")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _core_bound(requirements: list[str]) -> tuple[str, tuple[int, ...]]:
    """Return ``(operator, version)`` of the one omnibase-core entry in the list."""
    found = [m for m in map(_SPEC.match, requirements) if m is not None]
    assert len(found) == 1, f"expected exactly one {_CORE} entry, got {requirements}"
    return found[0].group("op"), _version_tuple(found[0].group("version"))


def required_floor(requirements: list[str]) -> tuple[int, ...]:
    """The lowest omnibase-core a package accepts: the version of its ``>=`` entry."""
    op, version = _core_bound(requirements)
    assert op == ">=", f"{_CORE} requirement is not a >= floor: {requirements}"
    return version


def pinned_version(overrides: list[str]) -> tuple[int, ...]:
    """The omnibase-core version a repository pins: its ``==`` override."""
    op, version = _core_bound(overrides)
    assert op == "==", f"{_CORE} override is not an exact pin: {overrides}"
    return version


def refusals(pin: tuple[int, ...], downstream: dict[str, tuple[int, ...]]) -> list[str]:
    """Name every downstream whose required floor is newer than ``pin``."""
    shown = ".".join(map(str, pin))
    return [
        f"{name} requires {_CORE}>={'.'.join(map(str, floor))}, newer than pin {shown}"
        for name, floor in downstream.items()
        if floor > pin
    ]


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _downstream_floors() -> dict[str, tuple[int, ...]]:
    floors: dict[str, tuple[int, ...]] = {}
    for name in DOWNSTREAM_REQUIREMENT_FIXTURES:
        data = _load(FIXTURES / name)
        floors[name] = required_floor(data["project"]["dependencies"])
    return floors


def _lock_core_version(lock_text: str) -> tuple[int, ...]:
    packages = [p for p in tomllib.loads(lock_text)["package"] if p["name"] == _CORE]
    assert len(packages) == 1, f"uv.lock holds {len(packages)} {_CORE} entries"
    return _version_tuple(packages[0]["version"])


def test_recorded_incident_pair_is_refused() -> None:
    """Positive control: omnimarket v0.4.316 against the 44542e2a0 pin must refuse."""
    pin = pinned_version(_load(INCIDENT_PIN)["tool"]["uv"]["override-dependencies"])
    downstream = {
        INCIDENT_DOWNSTREAM.name: required_floor(
            _load(INCIDENT_DOWNSTREAM)["project"]["dependencies"]
        )
    }

    assert pin == (0, 47, 44)
    assert downstream[INCIDENT_DOWNSTREAM.name] == (0, 47, 46)
    assert refusals(pin, downstream) == [
        "omnimarket_v0.4.316_core_requirement.toml requires omnibase-core>=0.47.46, "
        "newer than pin 0.47.44"
    ]


def test_recorded_incident_lock_entry_matches_its_pin() -> None:
    """The recorded uv.lock entry resolved the same 0.47.44 the override named."""
    recorded = _load(INCIDENT_PIN)
    pin = pinned_version(recorded["tool"]["uv"]["override-dependencies"])
    locked = [p for p in recorded["package"] if p["name"] == _CORE]
    assert [_version_tuple(p["version"]) for p in locked] == [pin]


def test_equal_or_older_downstream_floor_is_accepted() -> None:
    """Negative control: a floor at or below the pin is not refused."""
    assert refusals((0, 47, 46), {"at-pin": (0, 47, 46), "below": (0, 47, 44)}) == []


def test_repo_pin_is_not_older_than_any_recorded_downstream_floor() -> None:
    """The current pyproject pin satisfies every recorded downstream requirement."""
    overrides = _load(PYPROJECT)["tool"]["uv"]["override-dependencies"]
    floors = _downstream_floors()
    assert floors, "no recorded downstream requirement was loaded"
    assert refusals(pinned_version(overrides), floors) == []


def test_resolved_lock_is_not_older_than_any_recorded_downstream_floor() -> None:
    """The resolved uv.lock core satisfies every recorded downstream requirement."""
    locked = _lock_core_version(UV_LOCK.read_text(encoding="utf-8"))
    assert refusals(locked, _downstream_floors()) == []


def test_lock_resolves_the_version_pyproject_pins() -> None:
    """uv.lock and the [tool.uv] pin agree, so the pin is what the image installs."""
    overrides = _load(PYPROJECT)["tool"]["uv"]["override-dependencies"]
    assert _lock_core_version(UV_LOCK.read_text(encoding="utf-8")) == pinned_version(
        overrides
    )
