# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overlay key ``runtime.kernel_plugins`` reaches every runtime that sets the package surface (OMN-21591)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "render_runtime_policy_env", _REPO / "scripts" / "render_runtime_policy_env.py"
)
assert _SPEC is not None and _SPEC.loader is not None
renderer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(renderer)

pytestmark = pytest.mark.unit

_KEY = "ONEX_RUNTIME_KERNEL_PLUGINS"
_PACKAGES = "ONEX_ACTIVE_RUNTIME_PACKAGES"
_REQUIRED = f"${{{_KEY}:?runtime policy contract must set {_KEY}}}"
_ANCHOR = re.compile(r"^x-[\w-]*runtime-env: &[\w-]+$")


def _carriers() -> list[Path]:
    """Every tracked compose or catalog file that sets the active package surface."""
    paths = [
        *sorted((_REPO / "docker" / "catalog" / "services").glob("*.yaml")),
        *sorted((_REPO / "docker").glob("docker-compose.*.yml")),
    ]
    return [p for p in paths if f"{_PACKAGES}: $" in p.read_text()]


def _anchor_block(lines: list[str], header_index: int) -> str:
    """The text of one top-level compose block, up to the next top-level key."""
    block: list[str] = []
    for line in lines[header_index + 1 :]:
        if line and not line.startswith((" ", "#")):
            break
        block.append(line)
    return "\n".join(block)


def _runtime_env_anchors() -> dict[str, str]:
    """Every ``x-*runtime-env`` anchor block of a compose file, keyed ``file::anchor``.

    Read as text: the lane files use compose's ``!override`` tag, which
    ``yaml.safe_load`` rejects.
    """
    anchors: dict[str, str] = {}
    for path in sorted((_REPO / "docker").glob("docker-compose.*.yml")):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if _ANCHOR.match(line):
                name = line.split(":")[0]
                anchors[f"{path.name}::{name}"] = _anchor_block(lines, index)
    return anchors


def test_the_policy_contract_names_the_delegation_plugin_and_renders_it() -> None:
    env = renderer.render_env(renderer.load_contract())
    assert env[_KEY] == "delegation"
    tracked = (_REPO / "docker" / "runtime-policy.env").read_text().splitlines()
    assert f"{_KEY}=delegation" in tracked


def test_every_runtime_that_sets_the_package_surface_requires_the_key() -> None:
    carriers = _carriers()
    assert len(carriers) >= 8, "the positive control: the carriers were found"
    without = [p.name for p in carriers if _REQUIRED not in p.read_text()]
    assert without == [], without


def test_every_lane_owned_runtime_env_anchor_that_sets_the_packages_has_the_key() -> (
    None
):
    """A lane that builds its own env anchor inherits nothing and must carry the key.

    The kernel no longer scans ``onex.domain_plugins``; a lane whose runtime
    env sets the package surface but not the key never registers the
    delegation plugin and logs no error.
    """
    anchors = _runtime_env_anchors()
    setting = {k: v for k, v in anchors.items() if f"  {_PACKAGES}: $" in v}
    assert len(setting) >= 4, f"the positive control: {sorted(anchors)}"
    needle = f"  {_KEY}: {_REQUIRED}"
    without = sorted(k for k, v in setting.items() if needle not in v)
    assert without == [], without


def test_planted_violation_a_carrier_without_the_key_is_caught() -> None:
    planted = f"  {_PACKAGES}: ${{{_PACKAGES}:?x}}\n"
    assert _REQUIRED not in planted


def test_planted_violation_an_anchor_without_the_key_is_caught() -> None:
    planted = [
        "x-planted-runtime-env: &planted",
        f"  {_PACKAGES}: ${{{_PACKAGES}:?x}}",
        "other: 1",
    ]
    assert _ANCHOR.match(planted[0])
    block = _anchor_block(planted, 0)
    assert f"  {_PACKAGES}: $" in block
    assert f"  {_KEY}: {_REQUIRED}" not in block


def test_the_catalog_services_parse_with_the_key_in_hardcoded_env() -> None:
    checked = 0
    for path in sorted((_REPO / "docker" / "catalog" / "services").glob("*.yaml")):
        text = path.read_text()
        if _REQUIRED not in text:
            continue
        env = yaml.safe_load(text)["hardcoded_env"]
        assert env[_KEY] == _REQUIRED, path.name
        checked += 1
    assert checked >= 8
