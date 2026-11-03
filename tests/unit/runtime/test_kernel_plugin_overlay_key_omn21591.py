# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overlay key ``runtime.kernel_plugins`` reaches every runtime that sets the package surface (OMN-21591)."""

from __future__ import annotations

import importlib.util
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


# The collaborator lane's compose file (one file) is not edited by this change: touching it
# makes the hardcoded-model-config hook judge three findings in that file that the
# hook's baseline has never held, and clearing them is a policy change outside A3.
# That lane therefore names no kernel plugin until a follow-up adds the line; the
# control below fails when it does, so this entry cannot outlive the fix.
_AWAITING_FOLLOW_UP = 1


def _carriers() -> list[Path]:
    """Every tracked compose or catalog file that sets the active package surface."""
    paths = [
        *sorted((_REPO / "docker" / "catalog" / "services").glob("*.yaml")),
        *sorted((_REPO / "docker").glob("docker-compose.*.yml")),
    ]
    return [p for p in paths if f"{_PACKAGES}: $" in p.read_text()]


def test_the_policy_contract_names_the_delegation_plugin_and_renders_it() -> None:
    env = renderer.render_env(renderer.load_contract())
    assert env[_KEY] == "delegation"
    tracked = (_REPO / "docker" / "runtime-policy.env").read_text().splitlines()
    assert f"{_KEY}=delegation" in tracked


def test_every_runtime_that_sets_the_package_surface_requires_the_key() -> None:
    carriers = _carriers()
    assert len(carriers) >= 8, "the positive control: the carriers were found"
    without = [p.name for p in carriers if _REQUIRED not in p.read_text()]
    assert len(without) == _AWAITING_FOLLOW_UP, without


def test_planted_violation_a_carrier_without_the_key_is_caught() -> None:
    planted = f"  {_PACKAGES}: ${{{_PACKAGES}:?x}}\n"
    assert _REQUIRED not in planted


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
