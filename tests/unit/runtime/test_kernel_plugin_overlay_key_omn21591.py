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

# The collaborator lane's compose file is not edited by this change: touching it makes
# the hardcoded-model-config hook judge three lines in that file its baseline has never
# held, and one of them (the migration-gate keepalive) is the exact argv the stack-shape
# policy pins, so no form satisfies both without a policy change outside A3. That lane
# names no kernel plugin until a follow-up adds the line; the two assertions below name
# it exactly, so they fail when it is fixed and this entry cannot outlive the fix.
_AWAITING_FOLLOW_UP_FILE = "docker-compose.lakshman.yml"
_AWAITING_FOLLOW_UP_ANCHOR = f"{_AWAITING_FOLLOW_UP_FILE}::x-lakshman-runtime-env"


def _carriers(root: Path = _REPO) -> list[Path]:
    """Every compose or catalog file under ``root`` that sets the active package surface."""
    paths = [
        *sorted((root / "docker" / "catalog" / "services").glob("*.yaml")),
        *sorted((root / "docker").glob("docker-compose.*.yml")),
    ]
    return [p for p in paths if f"{_PACKAGES}: $" in p.read_text()]


def _carriers_without_the_key(root: Path = _REPO) -> list[str]:
    """The carrier files under ``root`` that do not require the key."""
    return [p.name for p in _carriers(root) if _REQUIRED not in p.read_text()]


def _anchor_block(lines: list[str], header_index: int) -> str:
    """The text of one top-level compose block, up to the next top-level key."""
    block: list[str] = []
    for line in lines[header_index + 1 :]:
        if line and not line.startswith((" ", "#")):
            break
        block.append(line)
    return "\n".join(block)


def _runtime_env_anchors(root: Path = _REPO) -> dict[str, str]:
    """Every ``x-*runtime-env`` anchor block of a compose file, keyed ``file::anchor``.

    Read as text: the lane files use compose's ``!override`` tag, which
    ``yaml.safe_load`` rejects.
    """
    anchors: dict[str, str] = {}
    for path in sorted((root / "docker").glob("docker-compose.*.yml")):
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            if _ANCHOR.match(line):
                name = line.split(":")[0]
                anchors[f"{path.name}::{name}"] = _anchor_block(lines, index)
    return anchors


def _package_anchors(root: Path = _REPO) -> dict[str, str]:
    """The runtime env anchors under ``root`` that set the package surface."""
    return {
        k: v for k, v in _runtime_env_anchors(root).items() if f"  {_PACKAGES}: $" in v
    }


def _anchors_without_the_key(root: Path = _REPO) -> list[str]:
    """The package-setting anchors under ``root`` that do not carry the key."""
    needle = f"  {_KEY}: {_REQUIRED}"
    return sorted(k for k, v in _package_anchors(root).items() if needle not in v)


def test_the_policy_contract_names_the_delegation_plugin_and_renders_it() -> None:
    env = renderer.render_env(renderer.load_contract())
    assert env[_KEY] == "delegation"
    tracked = (_REPO / "docker" / "runtime-policy.env").read_text().splitlines()
    assert f"{_KEY}=delegation" in tracked


def test_every_runtime_that_sets_the_package_surface_requires_the_key() -> None:
    assert len(_carriers()) >= 8, "the positive control: the carriers were found"
    without = _carriers_without_the_key()
    assert without == [_AWAITING_FOLLOW_UP_FILE], without


def test_every_lane_owned_runtime_env_anchor_that_sets_the_packages_has_the_key() -> (
    None
):
    """A lane that builds its own env anchor inherits nothing and must carry the key.

    The kernel no longer scans ``onex.domain_plugins``; a lane whose runtime
    env sets the package surface but not the key never registers the
    delegation plugin and logs no error.
    """
    setting = _package_anchors()
    assert len(setting) >= 4, f"the positive control: {sorted(_runtime_env_anchors())}"
    without = _anchors_without_the_key()
    assert without == [_AWAITING_FOLLOW_UP_ANCHOR], without


def _plant(root: Path, *, with_key: bool) -> None:
    """Write one catalog service and one lane compose file that set the packages."""
    key_line = f"  {_KEY}: {_REQUIRED}\n" if with_key else ""
    services = root / "docker" / "catalog" / "services"
    services.mkdir(parents=True)
    (services / "planted.yaml").write_text(
        f"hardcoded_env:\n  {_PACKAGES}: ${{{_PACKAGES}:?x}}\n{key_line}"
    )
    (root / "docker" / "docker-compose.planted.yml").write_text(
        "x-planted-runtime-env: &planted\n"
        f"  {_PACKAGES}: ${{{_PACKAGES}:?x}}\n{key_line}"
        "services: {}\n"
    )


def test_planted_violation_a_carrier_without_the_key_is_caught(
    tmp_path: Path,
) -> None:
    _plant(tmp_path, with_key=False)
    assert _carriers_without_the_key(tmp_path) == [
        "planted.yaml",
        "docker-compose.planted.yml",
    ]


def test_planted_violation_an_anchor_without_the_key_is_caught(
    tmp_path: Path,
) -> None:
    _plant(tmp_path, with_key=False)
    assert _anchors_without_the_key(tmp_path) == [
        "docker-compose.planted.yml::x-planted-runtime-env"
    ]


def test_planted_carrier_and_anchor_with_the_key_pass(tmp_path: Path) -> None:
    _plant(tmp_path, with_key=True)
    assert len(_carriers(tmp_path)) == 2
    assert len(_package_anchors(tmp_path)) == 1
    assert _carriers_without_the_key(tmp_path) == []
    assert _anchors_without_the_key(tmp_path) == []


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
