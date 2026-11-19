# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent writes the dev runtime's product clone pin file (OMN-25901).

The dev runtime mounts each product clone read-only and its ``dod_verify`` measures
a clone's freshness against a pin file (``DOD_VERIFY_PRODUCT_CLONE_PIN_FILE``,
``ModelProductClonePinSet``: ``version`` 1 and ``pins`` of ``repo_root``,
``pinned_sha``, ``upstream_ref``, ``recorded_at``, nothing else). The verifier counts
``git rev-list --count HEAD..<pinned_sha>``: zero is fresh, anything else is
``PRODUCT_CLONE_NOT_FRESH``. ``_freshness`` below is that one measure.

The writer is ``runtime_overlay_revision.write_product_clone_pins``, which the agent
already runs when a dev runtime job starts (through ``apply``), reading the node
contract's ``product_clone_pin`` block.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.executor import load_dev_boot_guard

from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = [pytest.mark.unit, pytest.mark.product_clone_pin]

REPO = Path(__file__).resolve().parents[4]
MODULE = "nodes.node_setup_local_provision_effect.runtime_overlay_revision"
PIN_KEYS = {"repo_root", "pinned_sha", "upstream_ref", "recorded_at"}


def _module() -> ModuleType:
    return load_dev_boot_guard(MODULE)


def _declared() -> dict[str, object]:
    return dict(_module().product_clone_pin_declaration(executor_mod))


def _names() -> list[str]:
    names = _declared()["repositories"]
    assert isinstance(names, list)
    return [str(n) for n in names]


def _git(clone: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def _make_registry(root: Path) -> dict[str, Path]:
    clones: dict[str, Path] = {}
    for name in _names():
        clone = root / name
        clone.mkdir(parents=True)
        _git(clone, "init", "-q")
        _git(clone, "commit", "-q", "--allow-empty", "-m", "first")
        clones[name] = clone
    return clones


def _freshness(clone: Path, pinned_sha: str) -> str:
    """The verifier's pinned measure: HEAD must contain the pin."""
    behind = _git(clone, "rev-list", "--count", f"HEAD..{pinned_sha}")
    return "FRESH" if behind == "0" else "STALE"


def _execution(root: Path) -> SimpleNamespace:
    """The agent module as the writer sees it: its location and its compose clone."""
    return SimpleNamespace(
        __file__=executor_mod.__file__, REPO_DIR=str(root / "omnibase_infra")
    )


def _written(root: Path) -> dict[str, object]:
    path = _module().write_product_clone_pins(_execution(root))
    assert path == root / ".onex_state" / "product-clone-pin" / "pins.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _pins(payload: dict[str, object]) -> list[dict[str, str]]:
    pins = payload["pins"]
    assert isinstance(pins, list)
    return pins


def test_pin_file_has_the_verifier_shape_and_one_pin_per_clone_read_from_that_clone(
    tmp_path: Path,
) -> None:
    clones = _make_registry(tmp_path)
    payload = _written(tmp_path)
    assert set(payload) == {"version", "pins"}
    assert payload["version"] == 1
    pins = _pins(payload)
    assert len(pins) == len(clones)
    for pin in pins:
        assert set(pin) == PIN_KEYS
        assert re.fullmatch(r"[0-9a-f]{40}", pin["pinned_sha"])
        name = pin["repo_root"].removeprefix("/app/")
        assert pin["repo_root"] == f"/app/{name}"
        assert pin["pinned_sha"] == _git(clones[name], "rev-parse", "HEAD")
    assert sorted(p["repo_root"] for p in pins) == sorted(f"/app/{n}" for n in _names())
    path = tmp_path / ".onex_state" / "product-clone-pin" / "pins.json"
    assert path.stat().st_mode & 0o777 == 0o644


def test_a_clone_fast_forwarded_after_the_pin_reads_fresh_and_a_reset_one_does_not(
    tmp_path: Path,
) -> None:
    clones = _make_registry(tmp_path)
    pins = {p["repo_root"]: p["pinned_sha"] for p in _pins(_written(tmp_path))}
    forward, reset, same = (
        clones["omnimarket"],
        clones["omnibase_core"],
        clones["omniclaude"],
    )
    assert _freshness(same, pins["/app/omniclaude"]) == "FRESH"
    # The canonical-clone sync fast-forwards a clone after the job wrote the pin.
    _git(forward, "commit", "-q", "--allow-empty", "-m", "synced")
    assert _freshness(forward, pins["/app/omnimarket"]) == "FRESH"
    # A clone replaced by an unrelated history no longer contains its pin.
    _git(reset, "checkout", "-q", "--orphan", "unrelated")
    _git(reset, "commit", "-q", "--allow-empty", "-m", "unrelated root")
    assert _freshness(reset, pins["/app/omnibase_core"]) == "STALE"


def test_a_clone_whose_revision_cannot_be_read_gets_no_pin(tmp_path: Path) -> None:
    _make_registry(tmp_path)
    (tmp_path / "omnidash" / ".git" / "HEAD").write_text(
        "not a revision\n", encoding="utf-8"
    )
    shutil.rmtree(tmp_path / "omnimemory")
    roots = {p["repo_root"] for p in _pins(_written(tmp_path))}
    assert "/app/omnidash" not in roots
    assert "/app/omnimemory" not in roots
    assert len(roots) == len(_names()) - 2


def test_rewriting_replaces_the_file_whole(tmp_path: Path) -> None:
    clones = _make_registry(tmp_path)
    first = _written(tmp_path)
    _git(clones["omnimarket"], "commit", "-q", "--allow-empty", "-m", "synced")
    assert first != _written(tmp_path)
    directory = tmp_path / ".onex_state" / "product-clone-pin"
    assert [p.name for p in directory.iterdir()] == ["pins.json"]


def test_apply_writes_the_pin_file_for_a_dev_job_even_with_no_overlay_revision(
    tmp_path: Path,
) -> None:
    clones = _make_registry(tmp_path)
    state_dir = tmp_path / "state"
    _module().apply(_execution(tmp_path), state_dir, {}, "")
    path = tmp_path / ".onex_state" / "product-clone-pin" / "pins.json"
    pins = _pins(json.loads(path.read_text(encoding="utf-8")))
    assert {p["pinned_sha"] for p in pins} == {
        _git(c, "rev-parse", "HEAD") for c in clones.values()
    }


@pytest.mark.parametrize(
    "filename",
    ["docker-compose.dev-lane.yml", "docker-compose.dev-tenant-fold.yml"],
)
def test_the_pinned_repositories_are_the_clones_the_compose_file_mounts(
    filename: str,
) -> None:
    compose = load_compose(REPO / "docker" / filename)
    mounted: set[str] = set()
    for service in compose["services"].values():
        for volume in (service or {}).get("volumes", []):
            match = re.fullmatch(r"\.\./\.\./(\w+):/app/\1:ro", str(volume))
            if match:
                mounted.add(match.group(1))
    assert mounted == set(_names())
