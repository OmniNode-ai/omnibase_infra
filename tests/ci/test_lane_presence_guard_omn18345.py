# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Behaviour and policy tests for the lane-presence guard (OMN-18345).

Two halves:

1. The guard's own four verdicts, including the one that must NOT be a skip —
   a lane that is present and unwell has to reach the caller's assertions.
2. A policy scan asserting every container-inspecting test under
   ``tests/integration/`` routes through the guard, so the third such test
   added later does not reproduce the 8/8-night constant red.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.integration import lane_presence

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_ROOT = REPO_ROOT / "tests" / "integration"

#: A module that shells out to one of these is inspecting live lane containers.
_CONTAINER_SUBCOMMANDS: frozenset[str] = frozenset({"ps", "logs", "inspect"})

#: `tests/integration/docker/` is out of scope and this is not a convenience
#: exemption. Those modules `docker build` and run the containers they then
#: inspect (`pytestmark` + a `skip_if_no_docker` fixture of their own), so they
#: depend on a daemon, never on a pre-existing lane. Both nightly jobs also pass
#: `--ignore=tests/integration/docker`, so no module under it can contribute to
#: the failure this guard exists to remove.
_SELF_PROVISIONING_DIR: Path = INTEGRATION_ROOT / "docker"


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. The guard's verdicts
# ---------------------------------------------------------------------------


def _require_lane_without_skipping(containers: list[str]) -> dict[str, str]:
    """Call the guard and turn an UNEXPECTED skip into a FAILURE, not a skip.

    Without this, a guard neutered to skip unconditionally would make the
    must-not-skip tests below go green-by-skipping — pytest exits 0 on skips.
    That is the non-probative shape this whole ticket exists to remove, so the
    tests that assert "does not skip" must themselves go RED, never yellow.
    """
    try:
        return lane_presence.require_lane(containers)
    except pytest.skip.Exception as exc:  # pragma: no cover - only on regression
        pytest.fail(f"the guard skipped where it must not: {exc}")


def test_skips_when_docker_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane_presence, "docker_path", lambda: None)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        lane_presence.require_lane(["omninode-runtime"])

    assert "docker is not installed" in str(excinfo.value)


def test_skips_when_no_required_container_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane_presence, "docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(lane_presence, "container_statuses", dict)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        lane_presence.require_lane(["omninode-runtime", "omnibase-infra-postgres"])

    message = str(excinfo.value)
    assert "lane absent" in message
    # The skip must name what it looked for, or a green nightly is unreadable.
    assert "omninode-runtime" in message


def test_does_not_skip_when_the_container_is_present_but_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The falsifier for AC1: present-and-unwell is a FINDING, never a skip.

    A guard that also skipped here would turn every real lane defect into a
    green nightly, which is worse than the constant red it replaces.
    """
    monkeypatch.setattr(lane_presence, "docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        lane_presence,
        "container_statuses",
        lambda: {"omninode-runtime": "Up 3 minutes (unhealthy)"},
    )

    statuses = _require_lane_without_skipping(["omninode-runtime"])

    assert statuses["omninode-runtime"] == "Up 3 minutes (unhealthy)"


def test_does_not_skip_when_the_lane_is_up_with_one_container_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially-present lane is drift, so the caller must get to assert on it."""
    monkeypatch.setattr(lane_presence, "docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        lane_presence,
        "container_statuses",
        lambda: {"omnibase-infra-postgres": "Up 2 hours (healthy)"},
    )

    statuses = _require_lane_without_skipping(
        ["omninode-runtime", "omnibase-infra-postgres"]
    )

    assert "omninode-runtime" not in statuses


def test_an_empty_container_list_is_refused_not_skipped() -> None:
    """An empty list would skip unconditionally — the mute button, refused."""
    with pytest.raises(ValueError, match="at least one container name"):
        lane_presence.require_lane([])


def test_exited_containers_count_as_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docker ps -a`, not `docker ps`: an exited runtime is a finding."""
    monkeypatch.setattr(lane_presence, "docker_path", lambda: "/usr/bin/docker")
    monkeypatch.setattr(
        lane_presence,
        "container_statuses",
        lambda: {"omninode-runtime": "Exited (1) 4 minutes ago"},
    )

    statuses = _require_lane_without_skipping(["omninode-runtime"])

    assert statuses["omninode-runtime"].startswith("Exited")


def test_the_guard_reads_docker_ps_dash_a() -> None:
    """Pin the flag itself — dropping `-a` silently turns findings into skips."""
    assert "-a" in lane_presence._DOCKER_PS


# ---------------------------------------------------------------------------
# 2. The policy scan
# ---------------------------------------------------------------------------


def _modules_invoking_container_subcommands() -> dict[Path, set[str]]:
    """Return integration test modules that shell out to docker, by subcommand.

    Detected structurally from the AST (a list/tuple literal whose first element
    is the string ``docker``), never by grepping prose, so a docstring mentioning
    `docker ps` does not enrol a module.
    """
    found: dict[Path, set[str]] = {}
    for path in sorted(INTEGRATION_ROOT.rglob("test_*.py")):
        if _SELF_PROVISIONING_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        subcommands: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.List | ast.Tuple):
                continue
            elements = node.elts
            if len(elements) < 2:
                continue
            head, second = elements[0], elements[1]
            if not (isinstance(head, ast.Constant) and head.value == "docker"):
                continue
            if isinstance(second, ast.Constant) and second.value in (
                _CONTAINER_SUBCOMMANDS
            ):
                subcommands.add(str(second.value))
        if subcommands:
            found[path] = subcommands
    return found


def test_the_scan_finds_something_positive_control() -> None:
    """A zero here would make the policy test below vacuously green."""
    found = _modules_invoking_container_subcommands()
    assert found, (
        "no integration module was detected as inspecting containers — the "
        "detector is broken, not the tree clean"
    )


def test_every_container_inspecting_module_routes_through_the_guard() -> None:
    """AC3: one guard, and the next such test cannot re-create the constant red."""
    offenders: list[str] = []
    for path, subcommands in _modules_invoking_container_subcommands().items():
        source = path.read_text(encoding="utf-8")
        if "lane_presence" not in source:
            offenders.append(
                f"{path.relative_to(REPO_ROOT)} shells out to "
                f"docker {'/'.join(sorted(subcommands))} without importing the "
                "lane-presence guard"
            )

    assert not offenders, (
        "Container-inspecting integration tests must call "
        "tests.integration.lane_presence.require_lane() so they SKIP where the "
        "lane is absent and FAIL where it is present and unwell:\n  "
        + "\n  ".join(offenders)
    )
