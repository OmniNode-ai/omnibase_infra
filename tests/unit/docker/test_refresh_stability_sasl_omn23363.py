# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The governed stability refresh re-asserts the SASL flip every run (OMN-23363).

``scripts/runtime_build/refresh_stability_lane.sh`` composes
``REFRESH_BUILD_SERVICES`` from ``CORE_SERVICES`` + the two SASL one-shots + the
projection writers, hands that one array to onex-runtime-deploy as the sole
``RUNTIME_BUILD_SERVICES_OVERRIDE``, and re-raises a nonzero deploy exit so a
failed deploy fails the refresh instead of reading as a green one. These
tests pin that wiring from the script text alone:

* both SASL one-shots ride the build scope, principal before flip;
* the rollback set is the build scope minus exactly the two image-only
  one-shots (they run a pinned upstream image, so no
  ``<project>-<service>:latest`` anchor exists for them);
* the deploy exit path is captured, guarded and re-raised;
* the scope reaches the deploy through exactly one override line.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
SCRAM_USER = "lane-redpanda-scram-user"
SASL_ENABLE = "lane-redpanda-sasl-enable"
ONE_SHOTS = (SCRAM_USER, SASL_ENABLE)
OVERRIDE_EXPORT_PREFIX = "export RUNTIME_BUILD_SERVICES_OVERRIDE="
OVERRIDE_EXPORT = f'{OVERRIDE_EXPORT_PREFIX}"${{REFRESH_BUILD_SERVICES[*]}}"'
CAPTURE_NEEDLE = "|| DEPLOY_EXIT=$?"
DEPLOY_GUARD = 'if [[ "${DEPLOY_EXIT}" -ne 0 ]]; then'
EXIT_LINE = 'exit "${DEPLOY_EXIT}"'


def _script_lines() -> list[str]:
    return SCRIPT_PATH.read_text(encoding="utf-8").splitlines()


def _find_prefix(lines: list[str], prefix: str) -> int:
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        pytest.fail(f"expected exactly one line starting {prefix!r}, got {matches}")
    return matches[0]


def _scope_block(lines: list[str]) -> str:
    """From the ``readonly CORE_SERVICES=`` line through the ROLLBACK_SERVICES line."""
    start = _find_prefix(lines, "readonly CORE_SERVICES=")
    rollback = _find_prefix(lines, "readonly ROLLBACK_SERVICES=(")
    return "\n".join(lines[start : rollback + 1])


def _print_array(block: str, array: str) -> list[str]:
    """Evaluate the scope block under bash and print one array, one per line."""
    command = f'{block}; printf "%s\\n" "${{{array}[@]}}"'
    proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"bash failed printing {array}: {proc.stderr.strip()}")
    return (proc.stdout or "").splitlines()


def _core_services(block: str) -> list[str]:
    match = re.search(r"readonly CORE_SERVICES=\(([^)]*)\)", block)
    if match is None:
        pytest.fail("no readonly CORE_SERVICES=(...) in the extracted scope block")
    return match.group(1).split()


def _fail_unless_member(array: str, service: str, members: list[str]) -> None:
    if service not in members:
        pytest.fail(f"{service} missing from {array}: {members}")


def _fail_if_out_of_order(
    array: str,
    first: str,
    second: str,
    members: list[str],
) -> None:
    if members.index(first) >= members.index(second):
        pytest.fail(
            f"{first} (index {members.index(first)}) must precede "
            f"{second} (index {members.index(second)}) in {array}: {members}"
        )


def _line_index(lines: list[str], needle: str, start: int = 0) -> int:
    for idx in range(start, len(lines)):
        if needle in lines[idx]:
            return idx
    return -1


def _stripped_line_index(lines: list[str], exact: str, start: int) -> int:
    for idx in range(start, len(lines)):
        if lines[idx].strip() == exact:
            return idx
    return -1


def _fail_unless_anchor_chain(anchors: dict[str, int]) -> None:
    """Every anchor must exist and appear in strictly increasing line order."""
    missing = [name for name, idx in anchors.items() if idx < 0]
    indexes = list(anchors.values())
    if missing:
        pytest.fail(f"deploy-exit anchors missing from {SCRIPT_PATH.name}: {missing}")
    if any(indexes[i] >= indexes[i + 1] for i in range(len(indexes) - 1)):
        pytest.fail(
            f"deploy-exit anchors not strictly increasing in {SCRIPT_PATH.name}: "
            f"{anchors}"
        )


@pytest.mark.unit
def test_refresh_stability_scope_runs_both_sasl_one_shots_in_order() -> None:
    block = _scope_block(_script_lines())
    build = _print_array(block, "REFRESH_BUILD_SERVICES")
    rollback = _print_array(block, "ROLLBACK_SERVICES")
    for one_shot in ONE_SHOTS:
        _fail_unless_member("REFRESH_BUILD_SERVICES", one_shot, build)
    _fail_if_out_of_order("REFRESH_BUILD_SERVICES", SCRAM_USER, SASL_ENABLE, build)
    for service in _core_services(block):
        _fail_unless_member("REFRESH_BUILD_SERVICES", service, build)
        _fail_if_out_of_order("REFRESH_BUILD_SERVICES", service, SCRAM_USER, build)
    if not rollback:
        pytest.fail("the extracted scope block left ROLLBACK_SERVICES empty")


@pytest.mark.unit
def test_refresh_stability_rollback_set_excludes_the_image_only_one_shots() -> None:
    block = _scope_block(_script_lines())
    build = _print_array(block, "REFRESH_BUILD_SERVICES")
    rollback = _print_array(block, "ROLLBACK_SERVICES")
    for one_shot in ONE_SHOTS:
        if one_shot in rollback:
            pytest.fail(f"{one_shot} must not be in ROLLBACK_SERVICES: {rollback}")
    missing = sorted(set(build) - set(rollback) - set(ONE_SHOTS))
    if missing:
        pytest.fail(f"build-scope services missing from ROLLBACK_SERVICES: {missing}")
    extras = sorted(set(rollback) - set(build))
    if extras:
        pytest.fail(f"ROLLBACK_SERVICES members outside the build scope: {extras}")


@pytest.mark.unit
def test_refresh_stability_deploy_exit_fails_the_refresh() -> None:
    lines = _script_lines()
    override_idx = _line_index(lines, OVERRIDE_EXPORT)
    capture_idx = _line_index(lines, CAPTURE_NEEDLE, override_idx + 1)
    guard_idx = _line_index(lines, DEPLOY_GUARD, capture_idx + 1)
    exit_idx = _stripped_line_index(lines, EXIT_LINE, guard_idx + 1)
    _fail_unless_anchor_chain(
        {
            "override export": override_idx,
            "exit capture": capture_idx,
            "nonzero guard": guard_idx,
            "re-raise exit": exit_idx,
        }
    )


@pytest.mark.unit
def test_refresh_stability_scope_is_handed_to_the_deploy_as_one_override() -> None:
    lines = _script_lines()
    matches = [i for i, line in enumerate(lines) if OVERRIDE_EXPORT_PREFIX in line]
    if len(matches) != 1:
        pytest.fail(
            f"expected exactly one {OVERRIDE_EXPORT_PREFIX!r} line in "
            f"{SCRIPT_PATH.name}, found {len(matches)} "
            f"(1-based lines: {[i + 1 for i in matches]})"
        )
    if lines[matches[0]].strip() != OVERRIDE_EXPORT:
        pytest.fail(f"unexpected override line: {lines[matches[0]]!r}")
