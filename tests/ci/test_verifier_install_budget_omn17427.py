# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A test that installs the released verifier is budgeted for a cold install (OMN-17427).

``Tests (Split 3/15)`` went red on 2026-11-03 on branches that fully contain dev,
which was green on the same split at 17:27:53Z. The failing tests were the ones in
``test_ticket_contract_prepush_omn20870.py`` that call ``install_verifier``: each
died at ``Failed: Timeout (>60.0s) from pytest-timeout`` inside the install
subprocess (#5501 17:51:59Z, #5586 17:52:33Z, #5579 18:10:44Z). The install is a
shallow fetch of the omnimarket dev-merge tag and a ``uv sync`` of its lockfile on
an ephemeral runner that holds no cache, so it costs about what the suite-wide
``--timeout=60`` allows and finishes on one side of it or the other with the
network. The tag is not what moved: the install script refuses a tag whose commit
does not match its suffix, and the tag the caller passes is unchanged.

The fix is the one ``test_pr_title_check_omnimarket_node.py`` already applies to its
own install: the tests that install carry a literal ``timeout`` mark of at least twice a cold install, inside the lock's hold.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
_TEST_DIR = Path(__file__).parent

#: A cold install of the caller's tag, measured on 2026-11-03 with an empty uv cache
#: and an empty source directory: shallow fetch of the tag, then venv plus ``uv sync``.
_RECORDED_COLD_FETCH_S = 26.3
_RECORDED_COLD_SYNC_S = 35.0
_RECORDED_COLD_INSTALL_S = _RECORDED_COLD_FETCH_S + _RECORDED_COLD_SYNC_S

#: heavy_lock's ``--max-hold=15m`` in ``install_verifier``.
_INSTALL_LOCK_HOLD_S = 15 * 60

#: The files whose tests call ``install_verifier`` directly or through ``released_judge``.
_INSTALLING_FILES = (
    "test_ticket_contract_prepush_omn20870.py",
    "test_ticket_contract_prerequisite_parity_omn21642.py",
)
_INSTALLING_NAMES = {"released_judge", "install_verifier"}


def _ci_timeout_s() -> int:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    step = next(
        step
        for step in ci["jobs"]["test-parallel"]["steps"]
        if step.get("name") == "Run pytest (full suite)"
    )
    match = re.search(r"--timeout=(\d+)", step["run"])
    assert match is not None
    return int(match.group(1))


def _installing_tests() -> list[tuple[str, ast.FunctionDef]]:
    found: list[tuple[str, ast.FunctionDef]] = []
    for name in _INSTALLING_FILES:
        tree = ast.parse((_TEST_DIR / name).read_text(encoding="utf-8"))
        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            used = {a.arg for a in node.args.args} | {
                n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            }
            if used & _INSTALLING_NAMES:
                found.append((name, node))
    return found


def _timeout_mark_values(node: ast.FunctionDef) -> list[object]:
    values: list[object] = []
    for deco in node.decorator_list:
        if (
            isinstance(deco, ast.Call)
            and isinstance(deco.func, ast.Attribute)
            and deco.func.attr == "timeout"
        ):
            arg = deco.args[0] if deco.args else None
            values.append(arg.value if isinstance(arg, ast.Constant) else None)
    return values


def test_a_cold_install_does_not_fit_the_suite_timeout() -> None:
    assert _ci_timeout_s() < _RECORDED_COLD_INSTALL_S


def test_the_guard_finds_the_tests_that_install() -> None:
    names = {node.name for _file, node in _installing_tests()}

    assert {
        "test_the_prerequisite_runs_in_the_verifier_installed_from_the_callers_tag",
        "test_a_prose_check_value_is_refused_and_the_check_named",
        "test_a_command_shaped_contract_passes_the_released_guard",
        "test_hook_and_pr_create_give_one_verdict_per_fixture_parity",
        "test_the_incident_contract_with_no_dod_evidence_is_refused",
        "test_the_incident_contract_with_unbound_criteria_is_refused_unbound",
        "test_the_recorded_contract_that_pr_create_accepts_passes",
    } <= names


def test_every_test_that_installs_carries_the_install_budget() -> None:
    unbudgeted = []
    for file, node in _installing_tests():
        marks = _timeout_mark_values(node)
        ok = (
            len(marks) == 1
            and isinstance(marks[0], int)
            and 2 * _RECORDED_COLD_INSTALL_S < marks[0] < _INSTALL_LOCK_HOLD_S
        )
        if not ok:
            unbudgeted.append(f"{file}::{node.name}")

    assert unbudgeted == []
