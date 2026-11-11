# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool driver takes changed_path_live from node_lab_proof_judge_compute and applies its answer (OMN-23715).

The rule lives in the node; the driver hands it the clone and probe phase text and applies what it answers. An unset
command, an error, an unreadable answer or a NO_CHECK answer leave the check absent, so the receipt's verifier names it
missing; a FAIL answer fails it. Nothing here turns a failure into a pass.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from tests.helpers.declared_changed_path import (
    NO_CHECK,
    NODE_LIVE,
    dead,
    declared,
    node_says,
)
from tests.helpers.prepr_pool_overlay import pool_overlay_env

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pool_overlay_env")]

_HERE = Path(__file__)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fakes = _load(
    "prepr_pool_fakes_omn23715", _HERE.with_name("test_prepr_runtime_pool_omn18893.py")
)
pool = fakes.pool
_real_settings = pool.settings
OUTPUTS = {**fakes.GOOD, "clone": "clone text", "probe": "probe text"}
RECEIPT_ID = "node_overlay_store_read_effect-6de58c0cd262-h202"
NODE = "node_overlay_store_read_effect"


def _command(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(pool, "settings", lambda: _With(_real_settings(), text))


class _With:
    """The driver's settings with the judge command replaced."""

    def __init__(self, settings: Any, command: str) -> None:
        self._settings = settings
        self.lab_proof_judge_command = command

    def __getattr__(self, name: str) -> Any:
        return getattr(self._settings, name)


def _done(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_an_unset_command_asks_nothing_and_leaves_the_check_absent() -> None:
    with mock.patch.object(pool.subprocess, "run") as run:
        assert pool.declared_changed_path(OUTPUTS) is None
        assert "changed_path_live" not in pool.judge(OUTPUTS).checks
    run.assert_not_called()


def test_the_command_is_handed_the_clone_and_probe_text_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _command(monkeypatch, "uv run --project /p python -m node_x")
    answer = json.dumps(NODE_LIVE)
    with mock.patch.object(pool.subprocess, "run", return_value=_done(answer)) as run:
        assert pool.declared_changed_path(OUTPUTS) == NODE_LIVE
    argv = run.call_args.args[0]
    assert argv == [
        "uv",
        "run",
        "--project",
        "/p",
        "python",
        "-m",
        "node_x",
        "pool-run",
    ]
    assert json.loads(run.call_args.kwargs["input"]) == {
        "clone": "clone text",
        "probe": "probe text",
    }


def test_seam_with_the_node_the_answer_keys_the_driver_reads_are_the_nodes() -> None:
    """The node's own test (test_probe_chain_lab_proof_judge.py) holds the other side of this pair."""
    for answer in (NODE_LIVE, dead("n"), declared(NODE, RECEIPT_ID), NO_CHECK):
        assert {"verdict", "ok", "evidence", "notes"} <= set(answer)
    assert {
        a["verdict"]
        for a in (NODE_LIVE, dead("n"), declared(NODE, RECEIPT_ID), NO_CHECK)
    } == {
        "PASS",
        "FAIL",
        "NOT_APPLICABLE",
        "NO_CHECK",
    }


@pytest.mark.parametrize(
    "done",
    [_done("", 3), _done("not json"), _done("[1]", 0)],
    ids=["command-error", "unreadable-answer", "not-an-object"],
)
def test_failing_control_an_unavailable_node_never_turns_a_failure_into_a_pass(
    monkeypatch: pytest.MonkeyPatch, done: subprocess.CompletedProcess[str]
) -> None:
    _command(monkeypatch, "node_x")
    with mock.patch.object(pool.subprocess, "run", return_value=done):
        judged = pool.judge(OUTPUTS)
    assert "changed_path_live" not in judged.checks


def test_a_pass_answer_passes_the_check() -> None:
    with node_says(pool, NODE_LIVE):
        judged = pool.judge(OUTPUTS)
    assert judged.checks["changed_path_live"] is True


def test_failing_control_a_fail_answer_fails_the_check_and_names_the_node() -> None:
    with node_says(pool, dead("node_dead")):
        judged = pool.judge(OUTPUTS)
    assert judged.checks["changed_path_live"] is False
    assert "changed path not live, no runtime log mentions: node_dead" in judged.notes


def test_a_no_check_answer_leaves_the_check_absent() -> None:
    with node_says(pool, NO_CHECK):
        judged = pool.judge(OUTPUTS)
    assert "changed_path_live" not in judged.checks


def test_a_not_applicable_answer_puts_its_class_and_substitute_proof_in_the_receipt() -> (
    None
):
    with node_says(pool, declared(NODE, RECEIPT_ID)):
        judged = pool.judge(OUTPUTS)
    assert judged.checks["changed_path_live"] is True
    rows = pool.receipt_checks(judged, OUTPUTS, (8085, 8086))
    (row,) = [r for r in rows if r.name == "changed_path_live"]
    assert row.ok is True
    assert "NOT-APPLICABLE class=runs_outside_runtime" in row.evidence
    assert f"substitute={RECEIPT_ID}" in row.evidence
