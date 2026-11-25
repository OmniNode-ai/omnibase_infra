# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The live chain tests run only where a live-chain runner exports their variables (OMN-21487).

Nightly run 45325746514 selected ``tests/integration/`` by marker on a hosted runner and failed
the three tests that read one deploy agent and the ledger of record, because that runner
exports none of the ``CHAIN_CANARY_*`` variables. They are deselected there and kept, with
their fail-closed assertions, wherever a runner exports any one of the variables.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.integration.chain_canary.conftest import (
    LIVE_CHAIN_FIXTURE,
    LIVE_RUNNER_VARIABLES,
    split_live_chain_items,
)

pytestmark = pytest.mark.integration

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_LIVE_TESTS = (
    "test_error_the_agents_failed_jobs_have_their_not_working_rows",
    "test_golden_the_newest_reading_row_matches_the_agents_job_record",
    "test_golden_the_agents_last_terminal_job_has_its_reading_row",
)
#: The nightly integration sweep's own marker expression
#: (``.github/workflows/nightly-integration.yml``).
_NIGHTLY_MARKERS = "integration and not bastion"


def _item(name: str, *fixtures: str) -> Any:
    return SimpleNamespace(name=name, fixturenames=fixtures)


def test_a_runner_without_a_live_variable_deselects_only_the_live_tests() -> None:
    live = _item("live", LIVE_CHAIN_FIXTURE, "tmp_path")
    hermetic = _item("hermetic", "tmp_path")

    kept, deselected = split_live_chain_items([live, hermetic], {})

    assert kept == [hermetic]
    assert deselected == [live]


@pytest.mark.parametrize("variable", LIVE_RUNNER_VARIABLES)
def test_a_runner_with_any_live_variable_keeps_every_live_test(variable: str) -> None:
    live = _item("live", LIVE_CHAIN_FIXTURE)
    hermetic = _item("hermetic")

    kept, deselected = split_live_chain_items([live, hermetic], {variable: "1"})

    assert kept == [live, hermetic]
    assert deselected == []


def _collected(extra_environment: dict[str, str]) -> str:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in LIVE_RUNNER_VARIABLES
    }
    environment.update(extra_environment)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/chain_canary",
            "--collect-only",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            _NIGHTLY_MARKERS,
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return done.stdout


def test_the_nightly_selection_collects_none_of_the_live_tests() -> None:
    collected = _collected({})

    for name in _LIVE_TESTS:
        assert name not in collected
    # A positive control: the hermetic tests of the same directory are still collected.
    assert "test_chain_canary_live_selection_omn21487.py" in collected


def test_a_live_chain_runner_collects_every_live_test() -> None:
    collected = _collected({"CHAIN_CANARY_AGENT_LEDGER_HOST": "h201"})

    for name in _LIVE_TESTS:
        assert name in collected
