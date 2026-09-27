# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18930: cohort keys assembled from real .201 dev-lane captures compare correctly.

Three delegations with the same prompt ran through the deployed ``dev`` lane on
2026-09-24 (``onex delegate --bus kafka --lane dev --locus deployed-lane``).
Each fixture pair holds the terminal the delegate receipt recorded (only the
fields the assembler reads plus the verbatim terminal envelope) and the runtime
readback taken on the lane host within seconds of that terminal:

* capture A  -- build ``0ddd10ca``;
* capture A2 -- the same build, a same-cohort control;
* capture B  -- build ``a7ea811c``, after the merge-driven lane rebuild.

Assembled end to end through the two entry points, A and A2 must read
``SAME_COHORT`` and A and B must read ``CROSS_COHORT: build_identity`` and
nothing else. That is the ticket's lab-first falsifier run on real terminal
shapes, not synthetic ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.models.delegation import ModelDelegationCohortKey

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSEMBLE = _REPO_ROOT / "scripts" / "assemble_delegation_cohort_key.py"
_VALIDATE = _REPO_ROOT / "scripts" / "validate_delegation_cohort_keys.py"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "delegation" / "omn18930_lab"


def _assemble(capture: str, out_dir: Path) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            str(_ASSEMBLE),
            "--receipt",
            str(_FIXTURES / f"capture_{capture}_terminal_receipt.json"),
            "--runtime-readback",
            str(_FIXTURES / f"capture_{capture}_runtime_readback.json"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    path = out_dir / f"key_{capture}.json"
    path.write_text(result.stdout, encoding="utf-8")
    return path


def _compare(a: Path, b: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_VALIDATE), str(a), str(b)],
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.fixture(scope="module")
def keys(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out_dir = tmp_path_factory.mktemp("omn18930_keys")
    return {capture: _assemble(capture, out_dir) for capture in ("A", "A2", "B")}


def test_same_build_captures_are_one_cohort(keys: dict[str, Path]) -> None:
    result = _compare(keys["A"], keys["A2"])

    key = ModelDelegationCohortKey.model_validate_json(
        keys["A"].read_text(encoding="utf-8")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"SAME_COHORT: {key.key_sha256}\n"


def test_a_lane_rebuild_between_captures_is_cross_cohort_on_build_only(
    keys: dict[str, Path],
) -> None:
    result = _compare(keys["A"], keys["B"])

    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == "CROSS_COHORT: build_identity\n"


def test_each_capture_key_carries_the_consumer_that_handled_it(
    keys: dict[str, Path],
) -> None:
    for capture, path in keys.items():
        key = ModelDelegationCohortKey.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        receipt = json.loads(
            (_FIXTURES / f"capture_{capture}_terminal_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        terminal = receipt["receipt"]["result"]["terminal_payload"]
        assert terminal["payload"]["status"] == "completed"
        assert key.consumer_identity.node_name == "node_delegate_skill_orchestrator"
        assert key.lane == receipt["lane"] == "dev"
        assert key.deadline_seconds == float(
            terminal["payload"]["budget_evidence"]["execution_timeout_seconds"]
        )
