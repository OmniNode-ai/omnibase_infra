# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for OMN-18684 (OMN-15547 rule R1).

THE INCIDENT. ``341ac00cfac9b0f7e42498de265ace7006fdfa72`` (OMN-18412) rewrote
the CI runner routing decision as a COMPUTE node and renamed
``load_route_policy`` to ``load_contract_policy``. Nothing connected that
rename to ``dev-lane-liveness.yml``, whose ``saturation-record`` job reaches the
module through an inline ``python3`` heredoc -- a call site ruff, mypy and every
IDE rename read as a YAML string. The job's FIRST step therefore failed on the
next scheduled run and on every run after it::

    ImportError: cannot import name 'load_route_policy' from 'runner_route_decision'
    ##[error]Process completed with exit code 1.

WHY THAT WAS WORSE THAN A RED STEP. ``saturation-record`` is the G7 saturation
monitor the operator consented to on 2026-09-07. The import is the first thing
the job does, before any fleet or lab reading is taken, so the monitor emitted
no record at all -- not a stale one. Three consecutive scheduled runs were red
this way (35337065964, 35336042566, 35335179281) before anyone read a log, and
the only downstream signal was ``assert_evidence_artifact.py`` correctly
reporting both saturation paths ABSENT.

A NAIVE RENAME WOULD NOT HAVE FIXED IT, which is why the repair added a reader
rather than swapping a name. ``load_contract_policy`` imports the typed policy
model, and with it ``omnibase_infra``, ``omnibase_core`` and pydantic; this job
runs a bare hosted ``python3`` with no install behind it, deliberately, because
the monitor has to survive the saturation it reports on. Pointing the heredoc at
the typed loader trades an ``ImportError`` for a ``ModuleNotFoundError``.

THE ARTIFACT is the workflow file verbatim at ``e49dea8f27d55eb6f4ceb1fc27c5d461a639b319``
-- the head sha of failing run 35337065964, created 2026-09-18T10:55:10Z --
fetched as a git object, not retyped. Its sha256 is pinned in
``tests/incident_replays/registry.yaml``.

THE MODULE IN THE REPLAY IS THE LIVE ONE, not a stub. The captured workflow is
resolved against this checkout's real ``scripts/ci`` tree, so the replay is
asking the real question the real runner asked: does this file's heredoc name
something this module exports?

THE ACCEPT CONTROL is the repaired file in the working tree. A checker that
rejected every workflow would replay this incident perfectly and fail the whole
repository; the control is what tells the two apart.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests/fixtures/omn18684" / "dev-lane-liveness.e49dea8f2.yml.captured"
)
FIXTURE_SHA256 = "ee67f1a51305a86c28afe439ab8bec0405690bca7715af9b03b7bd64b2e3890a"
LIVE = REPO_ROOT / ".github/workflows/dev-lane-liveness.yml"

MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "check_workflow_inline_python_imports.py"
_spec = importlib.util.spec_from_file_location(
    "check_workflow_inline_python_imports", MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = checker
_spec.loader.exec_module(checker)

MONITOR_JOB = "saturation-record"
PROBE_STEP = "Probe the org runner registry"


@pytest.fixture(scope="module")
def captured_bytes() -> bytes:
    """The shipped bytes, with their pinned digest asserted before use.

    A replay whose artifact can be edited is not a replay.
    """
    payload = FIXTURE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FIXTURE_SHA256
    return payload


def _root_with(workflow: bytes, tmp_path: Path) -> Path:
    """A root carrying ``workflow`` and this checkout's REAL ``scripts/ci``.

    The module side of the pair is a symlink to the live tree rather than a
    copy, so the replay cannot pass by asserting against a stub that agrees
    with it.
    """
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/workflows/dev-lane-liveness.yml").write_bytes(workflow)
    (tmp_path / "scripts").symlink_to(REPO_ROOT / "scripts", target_is_directory=True)
    return tmp_path


def test_the_real_shipped_workflow_is_rejected_by_the_real_checker(
    captured_bytes: bytes, tmp_path: Path
) -> None:
    """The load-bearing assertion: the exact file that ran on 35337065964,
    against the exact module it imported from, is refused -- and the refusal
    names the job, the step and the missing name, which is everything a reader
    needs to fix it without opening a CI log.
    """
    root = _root_with(captured_bytes, tmp_path / "shipped")
    violations = [
        v
        for path in checker.discover(root)
        for v in checker.check_file(path, root=root)
    ]
    assert len(violations) == 1, [v.render() for v in violations]
    rendered = violations[0].render()
    assert MONITOR_JOB in rendered
    assert PROBE_STEP in rendered
    assert "load_route_policy" in rendered
    assert "scripts/ci/runner_route_decision.py" in rendered
    assert checker.main(["--root", str(root)]) == 1


def test_the_captured_step_really_did_import_the_renamed_name(
    captured_bytes: bytes,
) -> None:
    """The artifact exhibits the defect rather than merely failing for some
    other reason: the shipped heredoc names ``load_route_policy``, the module
    exports ``load_contract_policy``, and the two are different strings.
    """
    text = captured_bytes.decode("utf-8")
    assert "from runner_route_decision import probe_fleet, load_route_policy" in text
    defined = checker.module_level_names(
        (REPO_ROOT / "scripts/ci/runner_route_decision.py").read_text(encoding="utf-8")
    )
    assert "load_route_policy" not in defined
    assert "load_contract_policy" in defined


def test_the_repaired_workflow_is_accepted(captured_bytes: bytes) -> None:
    """The accept control. Without it, a checker that refused every workflow
    would pass the rejection test above while being worthless -- and the
    byte-inequality assertion is what stops this control silently becoming a
    second copy of the same test.
    """
    assert LIVE.read_bytes() != captured_bytes
    violations = checker.check_file(LIVE, root=REPO_ROOT)
    assert violations == [], "\n".join(v.render() for v in violations)


def test_the_repair_is_reachable_on_a_bare_interpreter(tmp_path: Path) -> None:
    """The half a rename alone would have missed.

    Asserts the repaired import target is the reader that does NOT pull in the
    typed model -- so the fix holds on the job's bare ``python3``, rather than
    turning one import failure into another.
    """
    text = LIVE.read_text(encoding="utf-8")
    assert "load_contract_runner_group" in text
    assert "load_route_policy" not in text
    source = (REPO_ROOT / "scripts/ci/runner_route_decision.py").read_text(
        encoding="utf-8"
    )
    cheap = source.split("def load_contract_runner_group")[1].split("\ndef ")[0]
    assert "ModelCIRunnerRoutePolicy" not in cheap
    assert "import yaml" in cheap
