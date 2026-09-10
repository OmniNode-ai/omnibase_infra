# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate for the OMN-18117 product-clone comparison pin.

Why this file exists
--------------------
The evidence-autoclose sweep materialises every product clone once, near the
start of the job, then spends ~20 minutes running ``dod_verify`` behaviour
checks against those trees. ``EvidenceCollector`` decided freshness at CHECK
time by fetching ``origin/<branch>`` again and comparing the clone's HEAD
against that fresh fetch's tip, so any ordinary merge landing on a product
repo's ``dev`` inside the window made an otherwise-correct clone read
``behind N``. Every behaviour check pinned there was then refused UNEXECUTED as
``PRODUCT_CLONE_STALE`` and the candidate was held with nothing judged — run
34428131180 (2026-09-10T02:06:28Z) held eight tickets exactly that way.

The fix has two halves in two repositories. omnimarket's collector learned to
measure HEAD against a recorded commit instead of a live fetch; this workflow
is what records it. If this half rots the collector simply finds no pin file,
falls silently back to the live comparison, and the defect returns with no
signal anywhere — which is the "detection that is never enforced" shape
CLAUDE.md Rule 5 forbids. These tests are that enforcement.

The renderer is pinned by EXECUTION, not by string match: the workflow
delimits it between two markers and this module extracts that exact program
out of the shipped YAML and runs it. A file the collector's models reject
(both declare ``extra="forbid"``) unpins every repository at once, so
"it looks right in the diff" is not enough.

Ticket: OMN-18117
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

_BEGIN_MARKER = "# ---8<--- OMN-18117 BEGIN product-clone-pin-render"
_END_MARKER = "# ---8<--- OMN-18117 END product-clone-pin-render"

_MATERIALISE_STEP = "Derive and materialise the cwd repo set the OCC contracts name"
# Every step that spawns a dod_verify behaviour check. Both must carry the pin
# file, or the diagnostic lies about what the sweep does.
_CHECK_BEARING_STEPS = (
    "Diagnose verdict divergence",
    "Run evidence autoclose sweep",
)

# The env var EvidenceCollector reads. Spelled once, here, so a rename that
# lands in only one of the two repositories fails this gate rather than
# silently unpinning the fleet.
_PIN_ENV = "DOD_VERIFY_PRODUCT_CLONE_PIN_FILE"

# The schema version omnimarket's ModelProductClonePinSet accepts. A renderer
# that emits any other version is ignored by the collector with a warning.
_SCHEMA_VERSION = 1

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def steps() -> list[dict[str, Any]]:
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    declared = loaded["jobs"]["evidence-autoclose-sweep"]["steps"]
    assert isinstance(declared, list)
    return declared


def _step_by_name(steps: list[dict[str, Any]], fragment: str) -> dict[str, Any]:
    matches = [s for s in steps if fragment in str(s.get("name", ""))]
    assert matches, (
        f"no step in evidence-autoclose-sweep.yml has a name containing "
        f"{fragment!r}; declared steps: {[s.get('name') for s in steps]}"
    )
    assert len(matches) == 1, f"{fragment!r} matched {len(matches)} steps"
    return matches[0]


@pytest.fixture(scope="module")
def render_program(steps: list[dict[str, Any]]) -> str:
    """The pin renderer the shipped workflow actually runs, verbatim."""
    run = str(_step_by_name(steps, _MATERIALISE_STEP)["run"])
    assert _BEGIN_MARKER in run and _END_MARKER in run, (
        "the materialise step must delimit its pin renderer with "
        f"{_BEGIN_MARKER!r} / {_END_MARKER!r} so this gate can execute the "
        "exact program the runner executes rather than paraphrasing it."
    )
    body = run.split(_BEGIN_MARKER, 1)[1].split(_END_MARKER, 1)[0]
    assert not body.lstrip("\n").startswith(" "), (
        "the renderer must sit at column 0 inside the run script (a nested "
        "heredoc would not terminate); found leading indentation."
    )
    return body


def _render(program: str, rows: str, tmp_path: Path) -> tuple[dict[str, Any], str]:
    prog = tmp_path / "render_pins.py"
    prog.write_text(program, encoding="utf-8")
    src = tmp_path / "pins.tsv"
    src.write_text(rows, encoding="utf-8")
    dest = tmp_path / "pins.json"
    proc = subprocess.run(
        [sys.executable, str(prog), str(src), str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"the workflow's pin renderer exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload, proc.stderr


def _row(repo_root: str, sha: str, ref: str = "origin/dev") -> str:
    return f"{repo_root}\t{sha}\t{ref}\n"


# ---------------------------------------------------------------------------
# The wiring: recorded here, read there
# ---------------------------------------------------------------------------


def test_the_materialise_step_records_a_pin_for_every_normalised_tree(
    steps: list[dict[str, Any]],
) -> None:
    """Both materialisation branches record, not just the per-dest one.

    The gate checkout takes an early ``continue``; a pin recorded only after
    the loop body would leave ``github.workspace`` — where every
    omnibase_infra-homed behaviour check runs — measured against the moving
    tip while every other tree is pinned.
    """
    run = str(_step_by_name(steps, _MATERIALISE_STEP)["run"])
    assert "record_clone_pin()" in run, (
        "the materialise step must define record_clone_pin; without it no "
        "comparison target is frozen and OMN-18117 returns silently."
    )
    calls = run.count('record_clone_pin "${target}"')
    assert calls == 2, (
        f"expected record_clone_pin to be called on both materialisation "
        f"branches (the gate checkout and each derived dest); found {calls}."
    )


def test_the_pin_records_the_upstream_tip_not_the_clone_head(
    steps: list[dict[str, Any]],
) -> None:
    """Pinning HEAD would be vacuous and would launder an already-behind tree.

    HEAD always equals itself, so a HEAD-derived pin makes every clone read
    FRESH — including one that was already missing the merge under
    adjudication when this loop picked it up, which is the OMN-16846 AC5
    refusal this must not weaken.
    """
    run = str(_step_by_name(steps, _MATERIALISE_STEP)["run"])
    body = run.split("record_clone_pin() {", 1)[1].split("\n          }", 1)[0]
    assert 'rev-parse "refs/remotes/origin/${branch}"' in body, (
        "record_clone_pin must read the remote-tracking tip as the pinned "
        "comparison target."
    )
    assert 'tip="$(git -C "${tree}" rev-parse HEAD' not in body, (
        "the pinned comparison target must not be the clone's own HEAD."
    )


@pytest.mark.parametrize("step_name", _CHECK_BEARING_STEPS)
def test_every_check_bearing_step_exports_the_pin_file(
    steps: list[dict[str, Any]], step_name: str
) -> None:
    env = _step_by_name(steps, step_name).get("env") or {}
    assert _PIN_ENV in env, (
        f"{step_name!r} spawns dod_verify behaviour checks but does not export "
        f"{_PIN_ENV}, so every clone it touches is measured against the moving "
        f"upstream tip. Declared env keys: {sorted(env)}"
    )


def test_both_check_bearing_steps_name_the_same_pin_file(
    steps: list[dict[str, Any]],
) -> None:
    """A diagnostic reading a different file would lie about the real sweep."""
    values = {
        str((_step_by_name(steps, name).get("env") or {})[_PIN_ENV])
        for name in _CHECK_BEARING_STEPS
    }
    assert len(values) == 1, f"the check-bearing steps disagree on {_PIN_ENV}: {values}"


def test_the_exported_pin_path_is_the_one_the_materialise_step_writes(
    steps: list[dict[str, Any]],
) -> None:
    """The two halves are joined by a literal path; assert they still match."""
    exported = str(
        (_step_by_name(steps, _CHECK_BEARING_STEPS[1]).get("env") or {})[_PIN_ENV]
    )
    basename = exported.rsplit("/", 1)[-1]
    run = str(_step_by_name(steps, _MATERIALISE_STEP)["run"])
    assert f'"${{RUNNER_TEMP}}/{basename}"' in run, (
        f"the sweep step reads {exported}, but the materialise step writes no "
        f"file named {basename}."
    )


# ---------------------------------------------------------------------------
# The renderer, executed
# ---------------------------------------------------------------------------


def test_the_rendered_file_carries_the_schema_version_the_collector_accepts(
    render_program: str, tmp_path: Path
) -> None:
    payload, _stderr = _render(render_program, "", tmp_path)
    assert payload["version"] == _SCHEMA_VERSION


def test_a_run_that_pinned_nothing_still_writes_a_file(
    render_program: str, tmp_path: Path
) -> None:
    """An absent file is indistinguishable from a step that never ran.

    The collector warns on an unreadable path once per candidate; a present
    file declaring an empty pin set says what actually happened.
    """
    payload, _stderr = _render(render_program, "", tmp_path)
    assert payload["pins"] == []


def test_a_well_formed_row_renders_the_fields_the_models_declare(
    render_program: str, tmp_path: Path
) -> None:
    """Both omnimarket models forbid extras, so an extra key unpins everything."""
    sha = "a" * 40
    payload, _stderr = _render(
        render_program,
        _row("/home/runner/work/omnibase_infra/omnimarket", sha),
        tmp_path,
    )

    assert len(payload["pins"]) == 1
    pin = payload["pins"][0]
    assert set(pin) == {"repo_root", "pinned_sha", "upstream_ref", "recorded_at"}
    assert pin["repo_root"] == "/home/runner/work/omnibase_infra/omnimarket"
    assert pin["pinned_sha"] == sha
    assert pin["upstream_ref"] == "origin/dev"
    assert _SHA_RE.match(pin["pinned_sha"])
    assert set(payload) == {"version", "pins"}


def test_a_short_sha_is_dropped_and_reported_not_emitted(
    render_program: str, tmp_path: Path
) -> None:
    """One bad row must not take the whole file — and the file down unpins all.

    ``pinned_sha`` carries a 40-hex pattern on the model, so an abbreviated
    SHA fails validation for the ENTIRE pin set, not just its own entry.
    """
    good = "b" * 40
    payload, stderr = _render(
        render_program,
        _row("/repo/short", "abc1234") + _row("/repo/good", good),
        tmp_path,
    )

    assert [p["repo_root"] for p in payload["pins"]] == ["/repo/good"]
    assert "PIN_SHA_NOT_FULL_HEX" in stderr


def test_a_relative_repo_root_is_dropped_and_reported(
    render_program: str, tmp_path: Path
) -> None:
    """The collector keys pins by realpath'd absolute root; relative never matches."""
    payload, stderr = _render(render_program, _row("omnimarket", "c" * 40), tmp_path)

    assert payload["pins"] == []
    assert "PIN_ROOT_NOT_ABSOLUTE" in stderr


def test_one_tree_gets_one_comparison_target(
    render_program: str, tmp_path: Path
) -> None:
    """Two rows for one root are two answers to one question. First wins, loudly."""
    first, second = "d" * 40, "e" * 40
    payload, stderr = _render(
        render_program, _row("/repo/a", first) + _row("/repo/a", second), tmp_path
    )

    assert [p["pinned_sha"] for p in payload["pins"]] == [first]
    assert "PIN_ROOT_DUPLICATE" in stderr


def test_a_malformed_row_is_reported_rather_than_dropped_silently(
    render_program: str, tmp_path: Path
) -> None:
    payload, stderr = _render(render_program, "not\ttwo-fields\n", tmp_path)

    assert payload["pins"] == []
    assert "PIN_ROW_MALFORMED" in stderr


def test_an_unreadable_source_still_produces_a_valid_empty_pin_file(
    render_program: str, tmp_path: Path
) -> None:
    """A failure to read the recorded rows must not abort the materialise step.

    Aborting would fail the whole sweep on a bookkeeping file; emitting a
    valid empty set lands every repository back on the live comparison, which
    is the pre-existing behaviour.
    """
    prog = tmp_path / "render_pins.py"
    prog.write_text(render_program, encoding="utf-8")
    dest = tmp_path / "pins.json"
    proc = subprocess.run(
        [sys.executable, str(prog), str(tmp_path / "absent.tsv"), str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "PIN_SOURCE_UNREADABLE" in proc.stderr
    assert json.loads(dest.read_text(encoding="utf-8")) == {
        "version": _SCHEMA_VERSION,
        "pins": [],
    }
