# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19413: render a target ref's desired state from this repository's own git history.

The unit suite drives the renderer with hand-written artifacts. This drives the
real source adapter: ``desired_state.py render --ref`` resolves the commit,
reads ``deploy/lane-census/lane-manifest.yaml`` and ``uv.lock`` at that commit
through git, dispatches the contract-declared ``lab_desired_state.render``
operation of ``node_lab_proof_plan_compute`` over its private in-memory bus,
and validates the result against ``lab-desired-state.v1``. The host surface is
used because it needs no Compose binary; the compose-lane path is the same
dispatch with Compose output added.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts/lab_sync"))
from desired_state import (
    _capture_render_inputs,
    load_desired_state,
    main,
)

pytestmark = pytest.mark.integration

_HOST = "lab-201"
_LANE = "dev"


@pytest.fixture(autouse=True)
def _repo_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A hook or worktree caller can export GIT_DIR/GIT_WORK_TREE; the adapter
    # must read this checkout's history, not the caller's.
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        monkeypatch.delenv(key, raising=False)


def _head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _render_args(output: Path) -> list[str]:
    return [
        "render",
        "--source-root",
        str(_REPO_ROOT),
        "--ref",
        "HEAD",
        "--host",
        _HOST,
        "--lane",
        _LANE,
        "--surface-kind",
        "host",
        "--output",
        str(output),
    ]


def test_target_ref_renders_byte_identical_and_schema_valid(tmp_path: Path) -> None:
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    assert main(_render_args(first)) == 0
    assert main(_render_args(second)) == 0

    assert first.read_bytes() == second.read_bytes()
    state = load_desired_state(first)
    assert state.document["target_ref"]["commit"] == _head_commit()
    assert state.document["target_ref"]["repo"] == "OmniNode-ai/omnibase_infra"
    assert state.surface_id == f"{_HOST}/host"
    assert state.document["containers"] == []


def test_identical_rerender_keeps_the_output_mtime(tmp_path: Path) -> None:
    output = tmp_path / "desired.json"
    assert main(_render_args(output)) == 0
    stamp = output.stat().st_mtime_ns

    assert main(_render_args(output)) == 0

    assert output.stat().st_mtime_ns == stamp


def test_captured_inputs_replay_to_the_same_bytes(tmp_path: Path) -> None:
    rendered = tmp_path / "rendered.json"
    assert main(_render_args(rendered)) == 0

    captured = _capture_render_inputs(
        argparse.Namespace(
            source_root=_REPO_ROOT,
            ref="HEAD",
            kind="merge",
            composition=None,
            host=_HOST,
            lane=_LANE,
            surface_kind="host",
        )
    )
    bundle = tmp_path / "inputs.json"
    bundle.write_text(json.dumps(captured))
    replayed = tmp_path / "replayed.json"

    assert main(["render", "--input", str(bundle), "--output", str(replayed)]) == 0

    assert replayed.read_bytes() == rendered.read_bytes()


def test_undeclared_host_is_refused_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "refused.json"
    args = _render_args(output)
    args[args.index(_HOST)] = "lab-not-in-manifest"

    assert main(args) == 1

    assert not output.exists()
    assert "render REFUSED" in capsys.readouterr().err
