# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the committed probe placement variable drift check."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci import check_probe_placement_drift as probe_drift

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
REPO = "example_repo"
PLACEMENT = '["self-hosted","lab-runner"]'


def _write_policy(path: Path, declared: dict[str, str | int | None]) -> Path:
    document = {"probe_placement_variables": {REPO: declared}}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _write_vars(path: Path, values: object) -> Path:
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    return _write_policy(
        tmp_path / "policy.yaml",
        {"LAB_PROBE_RUNS_ON_JSON": PLACEMENT, "UNSET_RUNS_ON_JSON": None},
    )


def _argv(policy_file: Path, vars_file: Path | str, repo: str = REPO) -> list[str]:
    return [
        "--repo",
        repo,
        "--vars-json-file",
        str(vars_file),
        "--policy",
        str(policy_file),
    ]


def test_all_declared_variables_agree(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(
        tmp_path / "vars.json",
        {"LAB_PROBE_RUNS_ON_JSON": PLACEMENT, "UNRELATED": "ignored"},
    )

    assert probe_drift.main(_argv(policy_file, vars_file)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert (
        captured.out
        == "probe placement drift: OK; 2 declared variable(s) in example_repo "
        "match the live value\n"
    )


def test_json_label_list_order_and_whitespace_do_not_drift(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(
        tmp_path / "vars.json",
        {"LAB_PROBE_RUNS_ON_JSON": '[ "lab-runner", "self-hosted" ]'},
    )

    assert probe_drift.main(_argv(policy_file, vars_file)) == 0
    captured = capsys.readouterr()
    assert "probe placement drift: OK" in captured.out
    assert captured.err == ""


def test_live_difference_reports_name_and_values(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live_value = '["self-hosted","other-runner"]'
    vars_file = _write_vars(
        tmp_path / "vars.json", {"LAB_PROBE_RUNS_ON_JSON": live_value}
    )

    assert probe_drift.main(_argv(policy_file, vars_file)) == 1
    captured = capsys.readouterr()
    assert "vars.LAB_PROBE_RUNS_ON_JSON" in captured.err
    assert repr(PLACEMENT) in captured.err
    assert repr(live_value) in captured.err
    assert "probe placement drift: 1 of 2" in captured.out


def test_committed_null_and_live_set_drift(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(
        tmp_path / "vars.json",
        {
            "LAB_PROBE_RUNS_ON_JSON": PLACEMENT,
            "UNSET_RUNS_ON_JSON": '"ubuntu-latest"',
        },
    )

    assert probe_drift.main(_argv(policy_file, vars_file)) == 1
    captured = capsys.readouterr()
    assert "vars.UNSET_RUNS_ON_JSON: committed unset" in captured.err
    assert "live '\"ubuntu-latest\"'" in captured.err


def test_committed_set_and_live_missing_drift_while_both_missing_agree(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(tmp_path / "vars.json", {"UNRELATED": "present"})

    assert probe_drift.main(_argv(policy_file, vars_file)) == 1
    captured = capsys.readouterr()
    assert "vars.LAB_PROBE_RUNS_ON_JSON" in captured.err
    assert "live unset" in captured.err
    assert "vars.UNSET_RUNS_ON_JSON" not in captured.err
    assert "1 of 2 declared variable(s)" in captured.out


def test_repo_not_declared_is_usage_error(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(tmp_path / "vars.json", {"UNRELATED": "present"})

    assert probe_drift.main(_argv(policy_file, vars_file, "missing_repo")) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR: ")
    assert "missing_repo" in captured.err


def test_empty_vars_object_explains_context_was_not_passed(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(tmp_path / "vars.json", {})

    assert probe_drift.main(_argv(policy_file, vars_file)) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR: ")
    assert "vars context was not passed, never that nothing is set" in captured.err


def test_vars_json_must_be_an_object(
    policy_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vars_file = _write_vars(tmp_path / "vars.json", ["not", "an", "object"])

    assert probe_drift.main(_argv(policy_file, vars_file)) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR: ")
    assert "must be an object" in captured.err


def test_declared_value_must_be_string_or_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy_file = _write_policy(
        tmp_path / "policy.yaml", {"LAB_PROBE_RUNS_ON_JSON": 42}
    )
    vars_file = _write_vars(tmp_path / "vars.json", {"UNRELATED": "present"})

    assert probe_drift.main(_argv(policy_file, vars_file)) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("ERROR: ")
    assert "must be a non-empty string or null" in captured.err


def test_real_policy_passes_and_can_detect_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for repo in ("omnibase_infra", "omninode_infra", "omnimarket"):
        declared = probe_drift.load_declared(REAL_POLICY, repo)
        live = {name: value for name, value in declared.items() if value is not None}
        live["UNRELATED_VARIABLE"] = "ignored"
        vars_file = _write_vars(tmp_path / f"{repo}.json", live)

        assert probe_drift.main(_argv(REAL_POLICY, vars_file, repo)) == 0, repo
        capsys.readouterr()

        name, value = next(
            (name, value) for name, value in declared.items() if value is not None
        )
        live[name] = f"{value}-changed"
        _write_vars(vars_file, live)
        assert probe_drift.main(_argv(REAL_POLICY, vars_file, repo)) == 1, repo
        capsys.readouterr()


def test_stdin_vars_input(
    policy_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"LAB_PROBE_RUNS_ON_JSON": PLACEMENT})),
    )

    assert probe_drift.main(_argv(policy_file, "-")) == 0
    captured = capsys.readouterr()
    assert "probe placement drift: OK" in captured.out
    assert captured.err == ""
