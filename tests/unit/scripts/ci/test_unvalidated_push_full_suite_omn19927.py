# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A ``dev`` push the merge queue did not validate gets the full suite (OMN-19927).

omnibase_infra#4111 reached ``dev`` as e263eee0 through a direct REST merge,
after its merge group had failed the full suite six times. The push run's
selector diffed ``HEAD~1``, mapped ``config/deploy_lane_routing.yaml`` to no
test that asserts on it, and ran two shards of smart selection: green. The
three ``tests/ci/`` files that assert on that table never ran, and every later
merge group failed on the same six cases.

These tests pin both halves: without the provenance input that diff still
narrows (the defect, kept visible), and with ``--force-full-suite
unvalidated_push`` it is the full suite under its own reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.detect_test_paths import compute_selection, main
from scripts.ci.test_selection_models import EnumFullSuiteReason

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
ADJ = REPO_ROOT / "scripts/ci/test_selection_adjacency.yaml"

# `git diff --name-only e263eee0~1 e263eee0`, read 2026-09-28.
CHANGED_AT_E263EEE = [
    "config/deploy_lane_routing.yaml",
    "scripts/deploy-agent/tests/unit/test_deploy_agent_instances_as_data_omn19543.py",
    "scripts/deploy-agent/tests/unit/test_dev_202_omnimarket_route_omn19510.py",
    "scripts/deploy-agent/tests/unit/test_idle_converge_omn19509.py",
]


def test_the_e263eee_diff_narrows_without_provenance() -> None:
    """The defect, kept visible: this is what the push run selected."""
    selection = compute_selection(
        changed_files=CHANGED_AT_E263EEE,
        adjacency_path=ADJ,
        ref_name="dev",
        event_name="push",
    )
    assert selection.is_full_suite is False
    assert "tests/" not in selection.selected_paths


def test_an_unvalidated_push_selects_the_full_suite_under_its_own_reason() -> None:
    selection = compute_selection(
        changed_files=CHANGED_AT_E263EEE,
        adjacency_path=ADJ,
        ref_name="dev",
        event_name="push",
        force_full_suite_reason=EnumFullSuiteReason.UNVALIDATED_PUSH,
    )
    assert selection.is_full_suite is True
    assert selection.full_suite_reason is EnumFullSuiteReason.UNVALIDATED_PUSH
    assert selection.selected_paths == ["tests/"]
    assert selection.split_count == 15


def test_the_forced_reason_outranks_the_docs_only_exemption() -> None:
    selection = compute_selection(
        changed_files=["docs/README.md"],
        adjacency_path=ADJ,
        ref_name="dev",
        event_name="push",
        force_full_suite_reason=EnumFullSuiteReason.UNVALIDATED_PUSH,
    )
    assert selection.full_suite_reason is EnumFullSuiteReason.UNVALIDATED_PUSH


def test_main_branch_keeps_its_own_reason() -> None:
    selection = compute_selection(
        changed_files=CHANGED_AT_E263EEE,
        adjacency_path=ADJ,
        ref_name="main",
        event_name="push",
        force_full_suite_reason=EnumFullSuiteReason.UNVALIDATED_PUSH,
    )
    assert selection.full_suite_reason is EnumFullSuiteReason.MAIN_BRANCH


def test_only_the_unvalidated_push_reason_may_be_forced() -> None:
    with pytest.raises(ValueError, match="unvalidated_push"):
        compute_selection(
            changed_files=CHANGED_AT_E263EEE,
            adjacency_path=ADJ,
            ref_name="dev",
            event_name="push",
            force_full_suite_reason=EnumFullSuiteReason.MERGE_GROUP,
        )


def test_cli_accepts_the_forced_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("\n".join(CHANGED_AT_E263EEE) + "\n")
    rc = main(
        [
            "--changed-files-from",
            str(changed),
            "--ref-name",
            "dev",
            "--event-name",
            "push",
            "--force-full-suite",
            "unvalidated_push",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["is_full_suite"] is True
    assert out["full_suite_reason"] == "unvalidated_push"


def test_cli_without_the_flag_keeps_smart_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changed = tmp_path / "changed.txt"
    changed.write_text("\n".join(CHANGED_AT_E263EEE) + "\n")
    assert (
        main(
            [
                "--changed-files-from",
                str(changed),
                "--ref-name",
                "dev",
                "--event-name",
                "push",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["full_suite_reason"] != "unvalidated_push"
