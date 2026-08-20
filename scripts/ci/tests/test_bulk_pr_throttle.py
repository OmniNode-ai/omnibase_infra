# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for bulk_pr_throttle.py (OMN-16284).

Covers the four explicit test categories from the ticket: wave partitioning,
threshold blocking (mocked gh calls), dry-run plan output, and refusal
paths — plus the gh CLI integration seam and the CLI entrypoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_CI = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_CI))


# ---------------------------------------------------------------------------
# Wave partitioning
# ---------------------------------------------------------------------------


class TestPartitionIntoWaves:
    def test_default_size_evenly_divides(self):
        from bulk_pr_throttle import partition_into_waves

        waves = partition_into_waves(list(range(1, 21)), 10)
        assert waves == [tuple(range(1, 11)), tuple(range(11, 21))]

    def test_custom_size(self):
        from bulk_pr_throttle import partition_into_waves

        waves = partition_into_waves([1, 2, 3, 4, 5], 2)
        assert waves == [(1, 2), (3, 4), (5,)]

    def test_uneven_last_wave(self):
        from bulk_pr_throttle import partition_into_waves

        waves = partition_into_waves(list(range(1, 12)), 5)
        assert waves == [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11,)]

    def test_single_wave_when_fewer_than_wave_size(self):
        from bulk_pr_throttle import partition_into_waves

        waves = partition_into_waves([1, 2, 3], 10)
        assert waves == [(1, 2, 3)]

    def test_zero_wave_size_raises(self):
        from bulk_pr_throttle import partition_into_waves

        with pytest.raises(ValueError, match="wave_size must be >= 1"):
            partition_into_waves([1, 2, 3], 0)

    def test_negative_wave_size_raises(self):
        from bulk_pr_throttle import partition_into_waves

        with pytest.raises(ValueError, match="wave_size must be >= 1"):
            partition_into_waves([1, 2, 3], -5)

    def test_exceeds_hard_ceiling_raises(self):
        from bulk_pr_throttle import MAX_WAVE_SIZE, partition_into_waves

        with pytest.raises(ValueError, match="hard ceiling"):
            partition_into_waves(list(range(1, 100)), MAX_WAVE_SIZE + 1)

    def test_exactly_at_ceiling_is_allowed(self):
        from bulk_pr_throttle import MAX_WAVE_SIZE, partition_into_waves

        waves = partition_into_waves(list(range(1, MAX_WAVE_SIZE + 1)), MAX_WAVE_SIZE)
        assert len(waves) == 1
        assert len(waves[0]) == MAX_WAVE_SIZE


# ---------------------------------------------------------------------------
# Refusal paths: total PR cap, unknown operation, empty inputs, missing owner/repo
# ---------------------------------------------------------------------------


class TestRefusalPaths:
    def test_within_default_cap_does_not_raise(self):
        from bulk_pr_throttle import DEFAULT_MAX_TOTAL_PRS, validate_total_prs

        validate_total_prs(
            list(range(1, DEFAULT_MAX_TOTAL_PRS + 1)),
            max_total_prs=DEFAULT_MAX_TOTAL_PRS,
            explicit_max_total_prs=False,
        )  # no raise

    def test_exceeds_default_cap_without_explicit_flag_raises(self):
        from bulk_pr_throttle import (
            DEFAULT_MAX_TOTAL_PRS,
            TotalPrLimitExceededError,
            validate_total_prs,
        )

        with pytest.raises(TotalPrLimitExceededError, match="explicit"):
            validate_total_prs(
                list(range(1, DEFAULT_MAX_TOTAL_PRS + 2)),
                max_total_prs=DEFAULT_MAX_TOTAL_PRS,
                explicit_max_total_prs=False,
            )

    def test_exceeds_cap_with_explicit_flag_does_not_raise(self):
        from bulk_pr_throttle import validate_total_prs

        validate_total_prs(
            list(range(1, 200)),
            max_total_prs=250,
            explicit_max_total_prs=True,
        )  # no raise

    def test_run_bulk_operation_rejects_empty_owner(self):
        from bulk_pr_throttle import BulkPrThrottleError, run_bulk_operation

        with pytest.raises(BulkPrThrottleError, match="owner"):
            run_bulk_operation(
                owner="",
                repo="omnibase_infra",
                pr_numbers=[1, 2],
                operation="noop-dry-run",
                dry_run=True,
            )

    def test_run_bulk_operation_rejects_empty_repo(self):
        from bulk_pr_throttle import BulkPrThrottleError, run_bulk_operation

        with pytest.raises(BulkPrThrottleError, match="repo"):
            run_bulk_operation(
                owner="OmniNode-ai",
                repo="",
                pr_numbers=[1, 2],
                operation="noop-dry-run",
                dry_run=True,
            )

    def test_run_bulk_operation_rejects_unknown_operation(self):
        from bulk_pr_throttle import BulkPrThrottleError, run_bulk_operation

        with pytest.raises(BulkPrThrottleError, match="unknown operation"):
            run_bulk_operation(
                owner="OmniNode-ai",
                repo="omnibase_infra",
                pr_numbers=[1],
                operation="delete-everything",
                dry_run=True,
            )

    def test_run_bulk_operation_rejects_empty_pr_list(self):
        from bulk_pr_throttle import BulkPrThrottleError, run_bulk_operation

        with pytest.raises(BulkPrThrottleError, match="non-empty"):
            run_bulk_operation(
                owner="OmniNode-ai",
                repo="omnibase_infra",
                pr_numbers=[],
                operation="noop-dry-run",
                dry_run=True,
            )

    def test_run_bulk_operation_requires_callables_outside_dry_run(self):
        from bulk_pr_throttle import BulkPrThrottleError, run_bulk_operation

        with pytest.raises(BulkPrThrottleError, match="required"):
            run_bulk_operation(
                owner="OmniNode-ai",
                repo="omnibase_infra",
                pr_numbers=[1, 2],
                operation="update-branch",
                dry_run=False,
                get_queue_depth=None,
                apply_pr_operation=None,
            )

    def test_run_bulk_operation_exceeding_total_cap_raises_before_any_gh_call(self):
        from bulk_pr_throttle import (
            DEFAULT_MAX_TOTAL_PRS,
            TotalPrLimitExceededError,
            run_bulk_operation,
        )

        calls = {"queue_depth": 0, "apply": 0}

        def get_queue_depth() -> int:
            calls["queue_depth"] += 1
            return 0

        def apply_pr_operation(owner: str, repo: str, pr: int, operation: str):
            calls["apply"] += 1
            raise AssertionError("should never be called")

        with pytest.raises(TotalPrLimitExceededError):
            run_bulk_operation(
                owner="OmniNode-ai",
                repo="onex_change_control",
                pr_numbers=list(range(1, DEFAULT_MAX_TOTAL_PRS + 5)),
                operation="update-branch",
                dry_run=False,
                get_queue_depth=get_queue_depth,
                apply_pr_operation=apply_pr_operation,
            )
        assert calls == {"queue_depth": 0, "apply": 0}


# ---------------------------------------------------------------------------
# Threshold blocking (mocked queue-depth callable — the "gh call" seam)
# ---------------------------------------------------------------------------


class TestWaitForQueueDepth:
    def test_immediate_pass_when_already_below_threshold(self):
        from bulk_pr_throttle import wait_for_queue_depth

        sleeps: list[float] = []
        depth = wait_for_queue_depth(
            get_queue_depth=lambda: 42,
            threshold=150,
            poll_seconds=5.0,
            max_wait_seconds=60.0,
            sleep_fn=sleeps.append,
        )
        assert depth == 42
        assert sleeps == []

    def test_blocks_and_polls_until_below_threshold(self):
        from bulk_pr_throttle import wait_for_queue_depth

        depths = iter([200, 180, 160, 100])
        sleeps: list[float] = []
        depth = wait_for_queue_depth(
            get_queue_depth=lambda: next(depths),
            threshold=150,
            poll_seconds=10.0,
            max_wait_seconds=1000.0,
            sleep_fn=sleeps.append,
        )
        assert depth == 100
        assert sleeps == [10.0, 10.0, 10.0]

    def test_boundary_depth_equal_to_threshold_does_not_block(self):
        from bulk_pr_throttle import wait_for_queue_depth

        sleeps: list[float] = []
        depth = wait_for_queue_depth(
            get_queue_depth=lambda: 150,
            threshold=150,
            poll_seconds=5.0,
            max_wait_seconds=60.0,
            sleep_fn=sleeps.append,
        )
        assert depth == 150
        assert sleeps == []

    def test_persistently_high_depth_times_out(self):
        from bulk_pr_throttle import QueueDepthTimeoutError, wait_for_queue_depth

        sleeps: list[float] = []
        with pytest.raises(QueueDepthTimeoutError, match="still above threshold"):
            wait_for_queue_depth(
                get_queue_depth=lambda: 999,
                threshold=150,
                poll_seconds=10.0,
                max_wait_seconds=25.0,
                sleep_fn=sleeps.append,
            )
        # polls until waited >= max_wait_seconds, never silently proceeds
        assert sum(sleeps) >= 25.0

    def test_no_bypass_flag_exists_for_the_gate(self):
        """The gate has no force/skip parameter — this is the mechanical guard."""
        import inspect

        from bulk_pr_throttle import wait_for_queue_depth

        params = set(inspect.signature(wait_for_queue_depth).parameters)
        assert not params & {"force", "skip", "bypass", "ignore_threshold"}


# ---------------------------------------------------------------------------
# Full run_bulk_operation flow (waves + logging + receipts)
# ---------------------------------------------------------------------------


class TestRunBulkOperationFlow:
    def test_dry_run_never_touches_gh_and_prints_plan(self, capsys):
        from bulk_pr_throttle import run_bulk_operation

        report = run_bulk_operation(
            owner="OmniNode-ai",
            repo="onex_change_control",
            pr_numbers=[1, 2, 3, 4, 5],
            operation="rerun-failed",
            wave_size=2,
            dry_run=True,
            get_queue_depth=None,
            apply_pr_operation=None,
        )
        assert report.dry_run is True
        assert len(report.waves) == 3
        assert report.waves[0].pr_numbers == (1, 2)
        assert report.waves[1].pr_numbers == (3, 4)
        assert report.waves[2].pr_numbers == (5,)
        for wave in report.waves:
            assert wave.outcomes == ()

        out = capsys.readouterr().out
        assert "DRY-RUN plan" in out
        assert "wave 1:" in out
        assert "wave 3:" in out

    def test_multi_wave_flow_calls_apply_per_pr_and_polls_depth_per_wave(self):
        from bulk_pr_throttle import PrOutcome, run_bulk_operation

        depth_calls = 0

        def get_queue_depth() -> int:
            nonlocal depth_calls
            depth_calls += 1
            return 10  # always under threshold

        applied: list[tuple[str, str, int, str]] = []

        def apply_pr_operation(
            owner: str, repo: str, pr: int, operation: str
        ) -> PrOutcome:
            applied.append((owner, repo, pr, operation))
            return PrOutcome(pr_number=pr, success=True, detail="ok")

        report = run_bulk_operation(
            owner="OmniNode-ai",
            repo="omnibase_infra",
            pr_numbers=[101, 102, 103, 104, 105],
            operation="update-branch",
            wave_size=2,
            queue_depth_threshold=150,
            dry_run=False,
            get_queue_depth=get_queue_depth,
            apply_pr_operation=apply_pr_operation,
        )

        assert len(report.waves) == 3
        assert [pr for _, _, pr, _ in applied] == [101, 102, 103, 104, 105]
        assert all(op == "update-branch" for _, _, _, op in applied)
        # one queue-depth poll before each wave, plus one after each wave
        assert depth_calls == 6
        for wave in report.waves:
            assert wave.queue_depth_before == 10
            assert wave.queue_depth_after == 10
            assert all(o.success for o in wave.outcomes)

    def test_flow_blocks_mid_batch_when_a_later_wave_sees_high_depth(self):
        """Threshold blocking (mocked gh call) applies per-wave, not just once."""
        from bulk_pr_throttle import PrOutcome, run_bulk_operation

        # wave 1 poll: 20 (ok). wave 1 after-poll: 20.
        # wave 2 poll: 300 (blocks), then 300 (blocks), then 50 (ok). wave 2 after: 50.
        depth_sequence = iter([20, 20, 300, 300, 50, 50])
        sleeps: list[float] = []

        def get_queue_depth() -> int:
            return next(depth_sequence)

        def apply_pr_operation(
            owner: str, repo: str, pr: int, operation: str
        ) -> PrOutcome:
            return PrOutcome(pr_number=pr, success=True, detail="ok")

        report = run_bulk_operation(
            owner="OmniNode-ai",
            repo="onex_change_control",
            pr_numbers=[1, 2, 3, 4],
            operation="rerun-failed",
            wave_size=2,
            queue_depth_threshold=150,
            dry_run=False,
            get_queue_depth=get_queue_depth,
            apply_pr_operation=apply_pr_operation,
            poll_seconds=1.0,
            max_wait_seconds=100.0,
            sleep_fn=sleeps.append,
        )
        assert len(report.waves) == 2
        assert report.waves[1].queue_depth_before == 50
        assert sleeps == [1.0, 1.0]

    def test_logs_timestamp_count_and_depth_before_after_to_stdout(self, capsys):
        from bulk_pr_throttle import PrOutcome, run_bulk_operation

        run_bulk_operation(
            owner="OmniNode-ai",
            repo="onex_change_control",
            pr_numbers=[6751, 6752, 6753],
            operation="rerun-failed",
            wave_size=10,
            dry_run=False,
            get_queue_depth=lambda: 5,
            apply_pr_operation=lambda o, r, pr, op: PrOutcome(
                pr_number=pr, success=True, detail="ok"
            ),
        )
        out = capsys.readouterr().out
        assert "wave 1/1" in out
        assert "depth_before=5" in out
        assert "depth_after=5" in out
        assert "6751" in out and "6752" in out and "6753" in out


# ---------------------------------------------------------------------------
# Receipt file
# ---------------------------------------------------------------------------


class TestWriteReceipt:
    def test_receipt_json_shape(self, tmp_path):
        from bulk_pr_throttle import PrOutcome, run_bulk_operation, write_receipt

        report = run_bulk_operation(
            owner="OmniNode-ai",
            repo="onex_change_control",
            pr_numbers=[1, 2],
            operation="rerun-failed",
            wave_size=10,
            dry_run=False,
            get_queue_depth=lambda: 3,
            apply_pr_operation=lambda o, r, pr, op: PrOutcome(
                pr_number=pr, success=True, detail="ok"
            ),
        )
        receipt_path = tmp_path / "receipt.json"
        write_receipt(report, receipt_path)

        data = json.loads(receipt_path.read_text())
        assert data["owner"] == "OmniNode-ai"
        assert data["repo"] == "onex_change_control"
        assert data["operation"] == "rerun-failed"
        assert len(data["waves"]) == 1
        wave = data["waves"][0]
        assert wave["queue_depth_before"] == 3
        assert wave["queue_depth_after"] == 3
        assert wave["pr_numbers"] == [1, 2]
        assert len(wave["outcomes"]) == 2
        assert wave["outcomes"][0]["pr_number"] == 1
        assert "started_at" in wave
        assert "completed_at" in wave

    def test_receipt_creates_parent_dirs(self, tmp_path):
        from bulk_pr_throttle import run_bulk_operation, write_receipt

        report = run_bulk_operation(
            owner="OmniNode-ai",
            repo="onex_change_control",
            pr_numbers=[1],
            operation="noop-dry-run",
            dry_run=True,
        )
        nested = tmp_path / "a" / "b" / "c" / "receipt.json"
        write_receipt(report, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# gh CLI integration seam (mock _run_gh — never call the real GitHub API)
# ---------------------------------------------------------------------------


class TestGhIntegration:
    def test_gh_queue_depth_parses_jq_output(self, monkeypatch):
        import bulk_pr_throttle

        def fake_run_gh(args):
            assert args[0] == "api"
            assert "actions/runs?status=queued" in args[1]
            assert args[-2:] == ["--jq", ".total_count"]
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="42\n", stderr=""
            )

        monkeypatch.setattr(bulk_pr_throttle, "_run_gh", fake_run_gh)
        assert (
            bulk_pr_throttle.gh_queue_depth("OmniNode-ai", "onex_change_control") == 42
        )

    def test_gh_queue_depth_raises_on_nonzero_exit(self, monkeypatch):
        import bulk_pr_throttle

        def fake_run_gh(args):
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="", stderr="HTTP 502"
            )

        monkeypatch.setattr(bulk_pr_throttle, "_run_gh", fake_run_gh)
        with pytest.raises(bulk_pr_throttle.BulkPrThrottleError, match="HTTP 502"):
            bulk_pr_throttle.gh_queue_depth("OmniNode-ai", "onex_change_control")

    def test_gh_apply_pr_operation_update_branch(self, monkeypatch):
        import bulk_pr_throttle

        seen = {}

        def fake_run_gh(args):
            seen["args"] = args
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="ok", stderr=""
            )

        monkeypatch.setattr(bulk_pr_throttle, "_run_gh", fake_run_gh)
        outcome = bulk_pr_throttle.gh_apply_pr_operation(
            "OmniNode-ai", "omnibase_infra", 2805, "update-branch"
        )
        assert outcome.success is True
        assert outcome.pr_number == 2805
        assert seen["args"] == [
            "api",
            "-X",
            "PUT",
            "repos/OmniNode-ai/omnibase_infra/pulls/2805/update-branch",
        ]

    def test_gh_apply_pr_operation_arm_automerge(self, monkeypatch):
        import bulk_pr_throttle

        def fake_run_gh(args):
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=""
            )

        monkeypatch.setattr(bulk_pr_throttle, "_run_gh", fake_run_gh)
        outcome = bulk_pr_throttle.gh_apply_pr_operation(
            "OmniNode-ai", "omnibase_infra", 2805, "arm-automerge"
        )
        assert outcome.success is True

    def test_gh_rerun_failed_no_failed_runs(self, monkeypatch):
        import bulk_pr_throttle

        def fake_run_gh(args):
            if args[:2] == ["pr", "view"]:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=json.dumps({"headRefOid": "abc123"}),
                    stderr="",
                )
            if args[0] == "api" and "actions/runs?head_sha=" in args[1]:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=json.dumps({"workflow_runs": []}),
                    stderr="",
                )
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr(bulk_pr_throttle, "_run_gh", fake_run_gh)
        outcome = bulk_pr_throttle.gh_apply_pr_operation(
            "OmniNode-ai", "onex_change_control", 6751, "rerun-failed"
        )
        assert outcome.success is True
        assert "no failed runs" in outcome.detail

    def test_gh_rerun_failed_reruns_matching_runs(self, monkeypatch):
        import bulk_pr_throttle

        rerun_calls = []

        def fake_run_gh(args):
            if args[:2] == ["pr", "view"]:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=json.dumps({"headRefOid": "deadbeef"}),
                    stderr="",
                )
            if args[0] == "api" and "actions/runs?head_sha=" in args[1]:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "workflow_runs": [
                                {"id": 111, "conclusion": "failure"},
                                {"id": 222, "conclusion": "success"},
                                {"id": 333, "conclusion": "failure"},
                            ]
                        }
                    ),
                    stderr="",
                )
            if args[:2] == ["api", "-X"] and "rerun-failed-jobs" in args[-1]:
                rerun_calls.append(args[-1])
                return subprocess.CompletedProcess(
                    args=args, returncode=0, stdout="", stderr=""
                )
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr(bulk_pr_throttle, "_run_gh", fake_run_gh)
        outcome = bulk_pr_throttle.gh_apply_pr_operation(
            "OmniNode-ai", "onex_change_control", 6751, "rerun-failed"
        )
        assert outcome.success is True
        assert len(rerun_calls) == 2
        assert any("111" in c for c in rerun_calls)
        assert any("333" in c for c in rerun_calls)
        assert not any("222" in c for c in rerun_calls)


# ---------------------------------------------------------------------------
# PR-number parsing
# ---------------------------------------------------------------------------


class TestParsePrNumbers:
    def test_parses_comma_separated(self):
        from bulk_pr_throttle import _parse_pr_numbers

        assert _parse_pr_numbers("1,2,3") == [1, 2, 3]

    def test_strips_whitespace(self):
        from bulk_pr_throttle import _parse_pr_numbers

        assert _parse_pr_numbers(" 1, 2 ,3 ") == [1, 2, 3]

    def test_ignores_empty_chunks(self):
        from bulk_pr_throttle import _parse_pr_numbers

        assert _parse_pr_numbers("1,,2,") == [1, 2]

    def test_invalid_number_raises(self):
        from bulk_pr_throttle import BulkPrThrottleError, _parse_pr_numbers

        with pytest.raises(BulkPrThrottleError, match="invalid PR number"):
            _parse_pr_numbers("1,abc,3")

    def test_all_empty_raises(self):
        from bulk_pr_throttle import BulkPrThrottleError, _parse_pr_numbers

        with pytest.raises(BulkPrThrottleError, match="zero PR numbers"):
            _parse_pr_numbers(" , , ")


# ---------------------------------------------------------------------------
# CLI entrypoint (main)
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_dry_run_end_to_end(self, capsys, tmp_path):
        from bulk_pr_throttle import main

        receipt = tmp_path / "receipt.json"
        result = main(
            [
                "--owner",
                "OmniNode-ai",
                "--repo",
                "onex_change_control",
                "--prs",
                "6751,6752,6753",
                "--operation",
                "rerun-failed",
                "--wave-size",
                "2",
                "--dry-run",
                "--receipt",
                str(receipt),
            ]
        )
        assert result == 0
        out = capsys.readouterr().out
        assert "DRY-RUN plan" in out
        assert receipt.exists()
        data = json.loads(receipt.read_text())
        assert data["dry_run"] is True

    def test_missing_owner_is_a_hard_argparse_error(self):
        from bulk_pr_throttle import main

        with pytest.raises(SystemExit):
            main(
                [
                    "--repo",
                    "onex_change_control",
                    "--prs",
                    "1",
                    "--operation",
                    "noop-dry-run",
                    "--dry-run",
                ]
            )

    def test_missing_repo_is_a_hard_argparse_error(self):
        from bulk_pr_throttle import main

        with pytest.raises(SystemExit):
            main(
                [
                    "--owner",
                    "OmniNode-ai",
                    "--prs",
                    "1",
                    "--operation",
                    "noop-dry-run",
                    "--dry-run",
                ]
            )

    def test_refuses_batch_larger_than_default_cap_without_flag(self, capsys):
        from bulk_pr_throttle import DEFAULT_MAX_TOTAL_PRS, main

        prs = ",".join(str(n) for n in range(1, DEFAULT_MAX_TOTAL_PRS + 5))
        result = main(
            [
                "--owner",
                "OmniNode-ai",
                "--repo",
                "onex_change_control",
                "--prs",
                prs,
                "--operation",
                "noop-dry-run",
                "--dry-run",
            ]
        )
        assert result == 1
        err = capsys.readouterr().err
        assert "REFUSED" in err

    def test_explicit_max_total_prs_flag_allows_large_batch(self, capsys):
        from bulk_pr_throttle import DEFAULT_MAX_TOTAL_PRS, main

        prs = ",".join(str(n) for n in range(1, DEFAULT_MAX_TOTAL_PRS + 5))
        result = main(
            [
                "--owner",
                "OmniNode-ai",
                "--repo",
                "onex_change_control",
                "--prs",
                prs,
                "--operation",
                "noop-dry-run",
                "--dry-run",
                "--max-total-prs",
                str(DEFAULT_MAX_TOTAL_PRS + 10),
            ]
        )
        assert result == 0

    def test_invalid_operation_is_a_hard_argparse_error(self):
        from bulk_pr_throttle import main

        with pytest.raises(SystemExit):
            main(
                [
                    "--owner",
                    "OmniNode-ai",
                    "--repo",
                    "onex_change_control",
                    "--prs",
                    "1",
                    "--operation",
                    "delete-everything",
                    "--dry-run",
                ]
            )

    def test_wave_size_over_hard_ceiling_refuses(self, capsys):
        from bulk_pr_throttle import MAX_WAVE_SIZE, main

        result = main(
            [
                "--owner",
                "OmniNode-ai",
                "--repo",
                "onex_change_control",
                "--prs",
                "1,2,3",
                "--operation",
                "noop-dry-run",
                "--dry-run",
                "--wave-size",
                str(MAX_WAVE_SIZE + 1),
            ]
        )
        assert result == 1
        err = capsys.readouterr().err
        assert "REFUSED" in err

    def test_non_dry_run_uses_real_gh_seam(self, monkeypatch, tmp_path):
        """Non-dry-run wires get_queue_depth/apply_pr_operation to the gh functions."""
        import bulk_pr_throttle
        from bulk_pr_throttle import PrOutcome, main

        monkeypatch.setattr(bulk_pr_throttle, "gh_queue_depth", lambda owner, repo: 5)
        monkeypatch.setattr(
            bulk_pr_throttle,
            "gh_apply_pr_operation",
            lambda owner, repo, pr, op: PrOutcome(
                pr_number=pr, success=True, detail="ok"
            ),
        )
        receipt = tmp_path / "receipt.json"
        result = main(
            [
                "--owner",
                "OmniNode-ai",
                "--repo",
                "onex_change_control",
                "--prs",
                "1,2",
                "--operation",
                "rerun-failed",
                "--receipt",
                str(receipt),
            ]
        )
        assert result == 0
        data = json.loads(receipt.read_text())
        assert data["dry_run"] is False
        assert data["waves"][0]["queue_depth_before"] == 5

    def test_non_dry_run_reports_failure_exit_code_on_pr_failure(
        self, monkeypatch, tmp_path
    ):
        import bulk_pr_throttle
        from bulk_pr_throttle import PrOutcome, main

        monkeypatch.setattr(bulk_pr_throttle, "gh_queue_depth", lambda owner, repo: 5)
        monkeypatch.setattr(
            bulk_pr_throttle,
            "gh_apply_pr_operation",
            lambda owner, repo, pr, op: PrOutcome(
                pr_number=pr, success=False, detail="boom"
            ),
        )
        receipt = tmp_path / "receipt.json"
        result = main(
            [
                "--owner",
                "OmniNode-ai",
                "--repo",
                "onex_change_control",
                "--prs",
                "1",
                "--operation",
                "rerun-failed",
                "--receipt",
                str(receipt),
            ]
        )
        assert result == 1
