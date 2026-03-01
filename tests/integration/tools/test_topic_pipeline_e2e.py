# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""E2E regression tests for the contract-driven topic enum pipeline (OMN-3186).

Converts PR #493 manual verification session into automated pytest assertions.
PR #493 was merged with zero files changed — all test evidence existed only as
PR description prose. These tests provide the regression protection.

Ticket: OMN-3186
Parent: OMN-3185
Plan ref: Task 1 in docs/plans/2026-02-28-integration-hardening-next-steps.md

Test Coverage:
    1. test_check_passes_on_clean_state        — --check exits 0 on clean repo
    2. test_generate_is_idempotent             — two consecutive --generate runs produce identical output
    3. test_check_detects_stale_enum           — --check exits 1 when a generated file is corrupted
    4. test_generate_restores_from_stale       — --generate fixes stale enum; --check then exits 0

CRITICAL: If test_check_detects_stale_enum fails (--check exits 0 on stale input),
that is a P0 bug in generate_topic_enums.py — file a separate ticket immediately.

All tests:
    - Are marked @pytest.mark.integration (auto-applied by tests/integration/conftest.py)
    - Clean up any mutations to generated files using try/finally
    - Run from the worktree root; path resolution is repo-root-relative
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.path_utils import find_project_root

pytestmark = pytest.mark.integration

_REPO_ROOT = find_project_root(Path(__file__).resolve().parent)
_SCRIPT = _REPO_ROOT / "scripts" / "generate_topic_enums.py"
_OUTPUT_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "enums" / "generated"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    """Run generate_topic_enums.py with the given arguments via uv run.

    Returns the completed process (never raises CalledProcessError — callers
    assert on returncode themselves so failures are clearly attributed).
    """
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )


class TestTopicPipelineE2E:
    """E2E regression suite for the contract-driven topic enum pipeline.

    Each test is independent: mutations are reverted in try/finally blocks
    so a failure in one test does not cascade to the next.
    """

    @pytest.mark.skipif(
        not _SCRIPT.is_file(),
        reason="generate_topic_enums.py script not found",
    )
    def test_check_passes_on_clean_state(self) -> None:
        """--check exits 0 on a clean (up-to-date) repository.

        Pre-condition: The generated enum files match the current contracts.
        This test first calls --generate to guarantee the pre-condition, then
        asserts --check agrees.

        This mirrors PR #493 happy-path verification: "check exits 0 when
        generated files are current."
        """
        # Ensure clean baseline via --generate (idempotent).
        generate_result = _run_script("--generate")
        assert generate_result.returncode == 0, (
            f"--generate setup failed (rc={generate_result.returncode}).\n"
            f"stdout: {generate_result.stdout}\n"
            f"stderr: {generate_result.stderr}"
        )

        # Now --check must agree that files are up to date.
        check_result = _run_script("--check")
        assert check_result.returncode == 0, (
            f"--check reported drift on a freshly-generated tree (rc={check_result.returncode}).\n"
            f"stdout: {check_result.stdout}\n"
            f"stderr: {check_result.stderr}"
        )
        assert "CHECK PASSED" in check_result.stdout, (
            f"Expected 'CHECK PASSED' in stdout.\nstdout: {check_result.stdout}"
        )

    @pytest.mark.skipif(
        not _SCRIPT.is_file(),
        reason="generate_topic_enums.py script not found",
    )
    def test_generate_is_idempotent(self) -> None:
        """Two consecutive --generate runs produce byte-identical output files.

        Idempotency is a correctness invariant of the pipeline: re-running
        --generate must not introduce spurious diffs (e.g. timestamps, ordering
        instability).

        This mirrors PR #493 idempotency verification.
        """
        # First run — establish baseline.
        result1 = _run_script("--generate")
        assert result1.returncode == 0, (
            f"First --generate failed (rc={result1.returncode}).\n"
            f"stdout: {result1.stdout}\nstderr: {result1.stderr}"
        )

        # Capture file contents after first run.
        snapshot_after_first: dict[str, str] = {}
        if _OUTPUT_DIR.exists():
            for f in sorted(_OUTPUT_DIR.iterdir()):
                if f.is_file() and f.suffix == ".py":
                    snapshot_after_first[f.name] = f.read_text(encoding="utf-8")

        assert snapshot_after_first, (
            f"No .py files found in output dir after first --generate: {_OUTPUT_DIR}"
        )

        # Second run — must produce the same result.
        result2 = _run_script("--generate")
        assert result2.returncode == 0, (
            f"Second --generate failed (rc={result2.returncode}).\n"
            f"stdout: {result2.stdout}\nstderr: {result2.stderr}"
        )

        snapshot_after_second: dict[str, str] = {}
        if _OUTPUT_DIR.exists():
            for f in sorted(_OUTPUT_DIR.iterdir()):
                if f.is_file() and f.suffix == ".py":
                    snapshot_after_second[f.name] = f.read_text(encoding="utf-8")

        assert snapshot_after_first == snapshot_after_second, (
            "Output files differ between two consecutive --generate runs.\n"
            f"Files after run 1: {sorted(snapshot_after_first)}\n"
            f"Files after run 2: {sorted(snapshot_after_second)}\n"
            "Diffs:\n"
            + "\n".join(
                f"  {name}: content changed"
                for name in set(snapshot_after_first) | set(snapshot_after_second)
                if snapshot_after_first.get(name) != snapshot_after_second.get(name)
            )
        )

    @pytest.mark.skipif(
        not _SCRIPT.is_file(),
        reason="generate_topic_enums.py script not found",
    )
    def test_check_detects_stale_enum(self) -> None:
        """--check exits 1 (not 0) when a generated file is corrupted/stale.

        CRITICAL: If this test fails (--check exits 0 on a corrupted file),
        that is a P0 bug in generate_topic_enums.py — the drift detector is
        broken. File a separate ticket immediately.

        This mirrors PR #493 error-path verification: "check exits 1 when
        generated files are stale."
        """
        # Ensure clean baseline.
        generate_result = _run_script("--generate")
        assert generate_result.returncode == 0, (
            f"--generate setup failed (rc={generate_result.returncode}).\n"
            f"stderr: {generate_result.stderr}"
        )

        # Find a generated enum_*_topic.py to corrupt.
        target: Path | None = None
        original_content: str = ""
        for f in sorted(_OUTPUT_DIR.iterdir()):
            if (
                f.is_file()
                and f.name.startswith("enum_")
                and f.name.endswith("_topic.py")
            ):
                target = f
                original_content = f.read_text(encoding="utf-8")
                break

        assert target is not None, (
            f"No enum_*_topic.py files found in {_OUTPUT_DIR} to corrupt for this test."
        )

        # Corrupt the file, then verify --check detects it.
        try:
            target.write_text(
                original_content + "\n# CORRUPTED BY TEST: intentional stale marker\n",
                encoding="utf-8",
            )

            check_result = _run_script("--check")

            # CRITICAL assertion: --check MUST exit 1 on stale content.
            assert check_result.returncode == 1, (
                f"CRITICAL P0 BUG: --check returned {check_result.returncode} "
                f"instead of 1 on a corrupted generated file.\n"
                f"Corrupted file: {target.relative_to(_REPO_ROOT)}\n"
                f"stdout: {check_result.stdout}\n"
                f"stderr: {check_result.stderr}\n"
                "The drift detector in generate_topic_enums.py is broken. "
                "File a P0 ticket immediately."
            )

            assert "CHECK FAILED" in check_result.stdout, (
                f"Expected 'CHECK FAILED' in stdout when file is stale.\n"
                f"stdout: {check_result.stdout}"
            )

        finally:
            # Restore the original content unconditionally.
            target.write_text(original_content, encoding="utf-8")

    @pytest.mark.skipif(
        not _SCRIPT.is_file(),
        reason="generate_topic_enums.py script not found",
    )
    def test_generate_restores_from_stale(self) -> None:
        """--generate fixes a stale enum file; --check then exits 0.

        This verifies the full repair cycle: corrupt a generated file, run
        --generate to restore it, then confirm --check agrees it is clean.

        This mirrors PR #493 repair-path verification.
        """
        # Ensure clean baseline.
        generate_result = _run_script("--generate")
        assert generate_result.returncode == 0, (
            f"--generate setup failed (rc={generate_result.returncode}).\n"
            f"stderr: {generate_result.stderr}"
        )

        # Find a generated enum_*_topic.py to corrupt.
        target: Path | None = None
        original_content: str = ""
        for f in sorted(_OUTPUT_DIR.iterdir()):
            if (
                f.is_file()
                and f.name.startswith("enum_")
                and f.name.endswith("_topic.py")
            ):
                target = f
                original_content = f.read_text(encoding="utf-8")
                break

        assert target is not None, (
            f"No enum_*_topic.py files found in {_OUTPUT_DIR} to corrupt for this test."
        )

        try:
            # Corrupt the file.
            target.write_text(
                original_content + "\n# CORRUPTED BY TEST: intentional stale marker\n",
                encoding="utf-8",
            )

            # Confirm it is now stale (pre-condition).
            pre_check = _run_script("--check")
            assert pre_check.returncode == 1, (
                f"Pre-condition failed: --check returned {pre_check.returncode} "
                f"instead of 1 on a corrupted file.\n"
                f"stdout: {pre_check.stdout}"
            )

            # Run --generate to repair.
            repair_result = _run_script("--generate")
            assert repair_result.returncode == 0, (
                f"--generate failed to repair stale file (rc={repair_result.returncode}).\n"
                f"stdout: {repair_result.stdout}\nstderr: {repair_result.stderr}"
            )

            # --check must now pass.
            post_check = _run_script("--check")
            assert post_check.returncode == 0, (
                f"--check still reports drift after --generate repair "
                f"(rc={post_check.returncode}).\n"
                f"stdout: {post_check.stdout}\nstderr: {post_check.stderr}"
            )

            assert "CHECK PASSED" in post_check.stdout, (
                f"Expected 'CHECK PASSED' after repair.\nstdout: {post_check.stdout}"
            )

            # Verify the restored content matches the original exactly.
            restored_content = target.read_text(encoding="utf-8")
            assert restored_content == original_content, (
                f"Restored content differs from original for {target.name}.\n"
                "This indicates --generate is not fully deterministic."
            )

        finally:
            # Restore original content unconditionally (belt-and-suspenders —
            # the --generate call above should have already done this).
            if target.exists():
                current = target.read_text(encoding="utf-8")
                if current != original_content:
                    target.write_text(original_content, encoding="utf-8")
