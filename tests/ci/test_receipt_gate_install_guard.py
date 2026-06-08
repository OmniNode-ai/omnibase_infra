# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard tests for the receipt-gate install path (OMN-9271).

Regression coverage asserting that the receipt-gate workflow correctly
installs omnibase_core from source (not from a stale PyPI wheel) and
contains an importability check that catches silent fallback.

Root cause documented in receipt-gate.yml comments (OMN-9198 / OMN-9283):
uv's resolver would silently prefer the PyPI wheel over an editable install
when both advertised the same version. Fix: build a wheel from source, then
install with --no-deps --no-index so the resolver cannot reach PyPI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_RECEIPT_GATE = REPO_ROOT / ".github" / "workflows" / "call-receipt-gate.yml"

# Resolve omni_home via env var (preferred) or walk up from worktree path.
# Worktree layout: $OMNI_HOME/omni_worktrees/<ticket>/<repo>/tests/ci/<file>
#   parents[2] = repo root, parents[3] = ticket dir,
#   parents[4] = omni_worktrees, parents[5] = omni_home
_OMNI_HOME_ENV = os.environ.get("OMNI_HOME")
_OMNI_HOME = (
    Path(_OMNI_HOME_ENV) if _OMNI_HOME_ENV else Path(__file__).resolve().parents[5]
)
_RECEIPT_GATE_YML = (
    _OMNI_HOME / "omnibase_core" / ".github" / "workflows" / "receipt-gate.yml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


class TestCallReceiptGateDelegation:
    """call-receipt-gate.yml must delegate to the centralised omnibase_core workflow."""

    def test_workflow_uses_reusable_omnibase_core_workflow(self) -> None:
        workflow = _load_yaml(CALL_RECEIPT_GATE)
        verify_job = workflow["jobs"]["verify"]
        uses = verify_job.get("uses", "")
        assert uses.startswith(
            "OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml"
        ), (
            "call-receipt-gate.yml must delegate to the omnibase_core reusable "
            f"workflow; found: {uses!r}"
        )

    def test_workflow_tracks_main_or_pinned_sha(self) -> None:
        workflow = _load_yaml(CALL_RECEIPT_GATE)
        uses: str = workflow["jobs"]["verify"]["uses"]
        assert "@" in uses, "workflow_call reference must include a ref after '@'"
        ref = uses.split("@", 1)[1]
        assert ref, "ref after '@' must not be empty"

    def test_caller_has_correct_permissions(self) -> None:
        workflow = _load_yaml(CALL_RECEIPT_GATE)
        perms = workflow.get("permissions", {})
        assert perms.get("contents") == "read"
        assert perms.get("pull-requests") == "read"

    def test_caller_sets_branch_aware_policy_mode(self) -> None:
        workflow = _load_yaml(CALL_RECEIPT_GATE)
        verify_job = workflow["jobs"]["verify"]
        with_block = verify_job.get("with", {})

        policy_expr = with_block.get("branch-policy-mode", "")
        assert "main-release" in policy_expr
        assert "dev-preflight" in policy_expr
        assert "github.event.pull_request.base.ref == 'main'" in policy_expr
        assert "github.event.merge_group.base_ref == 'refs/heads/main'" in policy_expr
        assert "github.event.pull_request.base.ref == 'dev'" in policy_expr
        assert "github.event.merge_group.base_ref == 'refs/heads/dev'" in policy_expr


class TestReceiptGateInstallSafeguards:
    """The reusable receipt-gate.yml from omnibase_core must have install safeguards.

    These tests check the LOCAL copy of receipt-gate.yml in the omnibase_core
    canonical clone under $OMNI_HOME, confirming the reusable workflow is sound.
    In CI the caller downloads the @main version, so this test acts as a canary
    ensuring the reusable workflow retains the OMN-9198/9283 safeguards.
    """

    RECEIPT_GATE_YML = _RECEIPT_GATE_YML

    @pytest.fixture(autouse=True)
    def _skip_if_no_canonical_clone(self) -> None:
        if not self.RECEIPT_GATE_YML.exists():
            pytest.skip(
                "omnibase_core canonical clone not present; skipping local-clone guard"
            )

    def _load_receipt_gate(self) -> dict[str, Any]:
        return _load_yaml(self.RECEIPT_GATE_YML)

    def test_verify_importable_step_present(self) -> None:
        workflow = self._load_receipt_gate()
        steps = workflow["jobs"]["verify"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any(
            "receipt_gate_cli" in name.lower() or "importable" in name.lower()
            for name in step_names
        ), (
            "receipt-gate.yml must contain a step that verifies receipt_gate_cli is "
            "importable after the install (OMN-9198 regression guard). "
            f"Step names found: {step_names}"
        )

    def test_install_uses_no_index_no_deps_wheel(self) -> None:
        """The install step must use --no-index and --no-deps to bypass PyPI.

        This is the OMN-9283 fix (PR #864): build a local wheel, then install
        with --no-index so uv cannot fall back to the stale PyPI package.
        """
        workflow = self._load_receipt_gate()
        steps = workflow["jobs"]["verify"]["steps"]
        install_step = next(
            (s for s in steps if s.get("name") == "Install omnibase_core"),
            None,
        )
        assert install_step is not None, (
            "receipt-gate.yml must have an 'Install omnibase_core' step"
        )
        run_script: str = install_step.get("run", "")
        assert "--no-index" in run_script, (
            "Install omnibase_core step must use --no-index to prevent uv from "
            "resolving the stale PyPI wheel (OMN-9198 / OMN-9283)"
        )
        assert "--no-deps" in run_script, (
            "Install omnibase_core step must use --no-deps to bypass uv's "
            "candidate-selection and force the local wheel (OMN-9283)"
        )

    def test_install_uninstalls_stale_pypi_package_first(self) -> None:
        """The install step must uninstall omnibase-core/compat before reinstalling from source."""
        workflow = self._load_receipt_gate()
        steps = workflow["jobs"]["verify"]["steps"]
        install_step = next(
            (s for s in steps if s.get("name") == "Install omnibase_core"),
            None,
        )
        assert install_step is not None
        run_script: str = install_step.get("run", "")
        assert "uv pip uninstall" in run_script and "omnibase-core" in run_script, (
            "Install omnibase_core step must uninstall the stale PyPI omnibase-core "
            "package before installing from source (OMN-9198)"
        )

    def test_install_builds_wheel_from_source(self) -> None:
        """The install step must build a wheel from source rather than using -e."""
        workflow = self._load_receipt_gate()
        steps = workflow["jobs"]["verify"]["steps"]
        install_step = next(
            (s for s in steps if s.get("name") == "Install omnibase_core"),
            None,
        )
        assert install_step is not None
        run_script: str = install_step.get("run", "")
        assert "uv build --wheel" in run_script, (
            "Install omnibase_core step must build a wheel from source. "
            "Editable installs fail in downstream callers (OMN-9283 root cause)."
        )

    def test_importable_step_prints_install_path_on_failure(self) -> None:
        """On import failure the step must print omnibase_core.__file__ for diagnosis."""
        workflow = self._load_receipt_gate()
        steps = workflow["jobs"]["verify"]["steps"]
        verify_step = next(
            (
                s
                for s in steps
                if "receipt_gate_cli" in s.get("name", "").lower()
                or "importable" in s.get("name", "").lower()
            ),
            None,
        )
        assert verify_step is not None
        run_script: str = verify_step.get("run", "")
        assert "omnibase_core.__file__" in run_script, (
            "The importability check step must print omnibase_core.__file__ on "
            "failure so engineers can diagnose which wheel was installed"
        )
