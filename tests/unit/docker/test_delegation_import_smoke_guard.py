# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12658: assert the runtime image carries a build-time delegation import smoke guard.

These tests pin the existence of the build-time import smoke step in
``docker/Dockerfile.runtime`` and the re-assertion step in the
``docker-build`` CI job. The guard imports the foundation packages
(``omnibase_compat``, ``omnibase_core``) plus the canonical delegation node
handler after all wheels are installed, so a skewed wheel fails the docker
build (and therefore the required ``docker-build`` status check) instead of
degrading silently as a runtime entry-point load failure.

The tests are static (no Docker daemon required) — they guarantee the guard
cannot be silently deleted without a failing unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_WORKFLOW_PATH = _PROJECT_ROOT / ".github" / "workflows" / "docker-build.yml"

# Canonical delegation node import asserted by the guard (OMN-12658).
_DELEGATION_IMPORT = (
    "from omnimarket.nodes.node_llm_delegation_call_effect "
    "import HandlerLlmDelegationCall"
)


@pytest.mark.unit
def test_dockerfile_has_delegation_import_smoke_guard(dockerfile_content: str) -> None:
    """The runtime Dockerfile must import compat + core + the delegation handler at build time."""
    assert "import omnibase_compat" in dockerfile_content
    assert "import omnibase_core" in dockerfile_content
    assert _DELEGATION_IMPORT in dockerfile_content


@pytest.mark.unit
def test_smoke_guard_runs_after_wheel_installs(dockerfile_content: str) -> None:
    """The guard must run after ONEX wheels are installed, not before.

    The delegation handler lives in omnimarket; its install step must precede
    the smoke import so a skewed wheel is actually exercised by the guard.
    """
    omnimarket_install_idx = dockerfile_content.find("omnimarket")
    guard_idx = dockerfile_content.find(_DELEGATION_IMPORT)
    assert omnimarket_install_idx != -1, (
        "omnimarket install step missing from Dockerfile"
    )
    assert guard_idx != -1, "delegation import smoke guard missing from Dockerfile"
    assert omnimarket_install_idx < guard_idx, (
        "delegation import smoke guard must run AFTER the omnimarket install step"
    )


@pytest.mark.unit
def test_ci_job_reasserts_delegation_import_on_pr() -> None:
    """The docker-build CI job must re-assert the delegation import on the loaded image."""
    content = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Delegation import smoke (OMN-12658)" in content
    assert _DELEGATION_IMPORT in content
    # The smoke step must run on PRs (where github.ref != main), so a skewed
    # wheel fails the required status check before merge.
    assert "github.ref != 'refs/heads/main'" in content
