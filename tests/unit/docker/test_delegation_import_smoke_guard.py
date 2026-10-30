# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12658: assert the runtime image carries a build-time delegation import smoke guard.

These tests pin the existence of the build-time import smoke step in
``docker/Dockerfile.runtime`` and the re-assertion step in the
``docker-build`` CI job. The guard imports the foundation packages
(``omnibase_compat``, ``omnibase_core``, ``omnibase_infra``) after all wheels
are installed, so a skewed wheel fails the docker
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

# Foundation import asserted by the guard (OMN-12658). OMN-21279: the image
# carries no omnimarket, so the guard no longer imports a market handler.
_FOUNDATION_IMPORT = (
    "import omnibase_compat; import omnibase_core; import omnibase_infra"
)


@pytest.mark.unit
def test_dockerfile_has_foundation_import_smoke_guard(dockerfile_content: str) -> None:
    """The runtime Dockerfile must import compat + core + infra at build time."""
    assert "import omnibase_compat" in dockerfile_content
    assert "import omnibase_core" in dockerfile_content
    assert "import omnibase_infra" in dockerfile_content
    assert "from omnimarket" not in dockerfile_content


@pytest.mark.unit
def test_smoke_guard_runs_after_wheel_installs(dockerfile_content: str) -> None:
    """The guard must run after the ONEX wheels are installed, not before."""
    install_idx = dockerfile_content.find("--reinstall-package omnibase-spi")
    guard_idx = dockerfile_content.find("import omnibase_compat; \\")
    assert install_idx != -1, "spi install step missing from Dockerfile"
    assert guard_idx != -1, "foundation import smoke guard missing from Dockerfile"
    assert install_idx < guard_idx, (
        "foundation import smoke guard must run AFTER the wheel install steps"
    )


@pytest.mark.unit
def test_ci_job_reasserts_foundation_import_on_pr() -> None:
    """The docker-build CI job must re-assert the foundation import on the loaded image."""
    content = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "Foundation import smoke (OMN-12658)" in content
    assert _FOUNDATION_IMPORT in content
    assert "omnimarket" not in content
    # The smoke step must run on PRs (where github.ref != main), so a skewed
    # wheel fails the required status check before merge.
    assert "github.ref != 'refs/heads/main'" in content
