# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12658: assert the runtime image carries a build-time delegation import smoke guard.

These tests pin the existence of the build-time import smoke step in
``docker/Dockerfile.runtime`` and the re-assertion step in the
``docker-build`` CI job. The guard imports the foundation packages
(``omnibase_core``, ``omnibase_infra``) after all wheels are installed, so a
skewed wheel fails the docker
build (and therefore the required ``docker-build`` status check) instead of
degrading silently as a runtime entry-point load failure.

The tests are static (no Docker daemon required) — they guarantee the guard
cannot be silently deleted without a failing unit test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_WORKFLOW_PATH = _PROJECT_ROOT / ".github" / "workflows" / "docker-build.yml"

# Foundation import asserted by the guard (OMN-12658). OMN-21279: the image
# carries no omnimarket, so the guard no longer imports a market handler.
# OMN-20074: the Dockerfile smoke no longer imports omnibase_compat (retired).
# The docker-build CI step still does: that workflow builds in release mode,
# which installs compat from a tarball, and editing the workflow needs a
# live-contact exception in the PR body, which the PR owner must carry. The
# follow-up that drops release-mode compat changes both together.
_FOUNDATION_IMPORT = (
    "import omnibase_compat; import omnibase_core; import omnibase_infra"
)
_LOCK_PATH = _PROJECT_ROOT / "uv.lock"
_SMOKE_MARKER = "foundation import smoke OK"
_FIRST_PARTY_MODULE = re.compile(r"\bomnibase_[a-z0-9_]+\b")


@pytest.mark.unit
def test_dockerfile_has_foundation_import_smoke_guard(dockerfile_content: str) -> None:
    """The runtime Dockerfile must import core + infra at build time."""
    assert "import omnibase_core" in dockerfile_content
    assert "import omnibase_infra" in dockerfile_content
    assert "from omnimarket" not in dockerfile_content


@pytest.mark.unit
def test_smoke_guard_runs_after_wheel_installs(dockerfile_content: str) -> None:
    """The guard must run after the ONEX wheels are installed, not before."""
    install_idx = dockerfile_content.find("--reinstall-package omnibase-spi")
    guard_idx = dockerfile_content.find(_SMOKE_MARKER)
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


def _locked_distributions() -> set[str]:
    """Distribution names carried by ``uv.lock``, PEP 503 normalized."""
    lock = tomllib.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    return {re.sub(r"[-_.]+", "-", pkg["name"]).lower() for pkg in lock["package"]}


def _smoke_block(text: str) -> str:
    """Return the statement that prints the foundation smoke marker.

    The statement is the paragraph holding the marker (blank-line delimited),
    minus comment lines, so prose about the smoke cannot satisfy or fail it.
    """
    marker_idx = text.index(_SMOKE_MARKER)
    start = text.rfind("\n\n", 0, marker_idx) + 1
    end = text.find("\n\n", marker_idx)
    paragraph = text[start : end if end != -1 else len(text)]
    return "\n".join(
        line for line in paragraph.splitlines() if not line.lstrip().startswith("#")
    )


def _modules_missing_from_lock(text: str, locked: set[str]) -> list[str]:
    """First-party modules the smoke names whose distribution the lock lacks."""
    block = _smoke_block(text)
    return sorted(
        module
        for module in set(_FIRST_PARTY_MODULE.findall(block))
        if module.replace("_", "-") not in locked
    )


@pytest.mark.unit
def test_dockerfile_smoke_imports_match_lock(dockerfile_content: str) -> None:
    """The foundation smoke names only packages the project's lock carries.

    The image's workspace build installs from ``uv.lock``; a smoke that imports
    a package the lock dropped fails every such build with ModuleNotFoundError.
    Covers the Dockerfile smoke, which runs in every build mode.
    """
    locked = _locked_distributions()
    # Positive controls: the lock names the packages the smoke must keep
    # importing, and the smoke block really names them (a parser that finds
    # nothing would otherwise pass vacuously).
    assert {"omnibase-core", "omnibase-infra"} <= locked
    block = _smoke_block(dockerfile_content)
    assert {"omnibase_core", "omnibase_infra"} <= set(
        _FIRST_PARTY_MODULE.findall(block)
    ), f"docker/Dockerfile.runtime: smoke block not parsed: {block!r}"
    assert _modules_missing_from_lock(dockerfile_content, locked) == [], (
        "docker/Dockerfile.runtime: foundation smoke names a package missing "
        "from uv.lock"
    )


@pytest.mark.unit
def test_dockerfile_smoke_imports_match_lock_detects_retired_package(
    dockerfile_content: str,
) -> None:
    """Negative control: putting a retired import back into the smoke is caught."""
    locked = _locked_distributions()
    assert "omnibase-compat" not in locked
    regressed = dockerfile_content.replace(
        "import omnibase_core; \\",
        "import omnibase_compat; import omnibase_core; \\",
        1,
    )
    assert regressed != dockerfile_content
    assert _modules_missing_from_lock(regressed, locked) == ["omnibase_compat"]
