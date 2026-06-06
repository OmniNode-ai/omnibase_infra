# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deploy-runtime regression coverage for Docker build context paths."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"


@pytest.mark.unit
def test_deploy_runtime_syncs_runtime_dockerfile_copy_sources() -> None:
    """deploy-runtime.sh must ship paths copied by Dockerfile.runtime."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    required_sources = (
        "workspace/sibling-repos/",
        "scripts/runtime_build/compute_workspace_provenance.py",
    )
    for source in required_sources:
        assert source in dockerfile

    assert '"${repo_root}/workspace/sibling-repos/"' in deploy_script
    assert '"${repo_root}/scripts/runtime_build/"' in deploy_script
