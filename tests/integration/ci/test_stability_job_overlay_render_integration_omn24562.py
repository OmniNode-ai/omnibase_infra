# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability job's LLM endpoint render against the real contract and a real overlay (OMN-24562).

No mocks of the renderer: the real node contract and ``llm_endpoint`` contract are read from
this checkout, the overlay comes from a real git clone, and only the deploy agent's lane
vocabulary is stood in for by a small namespace. A stability job of a dev-fenced agent takes
the overlay's address; a prod job takes nothing. Every address is synthetic (192.0.2.0/24).
"""

from __future__ import annotations

import enum
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_setup_local_provision_effect import (
    runtime_llm_endpoint_env,
)

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
SLOT_ID = "embeddings-201"
ALLOWED_LANES_ENV = "DEPLOY_AGENT_ALLOWED_LANES"


class _Lane(enum.Enum):
    DEV = "dev"
    STABILITY_TEST = "follow"
    PROD = "prod"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_home(root: Path) -> Path:
    repo = root / "overlay-home"
    repo.mkdir()
    _git(repo, "init", "-q")
    overlay = repo / "overlays" / "llm_endpoints" / "overlay.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {SLOT_ID: OVERLAY_ADDRESS}}
        ),
        encoding="utf-8",
    )
    _git(repo, "add", str(overlay.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _execution(lane: _Lane | None) -> SimpleNamespace:
    return SimpleNamespace(
        __file__=str(REPO_ROOT / "scripts/deploy-agent/deploy_agent/executor.py"),
        ENV_ALLOWED_LANES=ALLOWED_LANES_ENV,
        LANE_OVERLAY_DIR_ENV="ONEX_LANE_OVERLAY_DIR",
        # No lane overlay directory is configured here (OMN-25202: render_dev_lane reads it).
        resolve_lane_overlay_dir=lambda environ: None,
        EnumRuntimeLane=_Lane,
        _ActiveDeployLane=SimpleNamespace(lane=lane),
        parse_allowed_lanes=lambda raw: raw,
        is_lab_dev_fence=lambda lanes: lanes == _Lane.DEV.value,
    )


def _environ(tmp_path: Path) -> dict[str, str]:
    return {
        ALLOWED_LANES_ENV: _Lane.DEV.value,
        "DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(_overlay_home(tmp_path)),
        "DEPLOY_AGENT_STATE_DIR": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
    }


def test_golden_a_stability_job_takes_the_overlay_address(tmp_path: Path) -> None:
    env, _ = runtime_llm_endpoint_env.render_dev_lane(
        _execution(_Lane.STABILITY_TEST), _environ(tmp_path)
    )

    assert env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS


def test_control_a_prod_job_takes_nothing(tmp_path: Path) -> None:
    env, lane_dir = runtime_llm_endpoint_env.render_dev_lane(
        _execution(_Lane.PROD), _environ(tmp_path)
    )

    assert env == {}
    assert lane_dir is None
