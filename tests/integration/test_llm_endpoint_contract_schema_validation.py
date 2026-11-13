# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for LLM endpoint topology contract validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.contracts.model_llm_endpoint_entry import (
    ModelLlmEndpointEntry,
)
from omnibase_infra.models.contracts.model_llm_endpoint_env_file import load_endpoints
from omnibase_infra.models.contracts.model_llm_endpoint_overlay import (
    resolve_overlay_endpoints,
)

_REPO_ROOT = Path(__file__).parents[2]
_CONTRACT_PATH = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"
_GUARD = _REPO_ROOT / "scripts" / "check_llm_endpoint_env_contract.py"
_SLOT = "embeddings-201"
_OVERLAY_ADDRESS = "http://192.0.2.10:8100"
_REMOVED_ADDRESS = "http://192.0.2.10:8002"


def test_llm_endpoint_contract_entries_validate_against_typed_schema() -> None:
    payload = yaml.safe_load(_CONTRACT_PATH.read_text())
    endpoints = payload["endpoints"]

    parsed = [ModelLlmEndpointEntry.model_validate(item) for item in endpoints]

    assert parsed
    assert {entry.slot_id for entry in parsed}
    assert all(entry.role for entry in parsed)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_home(root: Path) -> Path:
    """A clone holding the llm_endpoints overlay at origin/main (OMN-24025)."""
    repo = root / "overlay-home"
    overlay = repo / "overlays" / "llm_endpoints" / "overlay.yaml"
    overlay.parent.mkdir(parents=True)
    _git(repo, "init", "-q")
    overlay.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {_SLOT: _OVERLAY_ADDRESS}}
        ),
        encoding="utf-8",
    )
    _git(repo, "add", str(overlay.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def test_real_contract_slot_resolves_from_the_overlay_clone_at_the_ref(
    tmp_path: Path,
) -> None:
    home = _overlay_home(tmp_path)
    working_tree_file = home / "overlays" / "llm_endpoints" / "overlay.yaml"
    working_tree_file.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {_SLOT: _REMOVED_ADDRESS}}
        ),
        encoding="utf-8",
    )

    resolved = resolve_overlay_endpoints(
        load_endpoints(_CONTRACT_PATH),
        {"DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home)},
    )

    slot = next(e for e in resolved if e["slot_id"] == _SLOT)
    assert slot["endpoint_url"] == _OVERLAY_ADDRESS
    assert slot["port"] == 8100


def test_real_guard_script_takes_the_rendered_environment_not_the_stale_store(
    tmp_path: Path,
) -> None:
    home = _overlay_home(tmp_path)
    store = tmp_path / "store.env"
    store.write_text(f"LLM_EMBEDDING_URL={_REMOVED_ADDRESS}\n", encoding="utf-8")
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "ONEX_SKILL_OVERLAY_ROOTS",
            "ONEX_WORKSPACE_CONFIG_ROOT",
            "LLM_EMBEDDING_URL",
        }
    }
    base_env["DEPLOY_AGENT_OVERLAY_HOME_CLONE"] = str(home)

    def run(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_GUARD), *args],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=scrub_git_location_env(env),
        )

    stale_store = run("--env-file", str(store), env=base_env)
    skipped_store = run(
        "--env-file", str(store), "--skip-env-var", "LLM_EMBEDDING_URL", env=base_env
    )
    rendered_process_env = run(env={**base_env, "LLM_EMBEDDING_URL": _OVERLAY_ADDRESS})
    stale_process_env = run(env={**base_env, "LLM_EMBEDDING_URL": _REMOVED_ADDRESS})

    assert stale_store.returncode == 1
    assert skipped_store.returncode == 0, skipped_store.stderr
    assert rendered_process_env.returncode == 0, rendered_process_env.stderr
    assert stale_process_env.returncode == 1
