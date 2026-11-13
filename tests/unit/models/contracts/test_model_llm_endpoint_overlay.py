# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the llm_endpoints overlay model and resolver (OMN-24017).

The embedding endpoint address no longer lives in this repo; it is read at
run time from the private overlays repository's ``llm_endpoints/overlay.yaml``.
All overlay files in these tests are synthetic, written under ``tmp_path``,
and use the documentation address range 192.0.2.0/24 — never a real lab
host or private address.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.contracts.model_llm_endpoint_overlay import (
    OVERLAY_HOME_REF,
    OVERLAY_ID,
    ModelLlmEndpointOverlay,
    resolve_overlay_endpoints,
)

pytestmark = pytest.mark.unit

_SYNTHETIC_ADDRESS = "http://192.0.2.10:8100"
_REMOVED_ADDRESS = "http://192.0.2.10:8002"


def _write_overlay(
    root: Path,
    endpoint_urls: dict[str, str] | None,
    *,
    schema_version: int = 1,
    extra_keys: dict[str, object] | None = None,
) -> Path:
    path = root / OVERLAY_ID / "overlay.yaml"
    path.parent.mkdir(parents=True)
    data: dict[str, object] = {"schema_version": schema_version}
    data["endpoint_urls"] = endpoint_urls if endpoint_urls is not None else {}
    if extra_keys:
        data.update(extra_keys)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return root


def _environ(*roots: Path, **extra: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if roots:
        env["ONEX_SKILL_OVERLAY_ROOTS"] = ":".join(str(r) for r in roots)
    env.update(extra)
    return env


def _flagged_slot(
    slot_id: str = "embeddings-201",
    **extra: Any,
) -> dict[str, Any]:
    slot: dict[str, Any] = {
        "slot_id": slot_id,
        "role": "embedding",
        "status": "running",
        "model_hf_id": "vendor/embed-model",
        "endpoint_from_overlay": True,
    }
    slot.update(extra)
    return slot


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )


def _make_git_repo(
    tmp_path: Path,
    overlay_urls: dict[str, str] | None = None,
) -> Path:
    """A git repo with the overlay committed on origin/main."""
    repo = tmp_path / "clone"
    repo.mkdir(parents=True)
    _git("init", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    if overlay_urls is not None:
        overlay_dir = repo / "overlays" / OVERLAY_ID
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "overlay.yaml").write_text(
            yaml.safe_dump({"schema_version": 1, "endpoint_urls": overlay_urls}),
            encoding="utf-8",
        )
        _git("add", ".", cwd=repo)
        _git("commit", "-m", "add overlay", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
    return repo


# ---------------------------------------------------------------------------
# roots / clones
# ---------------------------------------------------------------------------


def test_roots_list_the_named_overlay_roots_in_order(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    roots = ModelLlmEndpointOverlay.roots(_environ(first, second))
    assert roots == [first, second]


def test_roots_do_not_include_clones(tmp_path: Path) -> None:
    named = tmp_path / "named"
    clone = tmp_path / "deploy-agent-home"
    workspace = tmp_path / "workspace" / "config"
    roots = ModelLlmEndpointOverlay.roots(
        _environ(named, DEPLOY_AGENT_OVERLAY_HOME_CLONE=str(clone))
    )
    assert roots == [named]

    clone_roots = ModelLlmEndpointOverlay.roots(
        _environ(ONEX_WORKSPACE_CONFIG_ROOT=str(workspace))
    )
    assert clone_roots == []


def test_clones_order_and_unset(tmp_path: Path) -> None:
    clone = tmp_path / "deploy-agent-home"
    workspace = tmp_path / "workspace" / "config"

    both = ModelLlmEndpointOverlay.clones(
        _environ(
            DEPLOY_AGENT_OVERLAY_HOME_CLONE=str(clone),
            ONEX_WORKSPACE_CONFIG_ROOT=str(workspace),
        )
    )
    assert both == [clone, workspace.parent / "overlays"]

    only_clone = ModelLlmEndpointOverlay.clones(
        _environ(DEPLOY_AGENT_OVERLAY_HOME_CLONE=str(clone))
    )
    assert only_clone == [clone]

    only_workspace = ModelLlmEndpointOverlay.clones(
        _environ(ONEX_WORKSPACE_CONFIG_ROOT=str(workspace))
    )
    assert only_workspace == [workspace.parent / "overlays"]

    none = ModelLlmEndpointOverlay.clones({})
    assert none == []


# ---------------------------------------------------------------------------
# read_clone
# ---------------------------------------------------------------------------


def test_read_clone_reads_the_ref_not_the_working_tree(
    tmp_path: Path,
) -> None:
    repo = _make_git_repo(tmp_path, {"embeddings-201": _SYNTHETIC_ADDRESS})
    overlay_path = repo / "overlays" / OVERLAY_ID / "overlay.yaml"
    overlay_path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {"embeddings-201": _REMOVED_ADDRESS}}
        ),
        encoding="utf-8",
    )
    overlay = ModelLlmEndpointOverlay.read_clone(repo)
    assert overlay is not None
    assert overlay.endpoint_urls["embeddings-201"] == _SYNTHETIC_ADDRESS


def test_read_clone_returns_none_when_the_ref_holds_no_overlay(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "clone"
    repo.mkdir()
    _git("init", cwd=repo)
    _git("config", "user.name", "test", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    (repo / "placeholder.txt").write_text("x", encoding="utf-8")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "placeholder", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
    assert ModelLlmEndpointOverlay.read_clone(repo) is None


def test_read_clone_refuses_a_missing_ref(tmp_path: Path) -> None:
    repo = _make_git_repo(tmp_path, {"embeddings-201": _SYNTHETIC_ADDRESS})
    with pytest.raises(ValueError, match="holds no"):
        ModelLlmEndpointOverlay.read_clone(repo, ref="origin/nonexistent")


def test_read_clone_refuses_a_non_directory_clone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        ModelLlmEndpointOverlay.read_clone(tmp_path / "does-not-exist")


def test_read_clone_refuses_an_invalid_overlay_at_the_ref(
    tmp_path: Path,
) -> None:
    repo = _make_git_repo(tmp_path, None)
    overlay_dir = repo / "overlays" / OVERLAY_ID
    overlay_dir.mkdir(parents=True)
    (overlay_dir / "overlay.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {"embeddings-201": "not-a-url"}}
        ),
        encoding="utf-8",
    )
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "bad overlay", cwd=repo)
    _git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=repo)
    with pytest.raises(ValueError, match="invalid llm_endpoints overlay"):
        ModelLlmEndpointOverlay.read_clone(repo)


# ---------------------------------------------------------------------------
# roots / read
# ---------------------------------------------------------------------------


def test_read_returns_none_when_no_root_holds_the_file(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert ModelLlmEndpointOverlay.read(_environ(tmp_path / "empty")) is None
    assert ModelLlmEndpointOverlay.read({}) is None


def test_read_valid_overlay(tmp_path: Path) -> None:
    root = _write_overlay(tmp_path, {"embeddings-201": _SYNTHETIC_ADDRESS})
    overlay = ModelLlmEndpointOverlay.read(_environ(root))
    assert overlay is not None
    assert overlay.schema_version == 1
    assert overlay.endpoint_urls == {"embeddings-201": _SYNTHETIC_ADDRESS}


def test_read_prefers_the_first_root_holding_the_overlay(tmp_path: Path) -> None:
    first = _write_overlay(tmp_path / "a", {"embeddings-201": _SYNTHETIC_ADDRESS})
    second = _write_overlay(tmp_path / "b", {"embeddings-201": _REMOVED_ADDRESS})
    overlay = ModelLlmEndpointOverlay.read(_environ(first, second))
    assert overlay is not None
    assert overlay.endpoint_urls["embeddings-201"] == _SYNTHETIC_ADDRESS


def test_read_a_root_without_the_overlay_is_skipped(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    held = _write_overlay(tmp_path / "held", {"embeddings-201": _SYNTHETIC_ADDRESS})
    overlay = ModelLlmEndpointOverlay.read(_environ(empty, held))
    assert overlay is not None
    assert overlay.endpoint_urls["embeddings-201"] == _SYNTHETIC_ADDRESS


def test_read_uses_named_roots_before_clones(tmp_path: Path) -> None:
    root = _write_overlay(tmp_path / "root", {"embeddings-201": _SYNTHETIC_ADDRESS})
    repo = _make_git_repo(tmp_path / "repo", {"embeddings-201": _REMOVED_ADDRESS})
    overlay = ModelLlmEndpointOverlay.read(
        _environ(root, DEPLOY_AGENT_OVERLAY_HOME_CLONE=str(repo))
    )
    assert overlay is not None
    assert overlay.endpoint_urls["embeddings-201"] == _SYNTHETIC_ADDRESS


@pytest.mark.parametrize(
    "data",
    [
        {
            "schema_version": 1,
            "endpoint_urls": {"embeddings-201": _SYNTHETIC_ADDRESS},
            "unknown_key": 1,
        },
        {
            "schema_version": 2,
            "endpoint_urls": {"embeddings-201": _SYNTHETIC_ADDRESS},
        },
        {
            "schema_version": 1,
            "endpoint_urls": {"embeddings-201": _SYNTHETIC_ADDRESS + "/v1/embeddings"},
        },
        {
            "schema_version": 1,
            "endpoint_urls": {"embeddings-201": "http://192.0.2.10"},
        },
        {
            "schema_version": 1,
            "endpoint_urls": {"embeddings-201": "ftp://192.0.2.10:8100"},
        },
    ],
    ids=[
        "extra-key",
        "bad-schema-version",
        "url-with-path",
        "url-without-port",
        "non-http-scheme",
    ],
)
def test_read_refuses_an_invalid_overlay_file(
    tmp_path: Path, data: dict[str, object]
) -> None:
    path = tmp_path / OVERLAY_ID / "overlay.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        ModelLlmEndpointOverlay.read(_environ(tmp_path))


# ---------------------------------------------------------------------------
# resolve_overlay_endpoints
# ---------------------------------------------------------------------------


def test_resolve_fills_host_port_and_url_for_a_flagged_slot(
    tmp_path: Path,
) -> None:
    root = _write_overlay(tmp_path, {"embeddings-201": _SYNTHETIC_ADDRESS})
    flagged = _flagged_slot()
    unflagged: dict[str, Any] = {
        "slot_id": "coder-5090",
        "host": "192.0.2.10",
        "port": 8000,
        "endpoint_url": "http://192.0.2.10:8000",
        "status": "running",
    }
    resolved = resolve_overlay_endpoints([flagged, unflagged], _environ(root))

    filled = resolved[0]
    assert filled["host"] == "192.0.2.10"
    assert filled["port"] == 8100
    assert filled["endpoint_url"] == _SYNTHETIC_ADDRESS
    assert filled["endpoint_from_overlay"] is True

    untouched = resolved[1]
    assert untouched == unflagged
    assert "endpoint_from_overlay" not in untouched


def test_resolve_returns_copies_not_the_input_dicts(tmp_path: Path) -> None:
    root = _write_overlay(tmp_path, {"embeddings-201": _SYNTHETIC_ADDRESS})
    flagged = _flagged_slot()
    resolved = resolve_overlay_endpoints([flagged], _environ(root))
    assert resolved[0] is not flagged
    assert flagged.get("host") is None, "input dict must not be mutated"


def test_resolve_raises_naming_the_slot_when_the_overlay_is_absent(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError) as excinfo:
        resolve_overlay_endpoints([_flagged_slot()], _environ(tmp_path / "empty"))
    message = str(excinfo.value)
    assert "'embeddings-201'" in message
    assert "no llm_endpoints overlay file" in message
    assert str(tmp_path / "empty") in message


def test_resolve_raises_naming_the_slot_when_it_is_missing_from_the_overlay(
    tmp_path: Path,
) -> None:
    root = _write_overlay(tmp_path, {"other-slot": _SYNTHETIC_ADDRESS})
    with pytest.raises(ValueError) as excinfo:
        resolve_overlay_endpoints([_flagged_slot()], _environ(root))
    message = str(excinfo.value)
    assert "'embeddings-201'" in message
    assert "not in the llm_endpoints overlay" in message


def test_resolve_strips_a_trailing_slash_from_the_overlay_url(
    tmp_path: Path,
) -> None:
    root = _write_overlay(tmp_path, {"embeddings-201": "http://192.0.2.10:8100/"})
    resolved = resolve_overlay_endpoints([_flagged_slot()], _environ(root))
    assert resolved[0]["endpoint_url"] == _SYNTHETIC_ADDRESS
    assert resolved[0]["port"] == 8100


def test_resolve_leaves_a_slot_unresolved_when_the_store_leaves_its_variable_empty(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty").mkdir()
    slot = _flagged_slot(url_env_var="LLM_EXAMPLE_URL")

    resolved = resolve_overlay_endpoints(
        [slot], _environ(tmp_path / "empty"), {"LLM_EXAMPLE_URL": " "}
    )

    assert resolved == [slot]
    assert "endpoint_url" not in resolved[0]


def test_resolve_refuses_when_the_store_names_a_value_and_no_overlay_holds_the_slot(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty").mkdir()
    slot = _flagged_slot(url_env_var="LLM_EXAMPLE_URL")

    with pytest.raises(ValueError, match="'embeddings-201'"):
        resolve_overlay_endpoints(
            [slot],
            _environ(tmp_path / "empty"),
            {"LLM_EXAMPLE_URL": "http://192.0.2.10:9"},
        )


# ---------------------------------------------------------------------------
# SEAM: the ref must equal the node contract's ref
# ---------------------------------------------------------------------------


def test_overlay_home_ref_equals_the_node_contract_ref() -> None:
    contract_path = (
        Path(__file__).resolve().parents[4]
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    overlay_home = contract["config"]["overlay_home_lane_env"]
    assert overlay_home["ref"] == OVERLAY_HOME_REF
    assert overlay_home["overlay_path"] == "overlays/runtime_lane_env/overlay.yaml"
