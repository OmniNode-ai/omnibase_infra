# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's embedding address rendered from the llm_endpoints overlay (OMN-24025).

The deploy agent builds the dev lane's compose environment from the operator
store. The embedding address is the one value that must not come from there: it is
the ``llm_endpoints`` overlay's ``embeddings-201`` entry, read from the overlay home's
clone at ``origin/main``, and the lane bifrost overlay's bound backend follows it.

The contract read here is the repository's real ``contracts/llm_endpoints.yaml`` and the
node contract's real declaration; every address is synthetic (192.0.2.0/24).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

LlmEndpointEnv = executor_mod.load_dev_boot_guard(
    "nodes.node_setup_local_provision_effect.runtime_llm_endpoint_env"
)

SLOT_ID = "embeddings-201"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
STALE_ADDRESS = "http://192.0.2.50:8002"
BACKEND_ID = "local-embedding"
EMBEDDING_PATH = "/v1/embeddings"
REPO_ROOT = Path(__file__).resolve().parents[4]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_home(root: Path, endpoint_urls: dict[str, str] | None) -> Path:
    """A git clone holding the overlay at ``origin/main`` (none when ``endpoint_urls`` is None)."""
    repo = root / "overlay-home"
    repo.mkdir()
    _git(repo, "init", "-q")
    if endpoint_urls is None:
        (repo / "README.md").write_text("empty\n", encoding="utf-8")
        _git(repo, "add", "README.md")
    else:
        overlay = repo / "overlays" / "llm_endpoints" / "overlay.yaml"
        overlay.parent.mkdir(parents=True)
        overlay.write_text(
            yaml.safe_dump({"schema_version": 1, "endpoint_urls": endpoint_urls}),
            encoding="utf-8",
        )
        _git(repo, "add", str(overlay.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _lane_overlays(root: Path, embedding_url: str | None = None) -> Path:
    lane_dir = root / "lane-overlays"
    lane_dir.mkdir()
    backends: list[dict[str, Any]] = [
        {
            "backend_id": "local-coder",
            "endpoint_url": "http://192.0.2.10:8000/v1/chat/completions",
        },
        {
            "backend_id": BACKEND_ID,
            "endpoint_url": embedding_url or f"{STALE_ADDRESS}{EMBEDDING_PATH}",
        },
    ]
    (lane_dir / "dev.bifrost.yaml").write_text(
        yaml.safe_dump({"lane": "dev", "backends": backends}, sort_keys=False),
        encoding="utf-8",
    )
    (lane_dir / "unrelated.yaml").write_text(
        "# kept byte for byte\nkey: value\n", encoding="utf-8"
    )
    return lane_dir


def _environ(root: Path, home: Path) -> dict[str, str]:
    state = root / "state"
    state.mkdir()
    return {
        "DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home),
        "DEPLOY_AGENT_STATE_DIR": str(state),
    }


def _render(environ: dict[str, str], lane_dir: Path | None) -> Any:
    return LlmEndpointEnv.render(
        executor_mod,
        environ,
        lane_overlay_dir=None if lane_dir is None else str(lane_dir),
    )


def _backend_url(directory: Path, name: str = "dev.bifrost.yaml") -> dict[str, str]:
    document = yaml.safe_load((directory / name).read_text(encoding="utf-8"))
    return {b["backend_id"]: b["endpoint_url"] for b in document["backends"]}


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Any]:
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    lane_dir = _lane_overlays(tmp_path)
    return {"environ": _environ(tmp_path, home), "lane_dir": lane_dir, "home": home}


def test_golden_renders_env_and_rewrites_only_the_bound_backend(
    world: dict[str, Any],
) -> None:
    rendered = _render(world["environ"], world["lane_dir"])

    expected_url = f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"
    assert rendered.env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    assert rendered.env["BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"] == expected_url
    target = Path(rendered.lane_overlay_dir)
    assert (
        target
        == Path(world["environ"]["DEPLOY_AGENT_STATE_DIR"]) / "rendered-lane-overlays"
    )
    urls = _backend_url(target)
    assert urls[BACKEND_ID] == expected_url
    assert urls["local-coder"] == "http://192.0.2.10:8000/v1/chat/completions"
    assert (target / "unrelated.yaml").read_bytes() == (
        world["lane_dir"] / "unrelated.yaml"
    ).read_bytes()
    assert STALE_ADDRESS in (world["lane_dir"] / "dev.bifrost.yaml").read_text(
        encoding="utf-8"
    )


def test_golden_without_a_lane_overlay_dir_renders_only_the_slot_env(
    world: dict[str, Any],
) -> None:
    rendered = _render(world["environ"], None)

    assert rendered.env == {"LLM_EMBEDDING_URL": OVERLAY_ADDRESS}
    assert rendered.lane_overlay_dir is None


def test_a_symlinked_lane_overlay_is_rendered_as_the_file_it_points_at(
    world: dict[str, Any],
) -> None:
    """The dev project mounts ``<project>.bifrost.yaml``, a link to the lane's own file."""
    link = world["lane_dir"] / "project.bifrost.yaml"
    link.symlink_to("dev.bifrost.yaml")

    rendered = _render(world["environ"], world["lane_dir"])

    target = Path(rendered.lane_overlay_dir)
    expected_url = f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"
    assert _backend_url(target, "project.bifrost.yaml")[BACKEND_ID] == expected_url
    assert _backend_url(target)[BACKEND_ID] == expected_url
    assert rendered.env["BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"] == expected_url


def test_seam_the_rendered_lane_overlay_loads_in_the_bifrost_overlay_model(
    world: dict[str, Any],
) -> None:
    """The container's renderer loads this file with ``ModelBifrostLaneOverlay``: so must a test."""
    from omnibase_infra.runtime.render_bifrost_delegation_contract import (
        _load_lane_overlay,
    )

    def backend(backend_id: str, url: str) -> dict[str, Any]:
        return {
            "backend_id": backend_id,
            "endpoint_url": url,
            "served_model_id": "model-a",
            "parameter_count": "1B",
            "context_window": 8192,
            "max_tokens": 128,
            "timeout_ms": 1000,
        }

    (world["lane_dir"] / "dev.bifrost.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": [
                    backend(
                        "local-coder", "http://192.0.2.10:8000/v1/chat/completions"
                    ),
                    backend(BACKEND_ID, f"{STALE_ADDRESS}{EMBEDDING_PATH}"),
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    rendered = Path(_render(world["environ"], world["lane_dir"]).lane_overlay_dir)

    before = _load_lane_overlay(world["lane_dir"] / "dev.bifrost.yaml")
    after = _load_lane_overlay(rendered / "dev.bifrost.yaml")
    changed = [
        b.backend_key
        for b, a in zip(before.backends, after.backends, strict=True)
        if b != a
    ]
    assert changed == [BACKEND_ID]
    assert after.backends[1].endpoint_url == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"


def test_a_second_render_leaves_the_rendered_files_untouched(
    world: dict[str, Any],
) -> None:
    first = Path(_render(world["environ"], world["lane_dir"]).lane_overlay_dir)
    stamp = {p.name: p.stat().st_mtime_ns for p in first.iterdir()}
    os.utime(first / "dev.bifrost.yaml", ns=(1, 1))
    stamp["dev.bifrost.yaml"] = 1

    second = Path(_render(world["environ"], world["lane_dir"]).lane_overlay_dir)

    assert {p.name: p.stat().st_mtime_ns for p in second.iterdir()} == stamp


def test_a_file_the_source_no_longer_has_is_removed_from_the_rendered_copy(
    world: dict[str, Any],
) -> None:
    target = Path(_render(world["environ"], world["lane_dir"]).lane_overlay_dir)
    (target / "stale.yaml").write_text("old: 1\n", encoding="utf-8")

    _render(world["environ"], world["lane_dir"])

    assert not (target / "stale.yaml").exists()
    assert (target / "dev.bifrost.yaml").is_file()


def test_the_overlay_is_read_at_the_ref_not_the_working_tree(
    world: dict[str, Any],
) -> None:
    overlay = world["home"] / "overlays" / "llm_endpoints" / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {SLOT_ID: STALE_ADDRESS}}
        ),
        encoding="utf-8",
    )

    rendered = _render(world["environ"], None)

    assert rendered.env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS


def test_error_a_clone_without_the_overlay_refuses(tmp_path: Path) -> None:
    home = _overlay_home(tmp_path, None)
    with pytest.raises(LlmEndpointEnvError(), match="no llm_endpoints overlay"):
        _render(_environ(tmp_path, home), None)


def test_error_no_overlay_home_named_refuses(tmp_path: Path) -> None:
    with pytest.raises(LlmEndpointEnvError(), match="no llm_endpoints overlay"):
        _render({"DEPLOY_AGENT_STATE_DIR": str(tmp_path)}, None)


def test_error_a_clone_that_holds_no_ref_refuses(tmp_path: Path) -> None:
    home = tmp_path / "overlay-home"
    home.mkdir()
    _git(home, "init", "-q")
    with pytest.raises(LlmEndpointEnvError(), match="holds no origin/main"):
        _render({"DEPLOY_AGENT_OVERLAY_HOME_CLONE": str(home)}, None)


def test_error_a_slot_the_overlay_omits_refuses_by_name(tmp_path: Path) -> None:
    home = _overlay_home(tmp_path, {"some-other-slot": OVERLAY_ADDRESS})
    with pytest.raises(LlmEndpointEnvError(), match=SLOT_ID):
        _render(_environ(tmp_path, home), None)


def test_error_an_unset_state_dir_refuses_by_name(world: dict[str, Any]) -> None:
    environ = {
        k: v for k, v in world["environ"].items() if k != "DEPLOY_AGENT_STATE_DIR"
    }
    with pytest.raises(LlmEndpointEnvError(), match="DEPLOY_AGENT_STATE_DIR"):
        _render(environ, world["lane_dir"])


def test_error_an_unparseable_bound_endpoint_refuses(tmp_path: Path) -> None:
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    lane_dir = _lane_overlays(tmp_path, embedding_url="not-a-url")
    with pytest.raises(LlmEndpointEnvError(), match="not-a-url"):
        _render(_environ(tmp_path, home), lane_dir)


def test_error_two_lane_files_that_bind_different_paths_refuse(
    world: dict[str, Any],
) -> None:
    second = world["lane_dir"] / "second.bifrost.yaml"
    second.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {"backend_id": BACKEND_ID, "endpoint_url": f"{STALE_ADDRESS}/other"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LlmEndpointEnvError(), match="differing endpoint URLs"):
        _render(world["environ"], world["lane_dir"])


def test_seam_the_real_contract_binds_the_slot_the_overlay_names() -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts" / "llm_endpoints.yaml").read_text(encoding="utf-8")
    )
    slots = {
        e["slot_id"]: e for e in contract["endpoints"] if e.get("endpoint_from_overlay")
    }
    declared = LlmEndpointEnv.declaration(executor_mod)

    assert set(slots) == {SLOT_ID}
    slot = slots[SLOT_ID]
    assert slot["url_env_var"] == "LLM_EMBEDDING_URL"
    assert slot["lane_overlay_binding"] == {
        "backend": BACKEND_ID,
        "env_var": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
    }
    assert slot["endpoint_url"] is None
    assert set(declared) >= {
        "contract_path",
        "state_dir_env",
        "rendered_dir_name",
        "lane_overlay_glob",
    }


def LlmEndpointEnvError() -> type[Exception]:
    error: type[Exception] = LlmEndpointEnv.LlmEndpointEnvError
    return error
