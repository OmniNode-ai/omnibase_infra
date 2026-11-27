# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's compose environment and deploy guard take the embedding address from the overlay (OMN-24025).

The operator store keeps whatever a person last typed. On the dev host it still names the
removed embedding server; a dev deploy must not read it, and no hand edit of the store may
be the way to move the address. These tests build the dev-fenced agent the way the
exemplar ``test_lane_env_overlay_home_omn22478`` does and read what compose and the guard
actually receive. Every address is synthetic (192.0.2.0/24).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.executor import DeployExecutor

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = [pytest.mark.unit, pytest.mark.llm_endpoint_env_render]

REPO_ROOT = Path(__file__).resolve().parents[4]
STALE_ADDRESS = "http://192.0.2.50:8002"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
# the coder slot is also overlay-addressed (OMN-26709), so every overlay names it
CODER_SLOT_ID = "coder-5090"
CODER_ADDRESS = "http://192.0.2.50:8000"
EMBEDDING_PATH = "/v1/embeddings"
SLOT_ID = "embeddings-201"
ENV_STATE_DIR = "DEPLOY_AGENT_STATE_DIR"
ENV_OVERLAY_HOME_CLONE = "DEPLOY_AGENT_OVERLAY_HOME_CLONE"
LANE_DECLARATION_FILE = "ci_bus_lanes.yaml"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _overlay_home(root: Path, endpoint_urls: dict[str, str]) -> Path:
    repo = root / "overlay-home"
    repo.mkdir()
    _git(repo, "init", "-q")
    overlay = repo / "overlays" / "llm_endpoints" / "overlay.yaml"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "endpoint_urls": {CODER_SLOT_ID: CODER_ADDRESS, **endpoint_urls},
            }
        ),
        encoding="utf-8",
    )
    _git(repo, "add", str(overlay.relative_to(repo)))
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _lane_overlays(root: Path) -> Path:
    lane_dir = root / "lane-overlays"
    lane_dir.mkdir()
    (lane_dir / "dev.bifrost.yaml").write_text(
        yaml.safe_dump(
            {
                "lane": "dev",
                "backends": [
                    {
                        "backend_id": "local-embedding",
                        "endpoint_url": f"{STALE_ADDRESS}{EMBEDDING_PATH}",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return lane_dir


@pytest.fixture
def dev_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A dev-fenced agent whose operator store still names the removed address."""
    lane_dir = _lane_overlays(tmp_path)
    # The folder above the lane overlays is the lane declaration folder (config/lab).
    (tmp_path / LANE_DECLARATION_FILE).write_text("lanes: {}\n", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(executor_mod, "deploy_repo_dir", lambda: str(REPO_ROOT))
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(lane_dir))
    # A shell that sourced the refresh holds the declaration folder; these tests own it.
    monkeypatch.delenv("ONEX_LANE_DECLARATION_DIR", raising=False)
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev")
    monkeypatch.setenv(ENV_STATE_DIR, str(state))
    monkeypatch.setenv("LLM_EMBEDDING_URL", STALE_ADDRESS)
    monkeypatch.delenv("BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL", raising=False)
    # Only the overlay home these tests build may hold the overlay; the host's own roots and
    # workspace clone must not.
    monkeypatch.delenv("ONEX_SKILL_OVERLAY_ROOTS", raising=False)
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.setattr(executor_mod._LaneOverlayClone, "path", None)
    monkeypatch.setattr(executor_mod, "operator_store_overlay", dict)
    monkeypatch.setattr(executor_mod, "_load_runtime_policy_env", dict)
    home = _overlay_home(tmp_path, {SLOT_ID: OVERLAY_ADDRESS})
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(home))
    return {"lane_dir": lane_dir, "state": state, "home": home, "tmp": tmp_path}


def _completed(returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=""
    )


def test_golden_compose_env_takes_the_address_from_the_overlay(
    dev_agent: dict[str, Any],
) -> None:
    env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    assert (
        env["BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"]
        == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"
    )
    rendered_dir = Path(env["ONEX_LANE_OVERLAY_DIR"])
    assert rendered_dir == dev_agent["state"] / "rendered-lane-overlays"
    bound = yaml.safe_load(
        (rendered_dir / "dev.bifrost.yaml").read_text(encoding="utf-8")
    )
    assert bound["backends"][0]["endpoint_url"] == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"


def test_golden_compose_env_keeps_the_lane_declaration_dir_on_the_real_folder(
    dev_agent: dict[str, Any],
) -> None:
    """OMN-24854: the render repoints ONEX_LANE_OVERLAY_DIR, not the declaration dir."""
    env = executor_mod._compose_env()

    assert env["ONEX_LANE_DECLARATION_DIR"] == str(
        dev_agent["lane_dir"].resolve().parent
    )
    rendered_dir = env["ONEX_LANE_OVERLAY_DIR"]
    assert rendered_dir == str(dev_agent["state"] / "rendered-lane-overlays")
    assert rendered_dir != env["ONEX_LANE_DECLARATION_DIR"]


def _render_dev_lane(
    environ: dict[str, str],
) -> tuple[dict[str, str], str | None]:
    guard = executor_mod.load_dev_boot_guard(executor_mod._LLM_ENV_MODULE)
    rendered: tuple[dict[str, str], str | None] = guard.render_dev_lane(
        executor_mod, environ
    )
    return rendered


def _refresh_exports(environ: dict[str, str], state: Path) -> str:
    guard = executor_mod.load_dev_boot_guard(executor_mod._LLM_ENV_MODULE)
    exports: str = guard.refresh_exports(
        executor_mod,
        environ,
        state / "no-clones",
        state,
        mounts_lane_declaration=True,
    )
    return exports


def _stability_refresh_exports(environ: dict[str, str], state: Path) -> str:
    # OMN-26121: refresh_stability_lane.sh passes with-declaration, the stability runtime mounts the folder.
    guard = executor_mod.load_dev_boot_guard(executor_mod._LLM_ENV_MODULE)
    exports: str = guard.refresh_exports(
        executor_mod,
        environ,
        state / "no-clones",
        state,
        mounts_lane_declaration=True,
    )
    return exports


def _shell_after(exports: str, *names: str) -> dict[str, str]:
    """What a shell that sourced ``exports`` holds for ``names``."""
    script = exports + "".join(f"printf '%s\\n' \"${name}\"\n" for name in names)
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )
    return dict(zip(names, result.stdout.splitlines(), strict=True))


@pytest.mark.parametrize("unset_form", ["absent", "empty", "blank"])
def test_control_render_dev_lane_with_no_lane_overlay_dir_sets_no_declaration_dir(
    dev_agent: dict[str, Any], unset_form: str
) -> None:
    environ = dict(os.environ)
    environ.pop("ONEX_LANE_OVERLAY_DIR")
    if unset_form != "absent":
        environ["ONEX_LANE_OVERLAY_DIR"] = "" if unset_form == "empty" else "  "

    env, rendered_dir = _render_dev_lane(environ)

    assert env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    assert "ONEX_LANE_DECLARATION_DIR" not in env
    assert rendered_dir is None


def test_golden_render_dev_lane_sets_the_declaration_dir_from_the_real_overlay_dir(
    dev_agent: dict[str, Any],
) -> None:
    env, rendered_dir = _render_dev_lane(dict(os.environ))

    assert env["ONEX_LANE_DECLARATION_DIR"] == str(dev_agent["tmp"].resolve())
    assert rendered_dir == str(dev_agent["state"] / "rendered-lane-overlays")


def test_golden_render_dev_lane_and_refresh_exports_agree_on_a_symlinked_overlay_dir(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-24854: one derivation, so a symlinked overlay directory reads the same twice."""
    elsewhere = dev_agent["tmp"] / "elsewhere"
    elsewhere.mkdir()
    link = elsewhere / "link-to-lane-overlays"
    link.symlink_to(dev_agent["lane_dir"], target_is_directory=True)
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(link))

    env, _ = _render_dev_lane(dict(os.environ))
    exported = _shell_after(
        _refresh_exports(dict(os.environ), dev_agent["state"]),
        "ONEX_LANE_DECLARATION_DIR",
    )

    expected = str(dev_agent["lane_dir"].resolve().parent)
    assert env["ONEX_LANE_DECLARATION_DIR"] == expected
    assert exported["ONEX_LANE_DECLARATION_DIR"] == expected
    assert expected != str(elsewhere)


def test_golden_render_dev_lane_and_refresh_exports_return_an_absolute_path_for_a_relative_overlay_dir(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(dev_agent["tmp"])
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", "lane-overlays")

    env, _ = _render_dev_lane(dict(os.environ))
    exported = _shell_after(
        _refresh_exports(dict(os.environ), dev_agent["state"]),
        "ONEX_LANE_DECLARATION_DIR",
    )

    expected = str(dev_agent["tmp"].resolve())
    assert env["ONEX_LANE_DECLARATION_DIR"] == expected
    assert exported["ONEX_LANE_DECLARATION_DIR"] == expected
    assert Path(expected).is_absolute()


def test_golden_two_refresh_passes_in_one_shell_keep_the_real_declaration_dir(
    dev_agent: dict[str, Any],
) -> None:
    """OMN-24854: pass two reads the rendered overlay dir pass one exported, not the real one."""
    names = ("ONEX_LANE_OVERLAY_DIR", "ONEX_LANE_DECLARATION_DIR")
    real = str(dev_agent["tmp"].resolve())
    rendered = str(dev_agent["state"] / "rendered-lane-overlays")
    environ = dict(os.environ)

    first = _shell_after(_refresh_exports(environ, dev_agent["state"]), *names)
    assert first["ONEX_LANE_DECLARATION_DIR"] == real
    assert first["ONEX_LANE_OVERLAY_DIR"] == rendered

    # The shell that applied pass one now holds the rendered directory.
    environ.update(first)
    second = _shell_after(_refresh_exports(environ, dev_agent["state"]), *names)

    assert second["ONEX_LANE_DECLARATION_DIR"] == real
    assert second["ONEX_LANE_DECLARATION_DIR"] != str(dev_agent["state"])


def test_golden_an_already_set_declaration_dir_is_kept(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = dev_agent["tmp"] / "pinned-config-lab"
    pinned.mkdir()
    (pinned / LANE_DECLARATION_FILE).write_text("lanes: {}\n", encoding="utf-8")
    monkeypatch.setenv("ONEX_LANE_DECLARATION_DIR", str(pinned))

    env, _ = _render_dev_lane(dict(os.environ))

    assert env["ONEX_LANE_DECLARATION_DIR"] == str(pinned.resolve())


def test_error_a_declaration_dir_that_names_no_directory_is_refused(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "ONEX_LANE_DECLARATION_DIR", str(dev_agent["tmp"] / "does-not-exist")
    )

    with pytest.raises(RuntimeError, match="ONEX_LANE_DECLARATION_DIR names no"):
        _render_dev_lane(dict(os.environ))


def test_error_a_stale_ambient_declaration_dir_without_the_declaration_is_refused(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-24854: an existing directory is no lane declaration folder unless it holds the file."""
    stale = dev_agent["tmp"] / "stale-ambient-dir"
    stale.mkdir()
    monkeypatch.setenv("ONEX_LANE_DECLARATION_DIR", str(stale))

    with pytest.raises(RuntimeError, match=f"no {LANE_DECLARATION_FILE}"):
        _render_dev_lane(dict(os.environ))
    with pytest.raises(RuntimeError, match=f"no {LANE_DECLARATION_FILE}"):
        _refresh_exports(dict(os.environ), dev_agent["state"])


def test_error_a_shell_holding_only_the_rendered_overlay_dir_is_refused(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-24854: a refresh from before this change exported only the rendered overlay dir.

    Its parent is the agent's state directory, which holds no lane declaration, so deriving
    the folder from it would mount the state directory at the runtime's config/lab.
    """
    names = ("ONEX_LANE_OVERLAY_DIR",)
    first = _shell_after(_refresh_exports(dict(os.environ), dev_agent["state"]), *names)
    rendered = Path(first["ONEX_LANE_OVERLAY_DIR"])
    assert rendered.is_dir()
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(rendered))

    with pytest.raises(RuntimeError, match=f"no {LANE_DECLARATION_FILE}"):
        _render_dev_lane(dict(os.environ))
    with pytest.raises(RuntimeError, match=f"no {LANE_DECLARATION_FILE}"):
        _refresh_exports(dict(os.environ), dev_agent["state"])


def test_control_the_derived_declaration_dir_passes_the_same_check_with_the_file_present(
    dev_agent: dict[str, Any],
) -> None:
    (dev_agent["tmp"] / LANE_DECLARATION_FILE).unlink()

    with pytest.raises(RuntimeError, match=f"no {LANE_DECLARATION_FILE}"):
        _render_dev_lane(dict(os.environ))

    (dev_agent["tmp"] / LANE_DECLARATION_FILE).write_text(
        "lanes: {}\n", encoding="utf-8"
    )
    env, _ = _render_dev_lane(dict(os.environ))
    assert env["ONEX_LANE_DECLARATION_DIR"] == str(dev_agent["tmp"].resolve())


@pytest.mark.parametrize("entry", ["render_dev_lane", "refresh_exports"])
def test_error_a_mistyped_overlay_dir_is_refused_not_rendered_without_the_lane_overlay(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(dev_agent["tmp"] / "lane-overlay"))

    with pytest.raises(RuntimeError, match="ONEX_LANE_OVERLAY_DIR names no directory"):
        if entry == "render_dev_lane":
            _render_dev_lane(dict(os.environ))
        else:
            _refresh_exports(dict(os.environ), dev_agent["state"])
    assert not (dev_agent["state"] / "rendered-lane-overlays").exists()


def test_control_an_agent_not_fenced_to_dev_keeps_the_stores_address(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", executor_mod.EnumRuntimeLane.STABILITY_TEST.value
    )

    env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == STALE_ADDRESS
    assert env["ONEX_LANE_OVERLAY_DIR"] == str(dev_agent["lane_dir"].resolve())
    assert "ONEX_LANE_DECLARATION_DIR" not in env
    assert "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL" not in env


def test_control_an_agent_not_fenced_to_dev_renders_nothing_over_a_stale_declaration_dir(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", executor_mod.EnumRuntimeLane.STABILITY_TEST.value
    )
    monkeypatch.setenv("ONEX_LANE_DECLARATION_DIR", "/stale/ambient/config/lab")

    env, rendered_dir = _render_dev_lane(dict(os.environ))

    assert (env, rendered_dir) == ({}, None)


def test_golden_a_stability_job_of_a_dev_agent_takes_the_address_from_the_overlay(
    dev_agent: dict[str, Any],
) -> None:
    """OMN-24562: the stability follow job failed the guard on the store's removed :8002."""
    with executor_mod.active_deploy_lane(executor_mod.EnumRuntimeLane.STABILITY_TEST):
        env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    assert (
        env["BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL"]
        == f"{OVERLAY_ADDRESS}{EMBEDDING_PATH}"
    )


def test_control_a_prod_job_of_a_dev_agent_keeps_the_stores_address(
    dev_agent: dict[str, Any],
) -> None:
    with executor_mod.active_deploy_lane(executor_mod.EnumRuntimeLane.PROD):
        env = executor_mod._compose_env()

    assert env["LLM_EMBEDDING_URL"] == STALE_ADDRESS


def test_error_a_dev_deploy_whose_overlay_cannot_be_read_is_refused(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_OVERLAY_HOME_CLONE)

    with pytest.raises(RuntimeError, match="llm_endpoints overlay"):
        executor_mod._compose_env()


def test_error_a_dev_deploy_whose_overlay_omits_the_slot_is_refused_by_name(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = dev_agent["tmp"] / "other"
    other.mkdir()
    home = _overlay_home(other, {"some-other-slot": OVERLAY_ADDRESS})
    monkeypatch.setenv(ENV_OVERLAY_HOME_CLONE, str(home))

    with pytest.raises(RuntimeError, match=SLOT_ID):
        executor_mod._compose_env()


def _captured_guard_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], dict[str, str] | None]]:
    seen: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(
        cmd: list[str],
        timeout: int,
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        seen.append((cmd, env))
        return _completed()

    monkeypatch.setattr(executor_mod, "_run", fake_run)
    return seen


def test_golden_guard_checks_the_rendered_environment_and_skips_the_stores_value(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = dev_agent["tmp"] / "store.env"
    store.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    monkeypatch.setenv("OMNIBASE_ENV_FILE", str(store))
    seen = _captured_guard_commands(monkeypatch)

    DeployExecutor().validate_llm_endpoint_env_contract()

    process_cmd, process_env = seen[0]
    assert "--env-file" not in process_cmd
    assert process_env is not None
    assert process_env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    file_commands = [cmd for cmd, _ in seen if "--env-file" in cmd]
    assert file_commands
    for cmd in file_commands:
        skipped = {cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--skip-env-var"}
        assert {"LLM_CODER_URL", "LLM_EMBEDDING_URL"} <= skipped


def test_golden_guard_of_a_stability_job_checks_the_rendered_environment(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-24562: the guard of a stability follow job reads the render, not the store."""
    store = dev_agent["tmp"] / "store.env"
    store.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    monkeypatch.setenv("OMNIBASE_ENV_FILE", str(store))
    seen = _captured_guard_commands(monkeypatch)

    with executor_mod.active_deploy_lane(executor_mod.EnumRuntimeLane.STABILITY_TEST):
        DeployExecutor().validate_llm_endpoint_env_contract()

    process_cmd, process_env = seen[0]
    assert "--env-file" not in process_cmd
    assert process_env is not None
    assert process_env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    for cmd, _ in seen:
        if "--env-file" in cmd:
            skipped = {
                cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--skip-env-var"
            }
            assert {"LLM_CODER_URL", "LLM_EMBEDDING_URL"} <= skipped


def test_control_guard_outside_the_dev_fence_skips_nothing(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "DEPLOY_AGENT_ALLOWED_LANES", executor_mod.EnumRuntimeLane.STABILITY_TEST.value
    )
    store = dev_agent["tmp"] / "store.env"
    store.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    monkeypatch.setenv("OMNIBASE_ENV_FILE", str(store))
    seen = _captured_guard_commands(monkeypatch)

    DeployExecutor().validate_llm_endpoint_env_contract()

    assert all("--skip-env-var" not in cmd for cmd, _ in seen)


def test_seam_every_name_the_guard_skips_is_a_contract_slot_env_var(
    dev_agent: dict[str, Any],
) -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts" / "llm_endpoints.yaml").read_text(encoding="utf-8")
    )
    slot_env_vars = {
        e["url_env_var"] for e in contract["endpoints"] if e.get("url_env_var")
    }

    skipped, _ = executor_mod._llm_endpoint_env_overlay(
        {k: v for k, v in os.environ.items() if k != "ONEX_LANE_OVERLAY_DIR"}
    )

    assert skipped
    assert set(skipped) <= slot_env_vars


def test_control_the_workspace_config_root_is_a_second_source_of_the_overlay_dir(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-25202: the no-overlay control above must drop this root too, so prove it is read.

    With ``ONEX_LANE_OVERLAY_DIR`` gone, the agent resolves ``config/lab/lane-overlays`` under
    ``ONEX_WORKSPACE_CONFIG_ROOT``; the ``dev_agent`` fixture deletes that variable so the
    control reads no ambient host config.
    """
    root = dev_agent["tmp"] / "workspace-config"
    folder = root / "config" / "lab"
    (folder / "lane-overlays").mkdir(parents=True)
    (folder / LANE_DECLARATION_FILE).write_text("lanes: {}\n", encoding="utf-8")
    monkeypatch.delenv("ONEX_LANE_OVERLAY_DIR")
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(root))

    env, _ = _render_dev_lane(dict(os.environ))

    assert env["ONEX_LANE_DECLARATION_DIR"] == str(folder.resolve())


def test_golden_a_stability_job_of_a_dev_agent_takes_the_real_declaration_dir(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-26121: the stability runtime mounts the lane declaration folder, so its job holds the real one."""
    with executor_mod.active_deploy_lane(executor_mod.EnumRuntimeLane.STABILITY_TEST):
        env, _ = _render_dev_lane(dict(os.environ))

    assert env["LLM_EMBEDDING_URL"] == OVERLAY_ADDRESS
    assert env["ONEX_LANE_DECLARATION_DIR"] == str(dev_agent["tmp"].resolve())


def test_error_a_stability_job_with_a_stale_declaration_dir_is_refused(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-26121: a folder with no ci_bus_lanes.yaml is never mounted into the stability runtime."""
    monkeypatch.setenv("ONEX_LANE_DECLARATION_DIR", "/stale/ambient/config/lab")

    with executor_mod.active_deploy_lane(executor_mod.EnumRuntimeLane.STABILITY_TEST):
        with pytest.raises(RuntimeError, match="ONEX_LANE_DECLARATION_DIR names no"):
            _render_dev_lane(dict(os.environ))


def test_golden_the_stability_refresh_exports_the_declaration_dir_and_refuses_a_stale_one(
    dev_agent: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = _stability_refresh_exports(dict(os.environ), dev_agent["state"])

    assert f"export LLM_EMBEDDING_URL={OVERLAY_ADDRESS}\n" in exports
    assert _shell_after(exports, "ONEX_LANE_DECLARATION_DIR")[
        "ONEX_LANE_DECLARATION_DIR"
    ] == str(dev_agent["tmp"].resolve())
    (dev_agent["tmp"] / LANE_DECLARATION_FILE).unlink()
    monkeypatch.setenv("ONEX_LANE_DECLARATION_DIR", "/stale/ambient/config/lab")
    with pytest.raises(RuntimeError, match="ONEX_LANE_DECLARATION_DIR names no"):
        _stability_refresh_exports(dict(os.environ), dev_agent["state"])


def test_seam_only_the_lanes_that_mount_the_declaration_take_the_declaration_dir(
    dev_agent: dict[str, Any],
) -> None:
    """OMN-25202, OMN-26121: every job whose compose mounts the folder gets it from the agent.

    A job's compose environment is ``_compose_env`` under that job's active lane. The dev
    lane's compose files include the dev lane file or the tenant fold file, and the stability
    compose carries its own mount (OMN-26121); those lanes' jobs hold the variable. The other
    lanes mount nothing, and their compose files never name the variable, so they hold none.
    """
    dev_files = {
        f"{executor_mod.deploy_repo_dir()}/docker/docker-compose.dev-lane.yml",
        f"{executor_mod.deploy_repo_dir()}/docker/docker-compose.dev-tenant-fold.yml",
    }
    configs = dict(executor_mod._lane_configs())
    layering = {
        lane
        for lane, config in configs.items()
        if dev_files & set(config.compose_files)
    }
    assert layering == {executor_mod.EnumRuntimeLane.DEV}
    mounting = layering | {executor_mod.EnumRuntimeLane.STABILITY_TEST}
    for name, instance_config in executor_mod.dev_instance_lane_configs().items():
        assert dev_files & set(instance_config.compose_files), name
        assert instance_config.lane == executor_mod.EnumRuntimeLane.DEV, name
    for lane in configs:
        with executor_mod.active_deploy_lane(lane):
            env = executor_mod._compose_env()
        assert ("ONEX_LANE_DECLARATION_DIR" in env) == (lane in mounting), lane
    stability_name = executor_mod.EnumRuntimeLane.STABILITY_TEST.value
    stability = (
        REPO_ROOT / "docker" / f"docker-compose.{stability_name}.yml"
    ).read_text(encoding="utf-8")
    assert "${ONEX_LANE_DECLARATION_DIR:?" in stability
    prod = (REPO_ROOT / "docker" / "docker-compose.prod.yml").read_text(
        encoding="utf-8"
    )
    assert "ONEX_LANE_DECLARATION_DIR" not in prod
