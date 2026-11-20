# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability-lane dry run names the per-host broker layer decision (OMN-24551).

Runs ``scripts/runtime_build/refresh_stability_lane.sh`` in plan mode inside the
OMN-24119 sandbox (three tagged clones, a release pin, stub ``docker``/``curl``/
``onex-runtime-deploy``, an operator env file naming the removed embedding server), with
``ONEX_DEPLOY_GRANT_ACK`` on the unreadable-grant sentinel the attribution preflight
needs in a sandbox. The plan's ``compose files`` line is the resolver's whole ``-f`` list
plus the layer decision (``layer_file=`` / ``kafka_addr=`` / ``pandaproxy_addr=``); a
declared address that is empty or malformed stops even a dry run before anything is read
or built. The overlay file name, the layer file name and the two variable names come from
the node contract's ``config.stability_host_broker_layer`` block, and the lane id is read
from the script; no lab lane name is spelled here. Every address is synthetic (192.0.2.x).
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
_CONTRACT = (
    _REPO / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
)

#: The lane id the script hardcodes; the pin file is named for it. Read from the script
#: so no lane id is spelled in this public repository.
_LANE_MATCH = re.search(r'^readonly LANE="([^"]+)"$', _SCRIPT.read_text(), re.M)
assert _LANE_MATCH is not None
LANE = _LANE_MATCH.group(1)

#: The layer decision's own declaration: the only source of every name below (OMN-24509).
_DOCUMENT = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
_BLOCK: dict[str, str] = dict(_DOCUMENT["config"]["stability_host_broker_layer"])
KAFKA_ENV = _BLOCK["kafka_addr_env"]
PANDAPROXY_ENV = _BLOCK["pandaproxy_addr_env"]
LAYER_FILE = _BLOCK["layer_file"]
LANE_OVERLAY_FILE = _BLOCK["lane_overlay_prefix"] + _BLOCK["lane_overlay_suffix"]
INFRA_FILE = "docker-compose.infra.yml"

#: Synthetic, documentation-only addresses (RFC 5737's 192.0.2.x): valid
#: comma-separated lists of listener://host:port entries.
KAFKA_ADDR = (
    "internal://redpanda:9092,external://192.0.2.10:39092,lan://192.0.2.11:39093"
)
PANDAPROXY_ADDR = "internal://redpanda:8082,external://192.0.2.10:28082"
MALFORMED_ADDR = "internal://redpanda"
STALE_ADDRESS = "http://192.0.2.50:8002"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"
SLOT_ID = "embeddings-201"

_POP_NAMES = {
    "DEPLOY_AGENT_OVERLAY_HOME_CLONE",
    "DEPLOY_AGENT_STATE_DIR",
    "ONEX_LANE_OVERLAY_DIR",
    "LLM_EMBEDDING_URL",
    "LANE_MANIFEST",
    "OMNIBASE_INFRA_COMPOSE_PROJECT",
    "ONEX_LANE_LOCK_HELD",
    "ONEX_DEPLOY_LANE_ENV_EXPORTS",
    "STABILITY_DESIRED_STATE",
    KAFKA_ENV,
    PANDAPROXY_ENV,
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


def _tagged_clone(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "commit", "-q", "--allow-empty", "-m", "x")
    _git(path, "tag", "v1.2.3")


def _stub(path: Path, log: Path) -> None:
    path.write_text(f'#!/bin/sh\necho "$0 $*" >> "{log}"\nexit 1\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _make_sandbox(
    tmp_path: Path, addresses: dict[str, str]
) -> tuple[dict[str, str], Path]:
    """The environment to run the script in, and the file the stubs log their calls to.

    ``addresses`` is applied after the scrub, so the two advertised-address variables
    are exactly as each test declares them regardless of what the host exports.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in ("omnibase_infra", "omnibase_core", "omnimarket"):
        _tagged_clone(ws / name)

    lane_overlays = tmp_path / "cfg" / "config" / "lab" / "lane-overlays"
    lane_overlays.mkdir(parents=True)
    (lane_overlays / f"{LANE}.release-pin.yaml").write_text(
        f"schema_version: 1\nlane: {LANE}\npin:\n"
        "  omnibase_infra: v1.2.3\n  omnibase_core: v1.2.3\n  omnimarket: v1.2.3\n",
        encoding="utf-8",
    )
    (lane_overlays / "lane-under-test.bifrost.yaml").write_text(
        yaml.safe_dump(
            {
                "lane": "lane-under-test",
                "backends": [
                    {
                        "backend_id": "local-embedding",
                        "endpoint_url": f"{STALE_ADDRESS}/v1/embeddings",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    calls = tmp_path / "stub-calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("docker", "curl", "onex-runtime-deploy"):
        _stub(bin_dir / name, calls)

    operator_env = tmp_path / "operator.env"
    operator_env.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    overlay_root = tmp_path / "overlay-root"
    (overlay_root / "llm_endpoints").mkdir(parents=True)
    (overlay_root / "llm_endpoints" / "overlay.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 1, "endpoint_urls": {SLOT_ID: OVERLAY_ADDRESS}}
        ),
        encoding="utf-8",
    )

    env = scrub_git_location_env(os.environ)
    for name in _POP_NAMES:
        env.pop(name, None)
    env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
    env["HOME"] = str(home)
    env["OMNI_HOME"] = str(ws)
    env["DEPLOY_RUNTIME"] = str(bin_dir / "onex-runtime-deploy")
    env["OMNIBASE_OPERATOR_ENV_FILE"] = str(operator_env)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(tmp_path / "cfg")
    env["ONEX_SKILL_OVERLAY_ROOTS"] = str(overlay_root)
    env["ONEX_DEPLOY_REASON"] = "OMN-24551 test"
    # The attribution preflight cannot read the grant registry in a sandbox; the sentinel
    # acknowledgement (its own documented token) lets the dry run proceed on the record.
    env["ONEX_DEPLOY_GRANT_ACK"] = "unreadable-grant-state"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "scripts" / "deploy-agent"), str(_REPO / "src")]
    )
    env.update(addresses)
    return env, calls


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _compose_files_line(combined: str) -> str:
    text = next(
        (entry for entry in combined.splitlines() if "compose files :" in entry), ""
    )
    assert text, f"no compose files line in:\n{combined}"
    return text


@pytest.mark.integration
def test_golden_dry_run_names_the_layer_and_the_addresses(tmp_path: Path) -> None:
    """Both addresses declared: the plan lists infra, the lane overlay, then the layer."""
    env, calls = _make_sandbox(
        tmp_path, {KAFKA_ENV: KAFKA_ADDR, PANDAPROXY_ENV: PANDAPROXY_ADDR}
    )
    result = _run(env)
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    line = _compose_files_line(combined)
    positions = [
        line.find(name) for name in (INFRA_FILE, LANE_OVERLAY_FILE, LAYER_FILE)
    ]
    assert -1 not in positions, line
    assert positions == sorted(positions), line
    assert f"layer_file={LAYER_FILE}" in combined
    assert f"kafka_addr={KAFKA_ADDR}" in combined
    assert f"pandaproxy_addr={PANDAPROXY_ADDR}" in combined
    assert not calls.exists()


@pytest.mark.integration
def test_golden_dry_run_without_the_values_names_no_layer(tmp_path: Path) -> None:
    """Neither address declared: the plan names no layer and reports both unset."""
    env, _calls = _make_sandbox(tmp_path, {})
    result = _run(env)
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert LAYER_FILE not in combined
    assert "layer_file=none" in combined
    assert "kafka_addr=unset" in combined
    assert "pandaproxy_addr=unset" in combined


@pytest.mark.integration
def test_error_a_malformed_address_stops_the_dry_run(tmp_path: Path) -> None:
    """A malformed declared address refuses by name and echoes nothing."""
    env, calls = _make_sandbox(tmp_path, {KAFKA_ENV: MALFORMED_ADDR})
    result = _run(env)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert KAFKA_ENV in combined
    assert "would run:" not in combined
    assert not calls.exists()
    assert MALFORMED_ADDR not in combined


@pytest.mark.integration
def test_error_a_declared_but_empty_address_stops_the_dry_run(tmp_path: Path) -> None:
    """A declared-but-empty address is refused by name, never read as no layer."""
    env, _calls = _make_sandbox(tmp_path, {KAFKA_ENV: "", PANDAPROXY_ENV: ""})
    result = _run(env)
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "declared but empty" in combined
    assert KAFKA_ENV in combined
    assert "would run:" not in combined
