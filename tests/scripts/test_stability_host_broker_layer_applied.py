# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-24509: the stability lane's compose file list applies the per-host broker layer.

``docker/docker-compose.stability-host-broker.yml`` (OMN-19998) makes the
stability broker advertise the serving host's own addresses. It reads the two
lane-specific variables ``STABILITY_TEST_REDPANDA_ADVERTISED_KAFKA_ADDR`` /
``STABILITY_TEST_REDPANDA_ADVERTISED_PANDAPROXY_ADDR`` and refuses by name
when one is unset.

``resolve_compose_file_args`` asks ``runtime_host_broker_layer.py``
-- whose overlay file name, layer file name, variable names and address shape
all come from the node contract's ``config.stability_host_broker_layer``
block -- whether the compose project is exactly the stability lane's and the
host declared either variable. The layer is appended AFTER the lane overlay,
and only then. A declared variable that is empty, whitespace-only or not a
comma-separated list of ``listener://host:port`` entries makes the resolver
return nonzero with the module's refusal on stderr instead of listing any
file, so a half declaration (one variable set) still lists the layer and
compose refuses the missing address by name. No other lane -- dogfood
included -- ever gets the layer.

"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES_SH = REPO_ROOT / "scripts" / "runtime_build" / "compose_files.sh"
LAYER_CONTRACT = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_setup_local_provision_effect"
    / "contract.yaml"
)
LAYER_MODULE = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_setup_local_provision_effect"
    / "runtime_host_broker_layer.py"
)
HOST_BROKER_LAYER = "docker-compose.stability-host-broker.yml"
_LANE = "stability"
STABILITY_PROJECT = "omnibase-infra-" + _LANE + "-test"
LANE_OVERLAY = "docker-compose." + _LANE + "-test.yml"
ADVERTISED_ADDR_VARS = (
    "STABILITY_TEST_REDPANDA_ADVERTISED_KAFKA_ADDR",
    "STABILITY_TEST_REDPANDA_ADVERTISED_PANDAPROXY_ADDR",
)
KAFKA_ADDR_VAR, PANDAPROXY_ADDR_VAR = ADVERTISED_ADDR_VARS
# Synthetic, documentation-only addresses (RFC 5737's 192.0.2.x): the module
# now checks the shape of every declared value, so the examples must be valid
# comma-separated lists of listener://host:port entries.
KAFKA_ADDR_VALUE = (
    "internal://redpanda:9092,external://192.0.2.10:39092,lan://192.0.2.11:39093"
)
PANDAPROXY_ADDR_VALUE = "internal://redpanda:8082,external://192.0.2.10:28082"
ADDRESSES = {
    KAFKA_ADDR_VAR: KAFKA_ADDR_VALUE,
    PANDAPROXY_ADDR_VAR: PANDAPROXY_ADDR_VALUE,
}


def _declaration() -> dict[str, str]:
    """The contract's ``config.stability_host_broker_layer`` block, read fresh."""
    document = yaml.safe_load(LAYER_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{LAYER_CONTRACT}: not a mapping"
    config = document["config"]
    assert isinstance(config, dict), f"{LAYER_CONTRACT}: no config mapping"
    block = config["stability_host_broker_layer"]
    assert isinstance(block, dict), f"{LAYER_CONTRACT}: no stability block"
    return {str(key): str(value) for key, value in block.items()}


def _run_overlay_resolver(
    compose_project: str,
    advertised_addr_env: dict[str, str] | None = None,
    script: Path = COMPOSE_FILES_SH,
) -> subprocess.CompletedProcess[str]:
    """Execute the real resolver in compose_files.sh for one compose project.

    Mirrors the harness in test_deploy_runtime_lane_overlay.py, with two extra
    parameters: an env dict applied on top of a scrubbed environment, so the
    two advertised-address variables are exactly as the test declares them
    regardless of what the host happens to export; and the script path to
    source instead of the committed compose_files.sh (the failing control
    strips the layer wiring out of a copy and sources that).
    """
    env = dict(os.environ)
    for var in ADVERTISED_ADDR_VARS:
        env.pop(var, None)
    env.update(advertised_addr_env or {})
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            f'source "{script}"',
            "declare -a out",
            f'resolve_compose_file_args out "/DEPLOY" "{compose_project}"',
            'printf "%s\\n" "${out[*]}"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _run_layer_module(
    *args: str,
    addr_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the module's own CLI with the two address variables scrubbed first."""
    env = dict(os.environ)
    for var in ADVERTISED_ADDR_VARS:
        env.pop(var, None)
    env.update(addr_env or {})
    return subprocess.run(
        ["python3", str(LAYER_MODULE), args[0], str(LAYER_CONTRACT), *args[1:]],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.unit
def test_stability_lane_lists_the_host_broker_layer_when_both_are_set() -> None:
    result = _run_overlay_resolver(STABILITY_PROJECT, ADDRESSES)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    layer_file = _declaration()["layer_file"]
    assert out == (
        "-f /DEPLOY/docker/docker-compose.infra.yml "
        "-f /DEPLOY/docker/" + LANE_OVERLAY + " "
        "-f /DEPLOY/docker/" + layer_file
    ), out
    # The layer comes AFTER the lane overlay, so its advertised-address
    # overrides win on every listener.
    assert out.index(LANE_OVERLAY) < out.index(layer_file)


@pytest.mark.unit
def test_stability_lane_keeps_its_two_file_list_when_neither_is_set() -> None:
    """A host that sets neither address keeps the lane's existing file list."""
    result = _run_overlay_resolver(STABILITY_PROJECT)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert out == (
        "-f /DEPLOY/docker/docker-compose.infra.yml -f /DEPLOY/docker/" + LANE_OVERLAY
    ), out
    assert HOST_BROKER_LAYER not in out


@pytest.mark.unit
@pytest.mark.parametrize("var", ADVERTISED_ADDR_VARS)
def test_half_declaration_still_lists_the_layer(var: str) -> None:
    """A half declaration is not silently ignored: compose refuses by name."""
    result = _run_overlay_resolver(STABILITY_PROJECT, {var: ADDRESSES[var]})
    assert result.returncode == 0, result.stderr
    assert HOST_BROKER_LAYER in result.stdout, (
        f"setting only {var} must still list the layer, so compose refuses "
        f"the unset variable by name: {result.stdout.strip()!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "compose_project",
    ["omnibase-infra", "omnibase-infra-prod", "omnibase-infra-" + "ju" + "dge"],
)
def test_no_other_lane_ever_lists_the_layer(compose_project: str) -> None:
    result = _run_overlay_resolver(compose_project, ADDRESSES)
    assert result.returncode == 0, result.stderr
    assert HOST_BROKER_LAYER not in result.stdout, (
        f"{compose_project} must never layer the per-host broker layer: "
        f"{result.stdout.strip()!r}"
    )


@pytest.mark.unit
def test_dogfood_lane_never_lists_the_layer() -> None:
    """The dogfood lane returns before any overlay, so it never gains the layer."""
    result = _run_overlay_resolver("omnibase-infra-" + "dog" + "food", ADDRESSES)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert HOST_BROKER_LAYER not in out and len(out.split()) == 2, (
        f"the dogfood lane must stay one standalone file with both addresses set: {out!r}"
    )


@pytest.mark.unit
def test_declared_but_empty_refuses_and_lists_nothing() -> None:
    """Both variables declared empty refuse by name; no compose file list at all."""
    result = _run_overlay_resolver(
        STABILITY_PROJECT, dict.fromkeys(ADVERTISED_ADDR_VARS, "")
    )
    assert result.returncode != 0, (
        f"a declared-but-empty address must refuse: stdout={result.stdout!r}"
    )
    named = [var for var in ADVERTISED_ADDR_VARS if var in result.stderr]
    assert named, f"the refusal must name a declared variable: {result.stderr!r}"
    assert "declared but empty" in result.stderr, result.stderr
    assert result.stdout.strip() == "", (
        f"a refused declaration must list no compose file: {result.stdout!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("var", ADVERTISED_ADDR_VARS)
def test_whitespace_only_declaration_refuses_and_names_the_variable(var: str) -> None:
    """A whitespace-only value is declared-but-empty: refused, naming that variable."""
    env = dict(ADDRESSES)
    env[var] = "   "
    result = _run_overlay_resolver(STABILITY_PROJECT, env)
    assert result.returncode != 0, (
        f"whitespace-only {var} must refuse: stdout={result.stdout!r}"
    )
    assert var in result.stderr, f"the refusal must name {var}: {result.stderr!r}"
    assert "declared but empty" in result.stderr, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_addr",
    [
        pytest.param("broker.example.invalid", id="no-listener-prefix-or-port"),
        pytest.param("internal://redpanda", id="no-port"),
        pytest.param(
            "internal://redpanda:9092 external://192.0.2.10:39092",
            id="space-instead-of-comma",
        ),
        pytest.param(
            "internal://redpanda:9092, external://192.0.2.10:39092",
            id="space-after-comma",
        ),
        pytest.param("internal://redpanda:9092\n", id="trailing-newline"),
        pytest.param("INTERNAL://redpanda:9092", id="upper-case-listener"),
    ],
)
def test_malformed_value_refuses_and_is_never_echoed(bad_addr: str) -> None:
    """A declared value that is not listener://host:port entries is refused by name.

    Each refusal names the variable, says the value must be a comma-separated
    list of ``listener://host:port`` entries, and never echoes the refused
    value itself back to stderr.
    """
    result = _run_overlay_resolver(STABILITY_PROJECT, {KAFKA_ADDR_VAR: bad_addr})
    assert result.returncode != 0, (
        f"malformed {KAFKA_ADDR_VAR} must refuse: stdout={result.stdout!r}"
    )
    assert KAFKA_ADDR_VAR in result.stderr, (
        f"the refusal must name {KAFKA_ADDR_VAR}: {result.stderr!r}"
    )
    assert "comma-separated list of listener://host:port entries" in result.stderr, (
        result.stderr
    )
    assert bad_addr not in result.stderr, (
        f"the refused value must never appear on stderr: {result.stderr!r}"
    )


@pytest.mark.unit
def test_layer_for_matches_the_overlay_name_exactly() -> None:
    """Equality, not a glob: only the contract's exact overlay name gains the layer.

    The module is loaded the way the standalone deploy agent loads it, by
    path, so this exercises the file on disk rather than an import of this
    test package.
    """
    spec = importlib.util.spec_from_file_location(
        "stability_layer_under_test", LAYER_MODULE
    )
    assert spec is not None, f"cannot read a module spec from {LAYER_MODULE}"
    assert spec.loader is not None, f"no loader for {LAYER_MODULE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared = module.declaration(LAYER_CONTRACT)

    def layer_for(overlay: str, environ: dict[str, str]) -> str | None:
        found: str | None = module.layer_for(overlay, environ, declared)
        return found

    assert layer_for(LANE_OVERLAY + ".bak", ADDRESSES) is None, (
        "a .bak sibling of the lane overlay must gain no layer"
    )
    assert layer_for("docker-compose.stability-other.yml", ADDRESSES) is None, (
        "another lane's overlay must gain no layer"
    )
    assert layer_for(LANE_OVERLAY, ADDRESSES) == _declaration()["layer_file"], (
        "the lane overlay with both addresses declared must gain the contract's"
        " layer file"
    )
    assert layer_for(LANE_OVERLAY, {}) is None, (
        "no declared variable must mean no layer, even for the lane overlay"
    )


@pytest.mark.unit
def test_describe_prints_the_layer_file_and_both_declared_values() -> None:
    """With both addresses in the environment, describe prints all three inputs."""
    result = _run_layer_module("describe", LANE_OVERLAY, addr_env=ADDRESSES)
    assert result.returncode == 0, result.stderr
    layer_file = _declaration()["layer_file"]
    expected = "\n".join(
        [
            "layer_file=" + layer_file,
            "kafka_addr=" + KAFKA_ADDR_VALUE,
            "pandaproxy_addr=" + PANDAPROXY_ADDR_VALUE,
        ]
    )
    assert result.stdout == expected + "\n", result.stdout + result.stderr


@pytest.mark.unit
def test_describe_reports_none_and_unset_when_neither_is_declared() -> None:
    """With neither variable in the environment, describe reports unset inputs."""
    result = _run_layer_module("describe", LANE_OVERLAY)
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "layer_file=none\nkafka_addr=unset\npandaproxy_addr=unset\n"
    ), result.stdout + result.stderr


@pytest.mark.unit
def test_unknown_command_is_refused_with_usage() -> None:
    """An unknown command exits 64 with the usage line on stderr."""
    result = _run_layer_module("frobnicate", LANE_OVERLAY)
    assert result.returncode == 64, (
        f"an unknown command must exit 64: stderr={result.stderr!r}"
    )
    assert "usage:" in result.stderr, result.stderr


@pytest.mark.unit
def test_failing_control_declared_layer_not_applied(tmp_path: Path) -> None:
    """Fails if the contract, the layer file or the wiring stops declaring what it needs.

    The contract must name a committed layer file under docker/, that file
    must still declare ${VAR:?...} on exactly the two variable names the
    resolver consults, and compose_files.sh must still consult the module. The
    control strips every wiring line out of a copy: the copy must list no
    layer while the real script still does, so removing the wiring goes red.
    """
    declared = _declaration()
    layer_path = REPO_ROOT / "docker" / declared["layer_file"]
    assert layer_path.parent.name == "docker" and layer_path.is_file(), (
        f"the contract's layer_file must be a committed file under docker/: {layer_path}"
    )
    layer = layer_path.read_text(encoding="utf-8")
    for var in ADVERTISED_ADDR_VARS:
        assert ("${" + var + ":?") in layer, (
            "the committed layer must still declare ${" + var + ":?...} so a host"
            " that omits it is refused by name"
        )
    script = COMPOSE_FILES_SH.read_text(encoding="utf-8")
    assert "runtime_host_broker_layer.py" in script, (
        "compose_files.sh must consult runtime_host_broker_layer.py"
        " for the layer decision (OMN-24509)"
    )
    unwired_lines = [
        line
        for line in script.splitlines(keepends=True)
        if "runtime_host_broker_layer" not in line
    ]
    unwired_sh = tmp_path / "compose_files_without_layer_wiring.sh"
    unwired_sh.write_text("".join(unwired_lines), encoding="utf-8")
    unwired = _run_overlay_resolver(STABILITY_PROJECT, ADDRESSES, script=unwired_sh)
    assert unwired.returncode == 0, unwired.stderr
    assert HOST_BROKER_LAYER not in unwired.stdout, (
        f"the unwired copy must list no layer: {unwired.stdout.strip()!r}"
    )
    wired = _run_overlay_resolver(STABILITY_PROJECT, ADDRESSES)
    assert wired.returncode == 0, wired.stderr
    assert HOST_BROKER_LAYER in wired.stdout, (
        f"the real compose_files.sh must list the layer: {wired.stdout.strip()!r}"
    )
