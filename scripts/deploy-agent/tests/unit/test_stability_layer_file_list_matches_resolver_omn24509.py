# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The executor's stability file list matches the shell resolver's, layer in (OMN-24509).

``resolve_compose_file_args`` (``scripts/runtime_build/compose_files.sh``)
appends the per-host broker layer after the stability overlay when the host
declares either advertised broker address; the executor spelled the lane's
list as the static ``_LANE_CONFIGS`` tuple, so a full-scope ``up`` through it
would recreate the stability broker WITHOUT the layer. It now takes its list
from the same module; these tests run the real shell resolver and the real
executor derivation side by side and refuse any difference. Every address is
synthetic (RFC 5737's 192.0.2.x) and every name comes from the contract's
``config.stability_host_broker_layer`` block, never a copy in this file.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPOSE_FILES_SH = REPO_ROOT / "scripts" / "runtime_build" / "compose_files.sh"
CONTRACT = REPO_ROOT / (
    "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
)
STABILITY_PROJECT = "omnibase-infra-" + "stability" + "-test"
KAFKA_ADDR_VALUE = (
    "internal://redpanda:9092,external://192.0.2.10:39092,lan://192.0.2.11:39093"
)
PANDAPROXY_ADDR_VALUE = "internal://redpanda:8082,external://192.0.2.10:28082"


def _declaration() -> dict[str, str]:
    """The contract's ``config.stability_host_broker_layer`` block, read fresh."""
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    block = document["config"]["stability_host_broker_layer"]
    return {str(name): str(value) for name, value in block.items()}


_DECLARATION = _declaration()
KAFKA_ADDR_VAR = _DECLARATION["kafka_addr_env"]
PANDAPROXY_ADDR_VAR = _DECLARATION["pandaproxy_addr_env"]
LAYER_FILE = _DECLARATION["layer_file"]
ADDRESSES = {
    KAFKA_ADDR_VAR: KAFKA_ADDR_VALUE,
    PANDAPROXY_ADDR_VAR: PANDAPROXY_ADDR_VALUE,
}


def run_resolver(env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the real ``resolve_compose_file_args`` for the stability project.

    The two advertised-address variables are scrubbed from the environment
    first, so the run sees exactly what ``env`` declares regardless of what
    the host happens to export.
    """
    environ = dict(os.environ)
    for var in ADDRESSES:
        environ.pop(var, None)
    environ.update(env)
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            f'source "{COMPOSE_FILES_SH}"',
            "declare -a out",
            f'resolve_compose_file_args out "/DEPLOY" "{STABILITY_PROJECT}"',
            'printf "%s\\n" "${out[*]}"',
        ]
    )
    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        env=environ,
        check=False,
    )


def shell_files(env: Mapping[str, str]) -> list[str]:
    """The basenames of the ``-f`` paths the real resolver lists for ``env``."""
    result = run_resolver(env)
    assert result.returncode == 0, result.stderr
    return [Path(path).name for path in result.stdout.split()[1::2]]


def _apply(
    monkeypatch: pytest.MonkeyPatch, env: Mapping[str, str], store: Mapping[str, str]
) -> None:
    """Make the executor's two env sources exactly ``env`` and ``store``."""
    for var in ADDRESSES:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)

    def overlay(environ: Mapping[str, str] | None = None) -> dict[str, str]:
        return dict(store)

    monkeypatch.setattr(executor_mod, "operator_store_overlay", overlay)


def executor_files(
    monkeypatch: pytest.MonkeyPatch,
    env: Mapping[str, str],
    store: Mapping[str, str] | None = None,
) -> list[str]:
    """The basenames of the stability ``-f`` list the executor derives.

    The same two sources ``_compose_env`` applies first: the process
    environment and the operator store, stubbed to ``store`` here.
    """
    _apply(monkeypatch, env, store or {})
    args = executor_mod._compose_file_args(EnumRuntimeLane.STABILITY_TEST)
    return [Path(path).name for path in args[1::2]]


DECLARED_CASES = [
    pytest.param({}, id="neither"),
    pytest.param(dict(ADDRESSES), id="both"),
    pytest.param({KAFKA_ADDR_VAR: KAFKA_ADDR_VALUE}, id="only-kafka"),
    pytest.param({PANDAPROXY_ADDR_VAR: PANDAPROXY_ADDR_VALUE}, id="only-pandaproxy"),
]


@pytest.mark.unit
@pytest.mark.parametrize("env", DECLARED_CASES)
def test_executor_file_list_matches_the_shell_resolver(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    shell = shell_files(env)
    assert executor_files(monkeypatch, env) == shell
    if env:
        assert shell[-1] == LAYER_FILE


@pytest.mark.unit
def test_the_operator_store_alone_declares_the_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store-assigned address reaches the decision, as it reaches compose."""
    executor = executor_files(monkeypatch, {}, store=dict(ADDRESSES))
    assert executor == shell_files(dict(ADDRESSES))
    assert executor[-1] == LAYER_FILE


@pytest.mark.unit
@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(dict.fromkeys(ADDRESSES, ""), id="both-empty"),
        pytest.param({KAFKA_ADDR_VAR: "internal://redpanda"}, id="malformed-kafka"),
    ],
)
def test_a_refused_declaration_stops_both_resolvers(
    monkeypatch: pytest.MonkeyPatch, declared: dict[str, str]
) -> None:
    """An empty or malformed address refuses in the shell AND in the executor."""
    result = run_resolver(declared)
    assert result.returncode != 0, (
        f"the shell resolver must refuse: stdout={result.stdout!r}"
    )
    layer_module = executor_mod.load_dev_boot_guard(
        "nodes.node_setup_local_provision_effect.runtime_host_broker_layer"
    )
    with pytest.raises(layer_module.StabilityHostBrokerLayerError):
        executor_files(monkeypatch, declared)


@pytest.mark.unit
def test_the_prod_lane_never_gains_the_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the stability lane's list is layered; prod stays its static pair."""
    _apply(monkeypatch, dict(ADDRESSES), {})
    args = executor_mod._compose_file_args(EnumRuntimeLane.PROD)
    assert LAYER_FILE not in [Path(path).name for path in args[1::2]]


@pytest.mark.unit
def test_control_the_static_tuple_drops_the_layer() -> None:
    """The old static list fails the comparison above: it has no layer file.

    With both addresses declared, the shell lists the layer while the static
    ``_LANE_CONFIGS[STABILITY_TEST].compose_files`` tuple does not -- the
    exact difference the executor comparison guards, so reverting
    ``lane_config_for`` to the static tuple goes red there.
    """
    static = [
        Path(path).name
        for path in executor_mod._lane_configs()[
            EnumRuntimeLane.STABILITY_TEST
        ].compose_files
    ]
    assert static != shell_files(dict(ADDRESSES))
    assert LAYER_FILE not in static
