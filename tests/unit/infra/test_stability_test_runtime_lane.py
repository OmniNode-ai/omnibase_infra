# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static checks for the OMN-10281 stability-test runtime lane."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
OVERLAY_FILE = REPO_ROOT / "docker" / "docker-compose.stability-test.yml"
RUNTIME_POLICY_ENV_FILE = REPO_ROOT / "docker" / "runtime-policy.env"

# OMN-14013: the durable, committed partition-cap value for the stability-test
# lane. This is deliberately ABOVE the base redpanda.yaml default (7000) --
# stability-test has historically accumulated ~4x dev's topic count with no
# retirement process, and a cap raised only via a live `rpk cluster config
# set` does NOT survive this lane's redeploys (warm_broker_topic_provisioning()
# in deploy-runtime.sh force-recreates the redpanda-partition-cap one-shot on
# every redeploy, re-running whatever value is hardcoded in the compose
# override below). Both belts that declare this value in
# docker-compose.stability-test.yml -- the `redpanda` service's `--set
# topic_partitions_per_shard=...` startup flag and the `redpanda-partition-cap`
# service's `rpk cluster config set topic_partitions_per_shard ...` call --
# must agree with this constant, or the assertions below fail.
STABILITY_TEST_TOPIC_PARTITIONS_PER_SHARD = 15000

RUNTIME_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
RUNTIME_ADDRESS_PREFIX = "runtime://omninode-pc/stability-test/"
OUT_OF_LANE_SERVICES = {
    "agent-actions-consumer",
    "skill-lifecycle-consumer",
    "context-audit-consumer",
    "intelligence-api",
    "omninode-contract-resolver",
    "phoenix",
    "autoheal",
    "infisical",
}
PRODUCTION_CONTAINER_NAMES = {
    "omninode-runtime",
    "omninode-runtime-effects",
    "omnibase-infra-postgres",
    "omnibase-infra-redpanda",
    "omnibase-infra-valkey",
    "omnibase-infra-forward-migration",
    "omnibase-infra-infisical",
    "omninode-agent-actions-consumer",
    "omninode-skill-lifecycle-consumer",
    "omninode-context-audit-consumer",
    "omnibase-intelligence-api",
    "omnibase-intelligence-migration",
    "omninode-contract-resolver",
    "omnibase-infra-phoenix",
    "omnibase-infra-autoheal",
}
PRODUCTION_NETWORK_NAMES = {
    "omnibase-infra-network",
    "omnimemory-network",
}
PRODUCTION_GROUP_IDS = {
    "onex-runtime-main",
    "onex-runtime-effects",
    "onex-runtime-workers",
}
PRODUCTION_VOLUME_NAMES = {
    "omninode-runtime-data",
    "omninode-runtime-logs",
    "omninode-effects-data",
    "omninode-effects-logs",
    "omninode-worker-logs",
    "omnibase-infra-postgres-data",
    "omnibase-infra-redpanda-data",
    "omnibase-infra-valkey-data",
}


def _construct_compose_value(
    loader: yaml.SafeLoader,
    node: yaml.Node,
) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _TestSafeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


_TestSafeLoader.add_constructor("!override", _construct_compose_value)


def _load_overlay() -> dict[str, Any]:
    overlay_text = OVERLAY_FILE.read_text(encoding="utf-8")
    overlay = yaml.load(overlay_text, Loader=_TestSafeLoader)  # noqa: S506
    assert isinstance(overlay, dict)
    return overlay


def _load_runtime_policy_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in RUNTIME_POLICY_ENV_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        env[key] = value
    return env


def _load_base_compose() -> dict[str, Any]:
    base_text = BASE_COMPOSE_FILE.read_text(encoding="utf-8")
    base = yaml.load(base_text, Loader=yaml.SafeLoader)
    assert isinstance(base, dict)
    return base


def _labels_by_key(labels: list[str]) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for label in labels:
        key, value = label.split("=", maxsplit=1)
        label_map[key] = value
    return label_map


@pytest.mark.unit
def test_stability_lane_has_distinct_compose_project_name() -> None:
    overlay = _load_overlay()

    assert overlay["name"] == "omnibase-infra-stability-test"
    assert overlay["name"] != "omnibase-infra"


@pytest.mark.unit
def test_stability_lane_runtime_services_do_not_reuse_production_names() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in RUNTIME_SERVICES:
        service = services[service_name]
        container_name = service["container_name"]
        assert "stability-test" in container_name
        assert container_name not in PRODUCTION_CONTAINER_NAMES
        assert "image" not in service
        assert "build" not in service
        assert service["restart"] == "unless-stopped"


@pytest.mark.unit
def test_stability_lane_runtime_worker_is_not_scaled_to_zero() -> None:
    """OMN-12990: the worker replica count must fail-fast on the ledgered value.

    The base compose default used to be a bare ``${WORKER_REPLICAS:-0}``
    (lane-prefixed and fail-closed since OMN-14968); a soft ``:-1`` default
    here would silently re-introduce the silent-drop hole on any recreate that
    omitted the policy env. The override must reference the contract-rendered
    ``STABILITY_TEST_WORKER_REPLICAS`` with the ``:?`` fail-fast form so a missing
    policy env aborts the recreate loudly instead of dropping the worker.
    """
    overlay = _load_overlay()
    worker = overlay["services"]["runtime-worker"]
    replicas = worker["deploy"]["replicas"]

    assert replicas.startswith("${STABILITY_TEST_WORKER_REPLICAS:?")
    assert ":-" not in replicas
    assert replicas != "${STABILITY_TEST_WORKER_REPLICAS:-0}"
    assert replicas != "${STABILITY_TEST_WORKER_REPLICAS:-1}"


@pytest.mark.unit
def test_stability_lane_worker_replicas_pinned_in_ledgered_policy_env() -> None:
    """OMN-12990: the worker replica count is declared in the ledgered env.

    The compose override is fail-fast on STABILITY_TEST_WORKER_REPLICAS; the value
    itself must live in docker/runtime-policy.env (rendered from the runtime policy
    contract), pinned to a running worker so a lane recreate preserves it.
    """
    policy_env = _load_runtime_policy_env()

    assert policy_env["STABILITY_TEST_WORKER_REPLICAS"] == "1"
    assert int(policy_env["STABILITY_TEST_WORKER_REPLICAS"]) >= 1


@pytest.mark.unit
def test_stability_lane_uses_workspace_selector_and_isolated_groups() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in RUNTIME_SERVICES:
        environment = services[service_name]["environment"]
        assert "BUILD_SOURCE" not in environment
        assert "EXPECTED_BUILD_SOURCE" not in environment
        assert environment["ONEX_ENVIRONMENT"] == "stability-test"
        assert environment["KAFKA_ENVIRONMENT"] == "stability-test"
        assert environment["ONEX_STATE_ROOT"] == "/app/data/.onex_state_stability_test"
        assert (
            environment["ONEX_INFRA_HOST"]
            == "${ONEX_INFRA_HOST:?ONEX_INFRA_HOST required for stability session health probes}"
        )
        assert (
            environment["ONEX_INFRA_USER"]
            == "${ONEX_INFRA_USER:?ONEX_INFRA_USER required for stability session health probes}"
        )
        assert (
            environment["OMNI_HOME"]
            == "${OMNI_HOME:?OMNI_HOME required for stability session health probes}"
        )
        assert environment["SSH_STRICT_HOST_KEY_CHECKING"] == (
            "${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
        )
        assert environment["ONEX_STATE_DIR"] == environment["ONEX_STATE_ROOT"]
        assert environment["KAFKA_INSTANCE_ID"].startswith("stability-test-")
        assert environment["KAFKA_MAX_POLL_INTERVAL_MS"] == "1800000"

        group_id = environment["ONEX_GROUP_ID"]
        assert group_id.startswith("onex-stability-test-")
        assert group_id not in PRODUCTION_GROUP_IDS


@pytest.mark.unit
def test_stability_lane_runtime_services_have_unique_addresses() -> None:
    overlay = _load_overlay()
    policy_env = _load_runtime_policy_env()
    services = overlay["services"]
    addresses: list[str] = []
    runtime_ids: list[str] = []
    capability_keys = {
        "omninode-runtime": "STABILITY_TEST_RUNTIME_MAIN_CAPABILITIES",
        "runtime-effects": "STABILITY_TEST_RUNTIME_EFFECTS_CAPABILITIES",
        "runtime-worker": "STABILITY_TEST_RUNTIME_WORKER_CAPABILITIES",
    }

    for service_name in RUNTIME_SERVICES:
        service = services[service_name]
        environment = service["environment"]
        labels = _labels_by_key(service["labels"])
        capability_key = capability_keys[service_name]

        assert environment["ONEX_BOX_ID"] == "omninode-pc"
        assert environment["ONEX_RUNTIME_ADDRESS"].startswith(RUNTIME_ADDRESS_PREFIX)
        assert (
            environment["ONEX_RUNTIME_ADDRESS"]
            == labels["com.omninode.runtime.address"]
        )
        assert environment["ONEX_RUNTIME_ID"] == labels["com.omninode.runtime.id"]
        assert environment["ONEX_RUNTIME_ID"].startswith("stability-test-")
        assert capability_key in environment["ONEX_RUNTIME_CAPABILITIES"]
        assert "runtime." in policy_env[capability_key]

        addresses.append(environment["ONEX_RUNTIME_ADDRESS"])
        runtime_ids.append(environment["ONEX_RUNTIME_ID"])

    assert len(addresses) == len(set(addresses))
    assert len(runtime_ids) == len(set(runtime_ids))


@pytest.mark.unit
def test_stability_lane_runtime_ports_override_production_bindings() -> None:
    overlay = _load_overlay()
    policy_env = _load_runtime_policy_env()
    services = overlay["services"]

    assert services["postgres"]["ports"] == [
        "${STABILITY_TEST_POSTGRES_EXTERNAL_PORT:-15436}:5432"
    ]
    assert services["redpanda"]["ports"] == [
        "39092:19092",
        "29644:9644",
    ]
    assert services["valkey"]["ports"] == [
        "${STABILITY_TEST_VALKEY_EXTERNAL_PORT:-26379}:6379"
    ]
    assert services["omninode-runtime"]["ports"] == [
        "${STABILITY_TEST_RUNTIME_MAIN_PORT:?runtime policy contract must set STABILITY_TEST_RUNTIME_MAIN_PORT}:8085"
    ]
    assert policy_env["STABILITY_TEST_RUNTIME_MAIN_PORT"] == "18085"
    assert services["runtime-effects"]["ports"] == [
        "${STABILITY_TEST_RUNTIME_EFFECTS_PORT:?runtime policy contract must set STABILITY_TEST_RUNTIME_EFFECTS_PORT}:8085"
    ]
    assert policy_env["STABILITY_TEST_RUNTIME_EFFECTS_PORT"] == "18086"


@pytest.mark.unit
def test_stability_lane_redpanda_requires_connected_network_advertise_host() -> None:
    overlay = _load_overlay()
    redpanda_command = " ".join(overlay["services"]["redpanda"]["command"])

    assert "100.109.203.94:39092" in redpanda_command
    assert "100.109.203.94:28082" in redpanda_command
    assert "STABILITY_TEST_REDPANDA_ADVERTISE_HOST" not in redpanda_command
    assert "${REDPANDA_ADVERTISE_HOST" not in redpanda_command
    assert "192.168.86.201:39092" not in redpanda_command
    assert "localhost:19092" not in redpanda_command

    contract_overlay = overlay["x-omninode-contract-overlay"]
    redpanda_contract = contract_overlay["stability-test-redpanda"]
    assert (
        redpanda_contract["advertised_kafka_addr"]
        == "internal://redpanda:9092,external://100.109.203.94:39092"
    )
    assert (
        redpanda_contract["advertised_pandaproxy_addr"]
        == "internal://redpanda:8082,external://100.109.203.94:28082"
    )


def _extract_partition_cap_value(text: str) -> int:
    """Parse the literal N from `... cluster config set topic_partitions_per_shard N`.

    A bare substring check (`"15000" in text`) is a trap here: this same
    command string also carries prose comments/history that mention other
    numbers (e.g. "reset to 7000", "raised to 8000 as a stopgap") -- those
    would make a substring assertion pass even if the actual `rpk` invocation
    regressed to a stale value. Extract the value the shell would actually
    pass to `rpk`, not just any digit sequence that appears in the text.
    """
    match = re.search(r"cluster config set topic_partitions_per_shard\s+(\d+)", text)
    assert match is not None, (
        f"no 'cluster config set topic_partitions_per_shard N' found in: {text!r}"
    )
    return int(match.group(1))


def _extract_partition_cap_set_flag_value(command_tokens: list[str]) -> int:
    """Parse the literal N from a `--set topic_partitions_per_shard=N` compose
    command token list (handles both a single joined token and a `--set` /
    `topic_partitions_per_shard=N` token pair)."""
    joined = " ".join(command_tokens)
    match = re.search(r"topic_partitions_per_shard=(\d+)", joined)
    assert match is not None, (
        f"no '--set topic_partitions_per_shard=N' flag found in: {command_tokens!r}"
    )
    return int(match.group(1))


@pytest.mark.unit
def test_stability_lane_sets_redpanda_partition_capacity_before_runtime() -> None:
    overlay = _load_overlay()
    services = overlay["services"]
    partition_cap_service = services["redpanda-partition-cap"]
    command = partition_cap_service["command"][0]

    assert partition_cap_service["container_name"] == (
        "omnibase-infra-stability-test-redpanda-partition-cap"
    )
    assert partition_cap_service["entrypoint"] == ["/bin/sh", "-c"]
    assert "set -eu" in command
    assert "topic_partitions_per_shard" in command
    assert (
        _extract_partition_cap_value(command)
        == STABILITY_TEST_TOPIC_PARTITIONS_PER_SHARD
    )
    assert "topic_memory_per_partition" in command
    assert "1048576" in command
    assert partition_cap_service["depends_on"]["redpanda"]["condition"] == (
        "service_healthy"
    )


@pytest.mark.unit
def test_stability_lane_redpanda_startup_flag_matches_partition_cap_service() -> None:
    """OMN-14013 regression guard: the lane's `redpanda` service overrides
    `command:` wholesale (Docker Compose replaces, not merges, a scalar-list
    key across `-f` files), which had silently DROPPED the base's `--set
    topic_partitions_per_shard=...` startup flag for this lane entirely (belt
    #2 of the documented 3-belt cap design was never applied here). Assert
    both belts are present here AND numerically agree -- a future edit to one
    without the other is exactly the kind of silent belt-drift this lane's
    partition-cap regressions came from."""
    overlay = _load_overlay()
    services = overlay["services"]
    redpanda_command = services["redpanda"]["command"]
    partition_cap_command = services["redpanda-partition-cap"]["command"][0]

    assert (
        _extract_partition_cap_set_flag_value(redpanda_command)
        == STABILITY_TEST_TOPIC_PARTITIONS_PER_SHARD
    )
    assert (
        _extract_partition_cap_value(partition_cap_command)
        == STABILITY_TEST_TOPIC_PARTITIONS_PER_SHARD
    )
    assert (
        services["omninode-runtime"]["depends_on"]["redpanda-partition-cap"][
            "condition"
        ]
        == "service_completed_successfully"
    )
    for service_name in RUNTIME_SERVICES:
        assert (
            services[service_name]["depends_on"]["migration-gate"]["condition"]
            == "service_healthy"
        )
    assert (
        services["omninode-runtime"]["depends_on"]["intelligence-migration"][
            "condition"
        ]
        == "service_completed_successfully"
    )


@pytest.mark.unit
def test_stability_lane_explicitly_leaves_omnimemory_inactive() -> None:
    overlay = _load_overlay()
    policy_env = _load_runtime_policy_env()
    services = overlay["services"]
    env_keys = {
        "omninode-runtime": (
            "STABILITY_TEST_RUNTIME_MAIN_OMNIMEMORY_ENABLED",
            "STABILITY_TEST_RUNTIME_MAIN_OMNIMEMORY_MEMGRAPH_HOST",
        ),
        "runtime-effects": (
            "STABILITY_TEST_RUNTIME_EFFECTS_OMNIMEMORY_ENABLED",
            "STABILITY_TEST_RUNTIME_EFFECTS_OMNIMEMORY_MEMGRAPH_HOST",
        ),
        "runtime-worker": (
            "STABILITY_TEST_RUNTIME_WORKER_OMNIMEMORY_ENABLED",
            "STABILITY_TEST_RUNTIME_WORKER_OMNIMEMORY_MEMGRAPH_HOST",
        ),
    }

    for service_name in RUNTIME_SERVICES:
        environment = services[service_name]["environment"]
        enabled_key, host_key = env_keys[service_name]
        assert enabled_key in environment["OMNIMEMORY_ENABLED"]
        assert host_key in environment["OMNIMEMORY_MEMGRAPH_HOST"]
        assert policy_env[enabled_key] == "false"
        assert policy_env[host_key] == ""


@pytest.mark.unit
def test_stability_lane_runtime_services_preserve_image_runtime_contracts() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in RUNTIME_SERVICES:
        volumes = services[service_name]["volumes"]
        assert "../contracts:/app/contracts:ro" not in volumes
        assert all(":/app/contracts" not in volume for volume in volumes)
        assert any(volume.endswith(":/app/skills:ro") for volume in volumes)


@pytest.mark.unit
def test_stability_lane_runtime_services_define_session_health_contract() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in RUNTIME_SERVICES:
        environment = services[service_name]["environment"]
        volumes = services[service_name]["volumes"]

        assert (
            "ONEX_INFRA_HOST required for stability session health probes"
            in environment["ONEX_INFRA_HOST"]
        )
        assert (
            "ONEX_INFRA_USER required for stability session health probes"
            in environment["ONEX_INFRA_USER"]
        )
        assert (
            "OMNI_HOME required for stability session health probes"
            in environment["OMNI_HOME"]
        )
        assert environment["SSH_STRICT_HOST_KEY_CHECKING"] == (
            "${SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
        )
        assert (
            "${OMNI_HOME:?OMNI_HOME required for stability session health probes}:"
            "${OMNI_HOME:?OMNI_HOME required for stability session health probes}:ro"
            in volumes
        )


@pytest.mark.unit
def test_stability_lane_runtime_healthchecks_are_semantic_not_shallow() -> None:
    """Every runtime container on the proof lane must override the shallow probe.

    OMN-15217. The base compose probe is ``curl -sf http://localhost:8085/health``,
    which asserts one property: HTTP status < 400. ``/health`` returns 200 for a
    running-but-DEGRADED runtime *by design* (degraded containers stay in rotation
    rather than triggering cascading restarts), so the base probe is a liveness
    check wearing a health check's name. Live proof of the mask on this lane,
    2026-07-27: all three runtime containers read ``Up 4 hours (healthy)`` while
    their monitors logged ``status=DEGRADED``.

    This lane's Docker health is what gets cited as stability-proof for prod
    promotion (OMN-13418), so the assertion is *replacement*, not addition: an
    overlay that merely appended a strict probe alongside the inherited one, or
    that covered two of three containers, would leave the lane able to lie.
    """
    base = _load_base_compose()
    overlay = _load_overlay()

    for service_name in RUNTIME_SERVICES:
        # The mask this override exists to close is still the base default —
        # dev/prod keep the shallow probe pending this lane's canary.
        assert base["services"][service_name]["healthcheck"]["test"] == [
            "CMD",
            "curl",
            "-sf",
            "http://localhost:8085/health",
        ], f"{service_name}: base probe changed; re-derive what this lane overrides"

        healthcheck = overlay["services"][service_name]["healthcheck"]

        # Exact list, not a substring/`in` check: the probe is what Docker
        # executes, so a partial match would accept an extra shallow fallback.
        assert healthcheck["test"] == [
            "CMD",
            "python",
            "/usr/local/bin/onex-container-healthcheck",
            "--degraded-policy",
            "fail",
        ], f"{service_name}: proof lane must run the strict semantic check"

        # The override must actually displace the inherited probe. Compose
        # replaces `healthcheck` wholesale, so curl surviving anywhere in the
        # resolved probe would mean the override was authored wrong.
        assert "curl" not in healthcheck["test"], (
            f"{service_name}: shallow curl probe survived the strict override"
        )

        # --degraded-policy fail is the load-bearing flag: without it the check
        # degrades to the same pass-on-DEGRADED semantics as curl -sf.
        assert healthcheck["test"][-2:] == ["--degraded-policy", "fail"], (
            f"{service_name}: strict policy flag missing — probe would pass DEGRADED"
        )

        # Flap budget. The monitor's first verdict lands ~one
        # RUNTIME_HEALTH_CHECK_INTERVAL (300s) after boot; retries must outlast
        # a single missed interval, and start_period must not shrink below the
        # base service's own budget or boot would flap on an absent verdict.
        assert healthcheck["interval"] == "30s"
        assert healthcheck["timeout"] == "10s"
        assert healthcheck["retries"] == 5
        assert (
            healthcheck["start_period"]
            == base["services"][service_name]["healthcheck"]["start_period"]
        ), f"{service_name}: strict probe must keep the base start_period budget"


@pytest.mark.unit
def test_stability_lane_core_infra_names_are_distinct() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in (
        "postgres",
        "redpanda",
        "valkey",
        "migration-gate",
        "forward-migration",
        "intelligence-migration",
    ):
        container_name = services[service_name]["container_name"]
        assert "stability-test" in container_name
        assert container_name not in PRODUCTION_CONTAINER_NAMES


@pytest.mark.unit
def test_stability_lane_intelligence_migration_is_isolated() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    assert services["intelligence-migration"]["container_name"] == (
        "omnibase-infra-stability-test-intelligence-migration"
    )


@pytest.mark.unit
def test_stability_lane_networks_do_not_reuse_production_names() -> None:
    overlay = _load_overlay()
    networks = overlay["networks"]

    for network_name, network_config in networks.items():
        concrete_name = network_config["name"]
        assert concrete_name not in PRODUCTION_NETWORK_NAMES, network_name
        assert "stability-test" in concrete_name, network_name

    assert networks["omnimemory-network"]["driver"] == "bridge"
    assert networks["omnimemory-network"]["external"] is False


@pytest.mark.unit
def test_stability_lane_disables_out_of_lane_inherited_services() -> None:
    overlay = _load_overlay()
    services = overlay["services"]

    for service_name in OUT_OF_LANE_SERVICES:
        assert services[service_name]["profiles"] == ["stability-test-disabled"]


@pytest.mark.unit
def test_stability_lane_volumes_do_not_reuse_production_names() -> None:
    overlay = _load_overlay()
    volumes = overlay["volumes"]
    concrete_names = [volume_config["name"] for volume_config in volumes.values()]

    for volume_name, volume_config in volumes.items():
        concrete_name = volume_config["name"]
        assert concrete_name not in PRODUCTION_VOLUME_NAMES, volume_name
        assert "stability-test" in concrete_name, volume_name

    assert len(concrete_names) == len(set(concrete_names))
