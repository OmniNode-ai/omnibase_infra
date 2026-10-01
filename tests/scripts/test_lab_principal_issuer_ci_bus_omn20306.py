# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20306: a developer machine gets its own CI-bus login for lab work.

The second issuer instance is the same code with the CI bus's grants file, its
compose file and the onboarding facts that point at it. These cover what the
grants allow, what they leave out, and that the instance is wired to the CI bus.
The broker is a fake: no network, no Docker.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from scripts import lab_principal_issuer as lpi

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.ci-bus.yaml"
DEV_GRANTS = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"
FACTS = REPO_ROOT / "deploy" / "lab" / "developer-onboarding.yaml"
COMPOSE = "docker/docker-compose.principal-issuer-ci-bus.yml"
LAB_WORK_REQUEST = "onex.cmd.omnimarket.lab-work-unit-requested.v1"

pytestmark = pytest.mark.unit


def _load_yaml(relative_path: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8")),
    )


def _facts() -> dict[str, str]:
    keys: dict[str, str] = {}
    for line in FACTS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([a-z_]+):[ \t]*(\S.*)", line)
        assert match, f"not a flat key: value line: {line!r}"
        keys[match.group(1)] = match.group(2)
    return keys


@pytest.fixture
def declaration() -> lpi.Declaration:
    return lpi.load_declaration(CI_GRANTS)


# --- the grants ----------------------------------------------------------------


def test_the_ci_bus_grant_file_is_valid_and_for_the_ci_bus(
    declaration: lpi.Declaration,
) -> None:
    assert declaration.lane == "ci-bus"
    assert lpi.main(["check", "--grants", str(CI_GRANTS)]) == 0


def test_only_the_unit_request_is_writable(declaration: lpi.Declaration) -> None:
    """A caller sends units; only a pool host publishes receipts or capacity."""
    writable = [g for g in declaration.grants if "write" in g.operations]
    assert [(g.resource, g.name, g.pattern) for g in writable] == [
        ("topic", LAB_WORK_REQUEST, "literal")
    ]


def test_the_caller_reads_its_terminals_and_the_capacity_ads(
    declaration: lpi.Declaration,
) -> None:
    readable = {
        g.name
        for g in declaration.grants
        if g.resource == "topic" and "read" in g.operations
    }
    assert readable == {
        "onex.evt.omnimarket.lab-work-unit-completed.v1",
        "onex.evt.omnimarket.lab-work-unit-failed.v1",
        "onex.evt.omnimarket.lab-host-capacity-advertised.v1",
    }


def test_no_pool_host_group_focused_test_push_validation_or_ci_topic(
    declaration: lpi.Declaration,
) -> None:
    names = " ".join(g.name for g in declaration.grants)
    for absent in (
        "node_lab_work_unit_effect",
        "focused-test-run",
        "node_focused_test_run_effect",
        "push-validation",
        "onex.evt.github.",
    ):
        assert absent not in names, absent
    (group,) = [g for g in declaration.grants if g.resource == "group"]
    assert group.name == "local.omnimarket.lab_work_client.consume.v1."
    assert group.pattern == "prefixed"
    assert group.operations == ("describe", "read")


def test_the_one_cluster_grant_is_idempotent_write(
    declaration: lpi.Declaration,
) -> None:
    clusters = [g for g in declaration.grants if g.resource == "cluster"]
    assert [(g.name, g.operations) for g in clusters] == [
        ("kafka-cluster", ("idempotent_write",))
    ]


def test_a_cluster_grant_reaches_rpk_as_the_cluster_flag(
    declaration: lpi.Declaration,
) -> None:
    seen: list[list[str]] = []

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    broker = lpi.Broker(lpi.BrokerConfig("http://admin:9644", "su", "su-pass"), run=run)
    (cluster,) = [g for g in declaration.grants if g.resource == "cluster"]
    broker.grant("dev-a-b", cluster)
    assert seen == [
        [
            "rpk",
            "security",
            "acl",
            "create",
            "--allow-principal",
            "User:dev-a-b",
            "--operation",
            "idempotent_write",
            "--cluster",
        ]
    ]


@pytest.mark.parametrize(
    "grant",
    [
        {
            "resource": "cluster",
            "name": "kafka-cluster",
            "pattern": "literal",
            "operations": ["alter"],
        },
        {
            "resource": "cluster",
            "name": "kafka-cluster",
            "pattern": "literal",
            "operations": ["idempotent_write", "describe"],
        },
        {
            "resource": "cluster",
            "name": "other",
            "pattern": "literal",
            "operations": ["idempotent_write"],
        },
        {
            "resource": "cluster",
            "name": "kafka-cluster",
            "pattern": "prefixed",
            "operations": ["idempotent_write"],
        },
        {
            "resource": "topic",
            "name": "t",
            "pattern": "literal",
            "operations": ["idempotent_write"],
        },
    ],
)
def test_a_cluster_grant_carries_idempotent_write_and_nothing_else(
    tmp_path: Path, grant: dict[str, object]
) -> None:
    doc = yaml.safe_load(CI_GRANTS.read_text())
    doc["grants"] = [grant]
    path = tmp_path / "grants.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(lpi.DeclarationError):
        lpi.load_declaration(path)


def test_the_dev_lane_grants_are_unchanged_by_the_cluster_rule() -> None:
    dev = lpi.load_declaration(DEV_GRANTS)
    assert dev.lane == "dev"
    assert not [g for g in dev.grants if g.resource == "cluster"]


# --- the deployment ------------------------------------------------------------


def test_the_instance_joins_the_ci_bus_network() -> None:
    issuer = _load_yaml(COMPOSE)
    ci_bus = _load_yaml("docker/docker-compose.ci-bus.yml")
    (network_key,) = ci_bus["services"]["redpanda"]["networks"]
    assert (
        issuer["networks"]["ci-bus"]["name"] == ci_bus["networks"][network_key]["name"]
    )
    manifest = _load_yaml("deploy/lane-census/lane-manifest.yaml")["lanes"]
    row = manifest["principal-issuer-ci-bus"]
    assert row["network"] == issuer["networks"]["ci-bus"]["name"]
    assert row["compose_file"] == COMPOSE
    assert row["compose_project"] == issuer["name"]


def test_the_instance_serves_the_ci_bus_grants_on_a_unix_socket_only() -> None:
    compose = _load_yaml(COMPOSE)
    service = compose["services"]["principal-issuer-ci-bus"]
    assert "ports" not in service
    assert "expose" not in service
    command = service["command"]
    assert (
        command[command.index("--grants") + 1]
        == "/app/developer-principal-grants.ci-bus.yaml"
    )
    assert command[command.index("--socket") + 1] == "/run/principal-issuer/issuer.sock"
    assert command[command.index("--proxy-uid") + 1] == "0"
    dockerfile = (REPO_ROOT / "docker/Dockerfile.principal-issuer").read_text()
    assert (
        "developer-principal-grants.ci-bus.yaml /app/developer-principal-grants.ci-bus.yaml"
        in dockerfile
    )
    assert "check --grants /app/developer-principal-grants.ci-bus.yaml" in dockerfile


def test_the_instance_authenticates_as_the_ci_bus_superuser_not_the_dev_lanes() -> None:
    env = _load_yaml(COMPOSE)["services"]["principal-issuer-ci-bus"]["environment"]
    assert env["RPK_USER"].startswith("${CI_BUS_KAFKA_SASL_USERNAME:?")
    assert env["RPK_PASS"].startswith("${CI_BUS_KAFKA_SASL_PASSWORD:?")
    assert "DEV_KAFKA" not in " ".join(str(v) for v in env.values())


def test_the_onboarding_facts_name_the_ci_bus_issuer() -> None:
    facts = _facts()
    assert facts["ci_bus_lane"] == yaml.safe_load(CI_GRANTS.read_text())["lane"]
    assert facts["ci_bus_issuer_url"].startswith("https://")
    assert facts["ci_bus_issuer_url"] != facts["principal_issuer_url"]
