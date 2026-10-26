# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21430: the three-broker dev bus compose and its recorded scratch proof.

The dev bus is one Redpanda broker on h201 that crashed three times on
2026-10-25/26. ``docker/docker-compose.dev-bus-ha.yml`` runs one member of a
three-broker cluster per host. These tests read the committed files only; no
broker is touched. They pin the properties the design rests on: three members
on three distinct hosts, replication factor 3 for new and internal topics, a
12G arena with swap refused, no silent address default, and the scratch proof's
recorded outcome.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
COMPOSE = DOCKER_DIR / "docker-compose.dev-bus-ha.yml"
HA_DIR = DOCKER_DIR / "redpanda" / "dev-bus-ha"
HOSTS = HA_DIR / "hosts.tsv"
BOOTSTRAP = HA_DIR / ".bootstrap.yaml"
PROOF = HA_DIR / "evidence" / "omn-21430-scratch-proof.txt"


def _hosts() -> list[dict[str, str]]:
    rows = [
        line
        for line in HOSTS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    header = [
        "name",
        "os",
        "tailnet_ip",
        "kafka",
        "internal",
        "rpc",
        "admin",
        "memory",
        "mem_limit",
        "role",
    ]
    return [dict(zip(header, row.split("\t"), strict=True)) for row in rows]


def _service() -> dict[str, Any]:
    doc = yaml.safe_load(COMPOSE.read_text())
    service: dict[str, Any] = doc["services"]["redpanda"]
    return service


def _env(host: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (HA_DIR / f"{host}.env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            env[key] = value
    return env


def test_three_members_on_three_hosts_not_all_on_201() -> None:
    hosts = _hosts()
    names = [h["name"] for h in hosts]
    assert len(names) == 3
    assert len(set(names)) == 3
    assert names.count("h201") == 1
    for host in names:
        assert (HA_DIR / f"{host}.env.example").is_file()


def test_every_member_joins_through_the_h201_rpc_address() -> None:
    h201 = next(h for h in _hosts() if h["name"] == "h201")
    seed = f"{h201['tailnet_ip']}:{h201['rpc']}"
    for host in ("h201", "h202", "h200"):
        assert _env(host)["DEV_BUS_HA_SEEDS"] == seed


def test_h201_member_keeps_the_dev_broker_volume() -> None:
    infra = yaml.safe_load((DOCKER_DIR / "docker-compose.infra.yml").read_text())
    dev_volume = infra["volumes"]["redpanda_data"]["name"]
    assert _env("h201")["DEV_BUS_HA_DATA_VOLUME"] == dev_volume
    assert "DEV_BUS_HA_DATA_VOLUME" not in _env("h202")
    assert "DEV_BUS_HA_DATA_VOLUME" not in _env("h200")


def test_replication_factor_three_for_new_and_internal_topics() -> None:
    bootstrap = yaml.safe_load(BOOTSTRAP.read_text())
    assert bootstrap["default_topic_replications"] == 3
    assert bootstrap["internal_topic_replication_factor"] == 3


def test_arena_is_twelve_gig_and_swap_is_refused() -> None:
    service = _service()
    command = " ".join(service["command"])
    assert "--memory ${DEV_BUS_HA_MEMORY:-12G}" in command
    assert "--smp 1" in command
    assert service["mem_limit"] == service["memswap_limit"]
    assert "--mode" not in command


def test_addresses_have_no_silent_default() -> None:
    command = " ".join(_service()["command"])
    assert re.search(r"\$\{DEV_BUS_HA_ADVERTISE_HOST:\?", command)
    assert re.search(r"\$\{DEV_BUS_HA_SEEDS:\?", command)
    assert ":-localhost" not in command


def test_admin_api_is_loopback_only() -> None:
    admin = [p for p in _service()["ports"] if p.endswith(":9644")]
    assert admin == ["127.0.0.1:${DEV_BUS_HA_ADMIN_PORT:-9644}:9644"]


def test_recorded_scratch_proof_outcome() -> None:
    proof = PROOF.read_text()
    assert "topics on broker: 377" in proof
    assert (
        "onex.cmd/evt topics: 377; topics (incl. __consumer_offsets) with RF below 3: 0"
    ) in proof
    one_down = proof.split("== 4.")[1].split("== 5.")[0]
    assert "IS-ALIVE" in one_down
    assert "false" in one_down
    assert "Produced to partition 0 at offset 14" in one_down
    assert re.search(r"baseline-comparison-request\.v1\s+0\s+15\s", one_down)
    assert "negative control held" in proof
    assert proof.rstrip().endswith("cleanup done")
