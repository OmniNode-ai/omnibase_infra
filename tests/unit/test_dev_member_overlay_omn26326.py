# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26326: the dev-member standby overlay declares the SCM consumer set.

``docker/docker-compose.dev-member.yml`` is the declared home of the standby that
answers the dev lane's GitHub write and read-gateway command topics while the dev
runtime is down. It reuses the ``effects`` role and ``ONEX_RUNTIME_NODE_SELECTION``
(the mechanism the claude hook-events carrier uses), joins the dev runtime's
consumer groups through the same ``KAFKA_INSTANCE_ID``, and carries no credential
value and no host address.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_core.constants.constants_runtime_profiles import (
    REGISTERED_RUNTIME_PROFILES,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "docker" / "docker-compose.dev-member.yml"
INFRA = ROOT / "docker" / "docker-compose.infra.yml"

SCM_NODES = (
    "node_github_repo_write_effect",
    "node_github_repo_gateway_effect",
)
SERVICE = "omninode-runtime"


def _load(path: Path) -> dict[str, object]:
    # compose's ``!override`` and merge tags only change how overlays merge.
    text = re.sub(r"!override[ \t]*|!!merge[ \t]*", "", path.read_text("utf-8"))
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def _environment() -> dict[str, str]:
    services = _load(OVERLAY)["services"]
    assert isinstance(services, dict)
    environment = services[SERVICE]["environment"]
    assert isinstance(environment, dict)
    return {key: str(value) for key, value in environment.items()}


def test_overlay_declares_exactly_one_runtime_service() -> None:
    assert OVERLAY.is_file(), "docker/docker-compose.dev-member.yml is not declared"
    services = _load(OVERLAY)["services"]
    assert isinstance(services, dict)
    assert list(services) == [SERVICE]
    # The runtime and nothing else: no depends_on edge to a dev-lane service.
    assert "depends_on" not in services[SERVICE]


def test_runtime_loads_only_the_effects_role_over_the_two_scm_nodes() -> None:
    environment = _environment()
    assert environment["RUNTIME_PROFILE"] == "effects"
    assert "effects" in REGISTERED_RUNTIME_PROFILES
    selection = [
        name.strip() for name in environment["ONEX_RUNTIME_NODE_SELECTION"].split(",")
    ]
    assert selection == list(SCM_NODES)
    # No folded role: the standby serves the two nodes and no other.
    assert "ONEX_RUNTIME_FOLDED_PROFILES" not in environment


def test_runtime_shares_the_dev_runtime_consumer_groups() -> None:
    infra_services = _load(INFRA)["services"]
    assert isinstance(infra_services, dict)
    dev_instance = infra_services["omninode-runtime"]["environment"][
        "KAFKA_INSTANCE_ID"
    ]
    assert dev_instance == "runtime-main"
    assert _environment()["KAFKA_INSTANCE_ID"] == dev_instance


def test_runtime_authenticates_as_the_member_principal_from_the_env_file() -> None:
    environment = _environment()
    # The principal name comes from the standby host's env file, never the file.
    assert re.fullmatch(
        r"\$\{KAFKA_SASL_USERNAME:\?[^}]+\}", environment["KAFKA_SASL_USERNAME"]
    )
    assert environment["KAFKA_SECURITY_PROTOCOL"] == "SASL_PLAINTEXT"
    assert environment["KAFKA_SASL_MECHANISM"] == "SCRAM-SHA-256"


def test_credentials_are_required_interpolations_never_values() -> None:
    environment = _environment()
    for name in ("KAFKA_SASL_PASSWORD", "GITHUB_TOKEN", "GH_TOKEN"):
        assert re.fullmatch(r"\$\{" + name + r":\?[^}]+\}", environment[name]), name


def test_overlay_names_no_host_address_and_no_dev_lane_dependency() -> None:
    text = OVERLAY.read_text("utf-8")
    code = re.sub(r"#.*", "", text).replace("127.0.0.1", "")
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", code)
    environment = _environment()
    for forbidden in ("POSTGRES_HOST", "VALKEY_HOST", "OMNIBASE_INFRA_DB_URL"):
        assert forbidden not in environment
    # Broker address, listener allowlist and image come from the standby host.
    for name in (
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_BROKER_ALLOWLIST",
        "ONEX_RUNTIME_LANE",
    ):
        assert ":?" in environment[name], name
    services = _load(OVERLAY)["services"]
    assert isinstance(services, dict)
    assert ":?" in str(services[SERVICE]["image"])


def test_active_runtime_packages_come_from_the_policy_file() -> None:
    """The set is the runtime policy file's (OMN-26405), never a literal list here."""
    value = _environment()["ONEX_ACTIVE_RUNTIME_PACKAGES"]
    assert value.startswith("${ONEX_ACTIVE_RUNTIME_PACKAGES:?"), value


def test_health_port_binds_loopback_only() -> None:
    services = _load(OVERLAY)["services"]
    assert isinstance(services, dict)
    ports = services[SERVICE]["ports"]
    assert ports
    assert all(str(port).startswith("127.0.0.1:") for port in ports)


def test_runtime_reads_no_lane_hotload_overlay() -> None:
    """The standby must not inherit the dev lane's contract hot-load row.

    The standby declares the dev lane (``ONEX_RUNTIME_LANE``) so it shares the dev
    runtime's groups. A lane's ``runtime_contract_hotload`` overlay row, read from
    ``ONEX_SKILL_OVERLAY_ROOTS``, turns the node-registration listener on. The
    standby member principal has no READ on that topic, so the listener's subscribe
    is refused on every retry, ``start()`` never reaches ``is_running`` and
    ``/health`` answers 503 with ``startup_in_progress`` while both SCM consumers
    answer. The standby loads exactly two contracts at boot and hot-loads nothing,
    so it names no overlay root at all.
    """
    environment = _environment()
    assert "ONEX_SKILL_OVERLAY_ROOTS" not in environment
    services = _load(OVERLAY)["services"]
    assert isinstance(services, dict)
    targets = [
        str(volume).split(":")[1]
        for volume in services[SERVICE]["volumes"]
        if ":" in str(volume)
    ]
    assert not [t for t in targets if t.startswith("/etc/onex/skill-overlays")]
