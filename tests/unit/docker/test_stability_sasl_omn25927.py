# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25927 (design OMN-23363): the stability lane's broker requires SASL.

The stability overlay declares, for its own broker only, the dev lane's
OMN-18012 shape: a one-shot that creates one SCRAM principal per stability
service plus ``stability-h051``, a one-shot that grants the service principals
``superusers`` and then flips ``enable_sasl``, and SASL client settings on every
service that connects to the broker. Each service authenticates as a principal
named by the service, with its password read from a fail-closed reference to
``STABILITY_TEST_KAFKA_SASL_PASSWORD_<SERVICE>``.

No docker required -- these read the committed compose files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
BASE = DOCKER_DIR / "docker-compose.infra.yml"

USER_SERVICE = "redpanda-scram-user"
FLIP_SERVICE = "redpanda-sasl-enable"
PARTITION_CAP = "redpanda-partition-cap"
# The refresh health gate's own login (verify_stability_refresh.py).
GATE_PRINCIPAL = "stability-refresh-gate"
H051_PRINCIPAL = "stability-h051"

# Every stability service that speaks Kafka and runs in the lane's active
# profiles, read from the live lane on 2026-11-19 (container env carrying
# KAFKA_BOOTSTRAP_SERVERS or KAFKA_BROKERS in the lane's compose project).
# runtime-effects is folded into the runtime (OMN-21288); the observability
# consumers and the contract resolver are profile-disabled on this lane.
STABILITY_KAFKA_CLIENTS = frozenset(
    {
        "omninode-runtime",
        "runtime-worker",
        "projection-api",
        "projection-tenant-registry-writer",
        "projection-delegation-writer",
        "projection-registration-writer",
        "projection-savings-writer",
        "projection-tenant-credentials-writer",
        "projection-live-events-writer",
        "tenant-projection-writer",
    }
)

_REFERENCE = re.compile(r"^\$\{(?P<var>[A-Z][A-Z0-9_]*):\?[^}]+\}$")
# Compose's merge directives; safe_load reads the documents once they are removed.
_COMPOSE_TAGS = re.compile(r"!(override|reset)\b")


def _load(path: Path) -> dict[str, Any]:
    text = _COMPOSE_TAGS.sub("", path.read_text(encoding="utf-8"))
    parsed: dict[str, Any] = yaml.safe_load(text)
    return parsed


def _stability_overlay() -> Path:
    """The lane overlay: the stability compose file that declares the runtime."""
    for path in sorted(DOCKER_DIR.glob("docker-compose.stability*.yml")):
        if "omninode-runtime" in (_load(path).get("services") or {}):
            return path
    raise AssertionError("no stability compose overlay declares omninode-runtime")


def _password_var(service: str) -> str:
    return "STABILITY_TEST_KAFKA_SASL_PASSWORD_" + service.upper().replace("-", "_")


def _referenced_var(value: object) -> str:
    match = _REFERENCE.match(str(value))
    assert match, (
        f"expected a fail-closed reference, got a literal or fallback: {value!r}"
    )
    return match.group("var")


def _principal_pairs(env: dict[str, Any], key: str) -> dict[str, str]:
    return dict(pair.split("=", 1) for pair in str(env[key]).split())


@pytest.fixture(scope="module")
def services() -> dict[str, Any]:
    loaded: dict[str, Any] = _load(_stability_overlay())["services"]
    return loaded


def test_stability_sasl_one_shots_are_declared_on_the_stability_overlay(
    services: dict[str, Any],
) -> None:
    for name in (USER_SERVICE, FLIP_SERVICE):
        assert name in services, f"{name} is missing from the stability overlay"


def test_stability_sasl_flip_waits_for_the_principals(services: dict[str, Any]) -> None:
    depends = services[FLIP_SERVICE]["depends_on"]
    assert depends[USER_SERVICE]["condition"] == "service_completed_successfully"
    script = "\n".join(services[FLIP_SERVICE]["command"])
    assert script.index("superusers") < script.index("enable_sasl true"), (
        "the superuser grant must land before the flip, or authorization locks every client out"
    )
    assert "UNAUTHENTICATED" in script
    assert "positive control" in script


def test_stability_sasl_every_kafka_client_authenticates_as_its_own_principal(
    services: dict[str, Any],
) -> None:
    for name in sorted(STABILITY_KAFKA_CLIENTS):
        env = services[name]["environment"]
        assert env.get("KAFKA_SECURITY_PROTOCOL") == "SASL_PLAINTEXT", name
        assert env.get("KAFKA_SASL_MECHANISM") == "SCRAM-SHA-256", name
        assert env.get("KAFKA_SASL_USERNAME") == name, name
        assert _referenced_var(env.get("KAFKA_SASL_PASSWORD")) == _password_var(name), (
            name
        )


def test_stability_sasl_principal_set_matches_the_clients(
    services: dict[str, Any],
) -> None:
    env = services[USER_SERVICE]["environment"]
    superusers = _principal_pairs(env, "STABILITY_TEST_SUPERUSER_PRINCIPALS")
    expected = {name: _password_var(name) for name in STABILITY_KAFKA_CLIENTS}
    expected[PARTITION_CAP] = _password_var(PARTITION_CAP)
    expected[GATE_PRINCIPAL] = _password_var(GATE_PRINCIPAL)
    assert superusers == expected
    acl_principals = _principal_pairs(env, "STABILITY_TEST_ACL_PRINCIPALS")
    assert acl_principals == {H051_PRINCIPAL: _password_var(H051_PRINCIPAL)}
    assert H051_PRINCIPAL not in superusers, (
        "stability-h051 holds ACL rows, never a superuser grant"
    )
    for var in [*superusers.values(), *acl_principals.values()]:
        assert _referenced_var(env[var]) == var
    assert services[FLIP_SERVICE]["environment"] == env


def test_stability_sasl_partition_cap_runs_after_the_flip_as_its_principal(
    services: dict[str, Any],
) -> None:
    cap = services[PARTITION_CAP]
    assert cap["environment"]["RPK_USER"] == PARTITION_CAP
    assert _referenced_var(cap["environment"]["RPK_PASS"]) == _password_var(
        PARTITION_CAP
    )
    assert (
        cap["depends_on"][FLIP_SERVICE]["condition"] == "service_completed_successfully"
    )


def test_stability_sasl_never_reaches_the_base() -> None:
    base = _load(BASE)["services"]
    assert USER_SERVICE not in base
    assert FLIP_SERVICE not in base
    assert "STABILITY_TEST_KAFKA_SASL_PASSWORD_" not in BASE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The refresh health gate reads the Kafka API as its own principal.
# ---------------------------------------------------------------------------
# After the flip, an unauthenticated `rpk topic list` or `rpk group list` is
# refused, so a refresh whose gate reads anonymously fails and rolls back with
# SASL left on. The gate authenticates as `stability-refresh-gate`, a declared
# superuser principal; the refresh exports the operator env, and the gate hands
# docker the password's NAME (RPK_PASS), never its value.

RUNTIME_BUILD = DOCKER_DIR.parent / "scripts" / "runtime_build"


def _verify_module() -> Any:
    import importlib.util
    import sys

    if str(RUNTIME_BUILD) not in sys.path:
        sys.path.insert(0, str(RUNTIME_BUILD))
    spec = importlib.util.spec_from_file_location(
        "verify_stability_refresh_omn25927",
        RUNTIME_BUILD / "verify_stability_refresh.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stability_sasl_gate_principal_is_a_declared_superuser(
    services: dict[str, Any],
) -> None:
    env = services[USER_SERVICE]["environment"]
    superusers = _principal_pairs(env, "STABILITY_TEST_SUPERUSER_PRINCIPALS")
    assert superusers.get(GATE_PRINCIPAL) == _password_var(GATE_PRINCIPAL)
    assert _referenced_var(env[_password_var(GATE_PRINCIPAL)]) == _password_var(
        GATE_PRINCIPAL
    )


READERS = [
    ("_get_total_partitions", "NAME PARTITIONS REPLICAS\nt 1 1\n"),
    ("list_consumer_groups", "BROKER GROUP STATE\n0 g Stable\n"),
    ("check_cluster_health", "Healthy: true\n"),
]
# A sentinel that only proves the value never reaches argv; not a credential.
SENTINEL = "omn25927-sentinel-not-a-secret"


def _commands(reader: str, stdout: str) -> list[list[str]]:
    import subprocess

    seen: list[list[str]] = []

    def runner(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    getattr(_verify_module(), reader)("broker", runner=runner)
    assert seen, reader
    return seen


@pytest.mark.parametrize(("reader", "stdout"), READERS)
def test_stability_sasl_gate_reads_the_kafka_api_as_its_principal(
    reader: str, stdout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    monkeypatch.setenv(_password_var(GATE_PRINCIPAL), SENTINEL)
    monkeypatch.delenv("RPK_PASS", raising=False)
    for cmd in _commands(reader, stdout):
        assert cmd[:2] == ["docker", "exec"], cmd
        assert cmd[2 : cmd.index("broker")] == [
            "-e",
            f"RPK_USER={GATE_PRINCIPAL}",
            "-e",
            "RPK_SASL_MECHANISM=SCRAM-SHA-256",
            "-e",
            "RPK_PASS",
        ], cmd
        assert SENTINEL not in " ".join(cmd), "a password reached the command line"
    assert os.environ["RPK_PASS"] == SENTINEL


@pytest.mark.parametrize(("reader", "stdout"), READERS)
def test_stability_sasl_gate_stays_anonymous_without_a_password(
    reader: str, stdout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_password_var(GATE_PRINCIPAL), raising=False)
    for cmd in _commands(reader, stdout):
        assert cmd[:3] == ["docker", "exec", "broker"], cmd
