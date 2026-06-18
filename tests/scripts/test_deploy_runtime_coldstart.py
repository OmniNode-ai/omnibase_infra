# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression coverage for the OMN-13220 cold-start crash-loop fixes.

Two distinct fresh-DB / cold-lane failure modes are guarded here:

1. Missing intelligence-migration. The compose file gates omninode-runtime on
   ``intelligence-migration: condition: service_completed_successfully``, but
   restart_services() uses ``up -d --no-deps`` which bypasses depends_on. On a
   fresh-DB lane that left public.db_metadata for omniintelligence unstamped, so
   the runtime crash-looped. deploy-runtime.sh's migration preflight must run
   intelligence-migration (and ``docker wait`` on it as a one-shot).

2. Cold-start consumer-timeout crash-loop. On a fully-cold lane the kernel joins
   a consumer group per subscribed topic; with 1300+ topics on a freshly
   provisioned broker the default 30s per-consumer KAFKA_TIMEOUT_SECONDS blew on
   the slow group-coordinator tail and the kernel recycled before reaching
   healthy. The deploy must (a) apply the broker partition cap before the kernel
   provisions topics (the ``--no-deps`` restart bypasses the compose
   redpanda-partition-cap gate) and (b) raise KAFKA_TIMEOUT_SECONDS for the
   restart-driven boot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"


def _deploy_script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _deploy_script_noncomment() -> str:
    """deploy-runtime.sh with comment-only lines stripped.

    Assertions about *active* behavior must not be satisfied by a comment that
    merely mentions the token, so the active-code checks run against this view.
    """
    lines = [
        line
        for line in _deploy_script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fix 1: intelligence-migration in the migration preflight
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_intelligence_migration_in_runtime_migration_services() -> None:
    """RUNTIME_MIGRATION_SERVICES must include intelligence-migration (fresh-DB)."""
    text = _deploy_script_noncomment()
    match = re.search(
        r"RUNTIME_MIGRATION_SERVICES=\((?P<body>.*?)\)",
        text,
        re.DOTALL,
    )
    assert match is not None, "RUNTIME_MIGRATION_SERVICES array not found"
    services = match.group("body").split()
    assert "intelligence-migration" in services, (
        "intelligence-migration missing from RUNTIME_MIGRATION_SERVICES. The "
        "compose file gates omninode-runtime on it via "
        "service_completed_successfully, but the --no-deps restart bypasses "
        "depends_on, so the preflight must run it or fresh-DB lanes crash-loop "
        "(OMN-13220)."
    )
    # Order must keep migration-gate last: it stamps migrations_complete after
    # both schema migrations have applied.
    assert services.index("intelligence-migration") < services.index(
        "migration-gate"
    ), "intelligence-migration must run before migration-gate stamps completion."


@pytest.mark.unit
def test_intelligence_migration_is_waited_on_as_oneshot() -> None:
    """The preflight must docker wait on intelligence-migration completion."""
    text = _deploy_script_noncomment()
    match = re.search(
        r"RUNTIME_MIGRATION_ONESHOTS=\((?P<body>.*?)\)",
        text,
        re.DOTALL,
    )
    assert match is not None, "RUNTIME_MIGRATION_ONESHOTS array not found"
    oneshots = match.group("body").split()
    assert "forward-migration" in oneshots
    assert "intelligence-migration" in oneshots
    # migration-gate is a long-running healthcheck keepalive, not a one-shot; a
    # docker wait on it would block the deploy forever.
    assert "migration-gate" not in oneshots, (
        "migration-gate is a keepalive, not a one-shot; it must not be in the "
        "docker-wait set or the deploy hangs."
    )
    # The wait loop must actually consult the one-shot set.
    assert "RUNTIME_MIGRATION_ONESHOTS" in text
    assert "docker wait" in text


# ---------------------------------------------------------------------------
# Fix 2a: broker partition-cap warmup before the kernel boots
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_warm_broker_runs_before_migration_and_restart() -> None:
    """warm_broker_topic_provisioning must run before migrations and restart.

    The ``--no-deps`` runtime restart bypasses the compose
    redpanda-partition-cap depends_on gate, so the deploy must apply the cap
    explicitly before the kernel provisions 1300+ topics on a cold broker.
    """
    text = _deploy_script_text()
    warm_idx = text.find('warm_broker_topic_provisioning "${deploy_target}"')
    migration_idx = text.find('run_runtime_migration_preflight "${deploy_target}"')
    restart_idx = text.find('restart_services "${deploy_target}"')

    assert warm_idx != -1, "warm_broker_topic_provisioning is not called in main()"
    assert migration_idx != -1, "run_runtime_migration_preflight is not called"
    assert restart_idx != -1, "restart_services is not called"
    assert warm_idx < migration_idx < restart_idx, (
        "Order must be warmup -> migration preflight -> restart so the broker "
        "partition cap and schema are ready before the kernel boots (OMN-13220)."
    )


@pytest.mark.unit
def test_warm_broker_applies_partition_cap_and_waits() -> None:
    """The warmup must bring up the partition cap and verify it completed."""
    text = _deploy_script_text()
    assert "warm_broker_topic_provisioning()" in text
    assert 'BROKER_PARTITION_CAP_SERVICE="redpanda-partition-cap"' in text
    # The broker must be brought to a healthy state before the cap rpk calls.
    assert "--wait" in text
    # The cap is a one-shot; the deploy must fail if it does not exit 0.
    assert 'cap_container="${compose_project}-${BROKER_PARTITION_CAP_SERVICE}"' in text
    assert "did not complete successfully." in text


# ---------------------------------------------------------------------------
# Fix 2b: raised per-consumer Kafka start timeout for the cold boot
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cold_start_kafka_timeout_exported_before_restart() -> None:
    """The deploy must export a raised KAFKA_TIMEOUT_SECONDS for the cold boot."""
    text = _deploy_script_text()
    assert "COLD_START_KAFKA_TIMEOUT_SECONDS" in text
    assert 'export KAFKA_TIMEOUT_SECONDS="${cold_start_timeout}"' in text
    # Active (non-comment) code must perform the export, not just mention it.
    assert "export KAFKA_TIMEOUT_SECONDS" in _deploy_script_noncomment()


@pytest.mark.unit
def test_cold_start_timeout_clamped_to_config_bound() -> None:
    """The exported timeout must be clamped to the config max (le=300)."""
    text = _deploy_script_noncomment()
    # Reject non-integers and clamp the upper bound so the kernel never receives
    # a KAFKA_TIMEOUT_SECONDS its config validation rejects.
    assert "must be a positive integer" in _deploy_script_text()
    assert "cold_start_timeout=300" in text
    assert "300" in text


@pytest.mark.unit
def test_compose_runtime_env_reads_kafka_timeout_seconds() -> None:
    """x-runtime-env must read KAFKA_TIMEOUT_SECONDS so the export reaches containers."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    # An exported shell var only reaches a container if the service env block
    # interpolates it; x-runtime-env is the shared anchor for runtime services.
    assert "KAFKA_TIMEOUT_SECONDS: ${KAFKA_TIMEOUT_SECONDS:-30}" in text, (
        "x-runtime-env must declare KAFKA_TIMEOUT_SECONDS with a 30s default so "
        "the deploy-script export propagates to runtime containers while a plain "
        "`docker compose up` keeps the prior steady-state value (OMN-13220)."
    )
