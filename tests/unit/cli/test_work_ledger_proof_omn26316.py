# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev roll's post-roll ledger proof answers as the deploy agent deploys it (OMN-26316).

Two readings of dev, 2026-11-23, both failed rolls of the dev runtime after its
verification had passed:

* job 43b9fb72 (build of 5a07ec39c, accepted 00:24:12Z, failed 00:41:24Z) and job
  ae9ceb3d (build of 09347b4ab, accepted 23:38:59Z, failed 23:59:51Z) each ended in
  ``DevBootFailedError: work ledger: ledger exited 1`` and the boot guard restored the
  previous image. Reproduced with the agent's own process environment on omni-201: the
  deploy agent holds ``KAFKA_SASL_MECHANISM=SCRAM-SHA-256`` and
  ``KAFKA_SECURITY_PROTOCOL=SASL_PLAINTEXT`` but keeps its SASL principal under the
  ``KAFKA_SASL_ENV_PREFIX`` names, so ``ModelKafkaEventBusConfig.default()`` refused the
  half-set pair before the lane's stored identity was applied.
* with that refusal removed the same read answered in 4.9 s, 8.7 s, 16.8 s and 19.5 s
  and timed out once at the declared 10 s bound.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from omnibase_infra.cli import cli_ledger
from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "contracts"
    / "work_ledger_query_client.contract.yaml"
)
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "omn26316" / "ledger_proof_latency.json"


@pytest.fixture
def deploy_agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kafka names the dev deploy agent's process holds on omni-201.

    The mechanism and protocol are set; the principal is not under the generic names.
    """
    for name in list(os.environ):
        if name.startswith("KAFKA_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.test:19092")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_PLAINTEXT")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256")
    monkeypatch.setenv("KAFKA_SASL_ENV_PREFIX", "DEV_")


@pytest.fixture
def stored_lane_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dev lane's declared address and the machine's stored identity."""
    declared = Path("lane-declaration.yaml")
    selection = ModelDelegateLaneSelection(
        lane="dev",
        bootstrap_servers="lane.test:19092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        declared_in=declared,
    )
    transport = ModelLaneClientTransport(
        lane="dev",
        bootstrap_servers="lane.test:19092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_username="stored-principal",
        sasl_password=SecretStr("stored-secret"),
        declared_in=declared,
    )
    monkeypatch.setattr(
        cli_ledger, "resolve_lane_binding", lambda lane: (selection, transport)
    )


def test_ledger_proof_transport_is_built_from_the_lane_not_a_half_set_env(
    deploy_agent_environment: None, stored_lane_identity: None
) -> None:
    """The deploy agent's SASL mechanism without a generic principal must not refuse the read."""
    config = cli_ledger.resolve_work_ledger_query_transport("dev")

    assert config.bootstrap_servers == "lane.test:19092"
    assert config.security_protocol == "SASL_PLAINTEXT"
    assert config.sasl_mechanism == "SCRAM-SHA-256"
    assert config.sasl_plain_username == "stored-principal"


def test_ledger_proof_transport_still_takes_other_environment_overrides(
    deploy_agent_environment: None,
    stored_lane_identity: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the lane's auth fields come from the lane; every other override still applies."""
    monkeypatch.setenv("KAFKA_TIMEOUT_SECONDS", "45")

    config = cli_ledger.resolve_work_ledger_query_transport("dev")

    assert config.timeout_seconds == 45


def test_ledger_proof_bound_covers_the_measured_answer_latency() -> None:
    """The declared bound leaves headroom over the slowest recorded answer (OMN-26316)."""
    measured = json.loads(FIXTURE.read_text(encoding="utf-8"))["answered_seconds"]
    bound = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["post_roll_proof"][
        "timeout_s"
    ]

    assert len(measured) >= 4
    assert bound >= 1.5 * max(measured), (
        f"declared bound {bound}s is under 1.5x the slowest recorded answer "
        f"({max(measured)}s)"
    )
