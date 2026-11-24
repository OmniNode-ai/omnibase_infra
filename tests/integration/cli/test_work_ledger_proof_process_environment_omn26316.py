# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The post-roll ledger proof builds its transport inside the deploy agent's process (OMN-26316).

The dev deploy agent runs the proof as a child process whose environment holds
``KAFKA_SASL_MECHANISM`` and ``KAFKA_SECURITY_PROTOCOL`` while its SASL principal sits
under the ``KAFKA_SASL_ENV_PREFIX`` names. The unit test sets that environment on the
test process; this test starts a separate Python process with exactly that environment
and reads the transport the ledger command resolves, so the result does not depend on
anything the test runner left in ``os.environ``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

_CHILD = """
import json
from pathlib import Path

from pydantic import SecretStr

from omnibase_infra.cli import cli_ledger
from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)

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
cli_ledger.resolve_lane_binding = lambda lane: (selection, transport)
config = cli_ledger.resolve_work_ledger_query_transport("dev")
print(
    json.dumps(
        {
            "bootstrap_servers": config.bootstrap_servers,
            "security_protocol": config.security_protocol,
            "sasl_mechanism": config.sasl_mechanism,
            "sasl_plain_username": config.sasl_plain_username,
        }
    )
)
"""


def _deploy_agent_process_environment() -> dict[str, str]:
    """This process's environment with the half-set kafka names the agent holds."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("KAFKA_")
    }
    environment.update(
        {
            "KAFKA_BOOTSTRAP_SERVERS": "broker.test:19092",
            "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
            "KAFKA_SASL_MECHANISM": "SCRAM-SHA-256",
            "KAFKA_SASL_ENV_PREFIX": "DEV_",
        }
    )
    return environment


def test_proof_transport_resolves_in_a_process_holding_the_agents_half_set_names() -> (
    None
):
    """The child exits 0 and its transport carries the lane's address and stored principal."""
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=_deploy_agent_process_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "bootstrap_servers": "lane.test:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": "SCRAM-SHA-256",
        "sasl_plain_username": "stored-principal",
    }
