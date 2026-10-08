# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the lag probe reads a real broker reply without a credential on argv.

The recorded reply is the live ``rpk group describe`` output of the dev-lane broker,
captured through the same ``docker exec -e RPK_USER -e RPK_PASS -e RPK_SASL_MECHANISM``
shape the probe now builds. During that capture a scan of every process argv on the host
found no line holding the SASL password, and the same call without the ``-e`` flags was
refused by the broker, so the environment is what authenticated it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import ModelBrokerAccess, read_group_total_lag

pytestmark = pytest.mark.unit

_SYNTHETIC_PASSWORD = "synthetic-credential-never-sent-anywhere"


@pytest.mark.live_contact("tests/ci/fixtures/omn17427_rpk_group_describe.json")
def test_the_recorded_broker_reply_parses_and_the_credential_stays_off_argv(
    recorded_response: dict[str, Any],
) -> None:
    reply = recorded_response["response"]["stdout"]
    seen: dict[str, Any] = {}

    def runner(
        argv: Sequence[str], *, timeout: float, env: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        seen["argv"] = list(argv)
        seen["env"] = dict(env or {})
        return subprocess.CompletedProcess(list(argv), 0, reply, "")

    access = ModelBrokerAccess(
        container="omnibase-infra-redpanda",
        brokers="redpanda:9092",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_username="synthetic-user",
        sasl_password=_SYNTHETIC_PASSWORD,
    )

    assert (
        read_group_total_lag(access, "agent-observability-postgres", runner=runner) == 0
    )
    assert _SYNTHETIC_PASSWORD not in " ".join(seen["argv"])
    assert "synthetic-user" not in " ".join(seen["argv"])
    assert seen["env"]["RPK_PASS"] == _SYNTHETIC_PASSWORD
