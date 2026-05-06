# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for production runtime Kafka consumer identity."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

COMPOSE_PATH = Path(__file__).parents[2] / "docker" / "docker-compose.infra.yml"


def test_production_runtime_services_have_distinct_kafka_instance_ids() -> None:
    data = yaml.safe_load(COMPOSE_PATH.read_text())
    assert isinstance(data, dict)

    services = data["services"]
    expected = {
        "omninode-runtime": "runtime-main",
        "runtime-effects": "runtime-effects",
        "runtime-worker": "runtime-worker",
    }

    observed = {
        service_name: services[service_name]["environment"]["KAFKA_INSTANCE_ID"]
        for service_name in expected
    }

    assert observed == expected
    assert len(set(observed.values())) == len(expected)
