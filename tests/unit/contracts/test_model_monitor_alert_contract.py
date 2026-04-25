# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
import pytest
from pydantic import ValidationError

from omnibase_infra.models.contracts.model_monitor_alert_contract import (
    ModelMonitorAlertContract,
)


def test_monitor_alert_contract_round_trips() -> None:
    raw = {
        "name": "monitor-alerts",
        "contract_version": "1.0.0",
        "description": "Monitor alert contract",
        "event_bus": {"publish_topics": ["onex.evt.omnibase_infra.monitor-alert.v1"]},
    }
    m = ModelMonitorAlertContract.model_validate(raw)
    assert m.name == "monitor-alerts"
    assert len(m.event_bus["publish_topics"]) == 1


def test_monitor_alert_contract_dict_version() -> None:
    raw = {
        "name": "monitor_alert_emitter",
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "description": "Declares the Kafka topics emitted by monitor_logs.py",
        "event_bus": {
            "publish_topics": ["onex.evt.omnibase-infra.monitor-alert-detected.v1"]
        },
    }
    m = ModelMonitorAlertContract.model_validate(raw)
    assert isinstance(m.contract_version, dict)
    assert (
        m.event_bus["publish_topics"][0]
        == "onex.evt.omnibase-infra.monitor-alert-detected.v1"
    )


def test_monitor_alert_contract_rejects_missing_name() -> None:
    with pytest.raises(ValidationError):
        ModelMonitorAlertContract.model_validate({"contract_version": "1.0.0"})


def test_monitor_alert_contract_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelMonitorAlertContract.model_validate(
            {"name": "x", "contract_version": "1.0.0", "unknown_field": "bad"}
        )
