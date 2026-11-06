# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve embedded workspace config and read the selected live broker."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from confluent_kafka.admin import AdminClient

from omnibase_infra.cli.cli_delegate import resolve_default_bus
from omnibase_infra.cli.delegate_lane import resolve_lane_selection_at
from omnibase_infra.cli.delegate_lane_credentials import (
    resolve_lane_client_transport_for,
)

pytestmark = pytest.mark.integration


def test_workspace_config_selects_live_broker() -> None:
    """Local Done probe, selected explicitly with this file's path."""
    workspace = Path(os.environ["OMNI_HOME"])
    resolved = resolve_default_bus(workspace_root=workspace)
    assert resolved.bus == "kafka", resolved.reason
    assert "workspace tier-1" in resolved.reason, resolved.reason
    assert resolved.lane, "Workspace config must name its shared bus lane"
    target = resolve_lane_selection_at(resolved.lane, workspace)
    transport = resolve_lane_client_transport_for(
        lane_target=target, onex_home=Path.home() / ".onex", environ=os.environ
    )
    assert transport is not None
    client_config: dict[str, str | int | float | bool] = {
        "bootstrap.servers": transport.bootstrap_servers,
        "security.protocol": transport.security_protocol,
    }
    if transport.is_sasl:
        assert transport.sasl_password is not None
        assert transport.sasl_username is not None
        assert transport.sasl_mechanism is not None
        client_config.update(
            {
                "sasl.mechanism": transport.sasl_mechanism,
                "sasl.username": transport.sasl_username,
                "sasl.password": transport.sasl_password.get_secret_value(),
            }
        )
    metadata = AdminClient(client_config).list_topics(timeout=15)
    assert metadata.brokers, "The selected live broker returned no brokers"
    assert metadata.cluster_id, "The selected live broker returned no cluster id"
    print(
        json.dumps(
            {
                "lane": resolved.lane,
                "provenance": resolved.reason,
                "cluster_id": metadata.cluster_id,
                "broker_count": len(metadata.brokers),
            }
        )
    )
