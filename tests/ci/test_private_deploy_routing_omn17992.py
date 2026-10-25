# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Portable overlay controls; fixtures contain no deployment identities."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
sys.path.insert(0, str(ROOT / "scripts/deploy-agent"))

TABLE = """default_instance: primary
instances:
  primary:
    hostnames: [worker.example.invalid]
    consumer_group: example-deployer
    flags: {max_dev_lag_minutes: 17}
    verify:
      receipt_lane: example-receipt
      runner_labels: [example-runner]
      deploy_agent_url: http://agent.example.invalid
      main_url: http://main.example.invalid
      effects_url: http://effects.example.invalid
      projection_url: http://projection.example.invalid
      compose_project: example-project
      runtime_container: example-runtime
      effects_container: example-effects
      postgres_container: example-postgres
      broker_container: example-broker
routes: []
"""


@pytest.fixture
def overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))
    import hashlib

    digest = hashlib.sha256(TABLE.encode()).hexdigest()
    path = tmp_path / f"config/onex/overlays/deploy_lane_routing/{digest}.yaml"
    contract = tmp_path / "contract.yaml"
    contract.write_text(f"overlay_sha256: {digest}\n")
    import deploy_lane_verify_route as ci_config
    from deploy_agent.routing import _private_config as agent_config

    monkeypatch.setattr(agent_config, "CONTRACT_PATH", contract)
    monkeypatch.setattr(ci_config._private_config, "CONTRACT_PATH", contract)
    path.parent.mkdir(parents=True)
    path.write_text(TABLE)
    return path


def test_agent_and_ci_read_same_private_table(overlay):
    from deploy_agent.routing import lane_flags_for_instance, load_routing_table
    from deploy_lane_verify_route import load_table, resolve
    from instance_receipt_lanes import receipt_lanes_for

    assert load_routing_table().default_instance == "primary"
    assert lane_flags_for_instance(ROOT, "primary").max_dev_lag_minutes == 17
    assert (
        resolve(load_table(), runtime_lane="dev", requested_by="gha/example/1").instance
        == "primary"
    )
    assert receipt_lanes_for("example") == ()


def test_missing_overlay_refuses_public_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))
    from deploy_agent.routing import RoutingTableError, load_routing_table
    from deploy_lane_verify_route import VerifyRouteError, load_table
    from instance_receipt_lanes import InstanceReceiptLanesError, receipt_lanes_for

    for error, call in [
        (RoutingTableError, load_routing_table),
        (VerifyRouteError, load_table),
        (InstanceReceiptLanesError, lambda: receipt_lanes_for("example")),
    ]:
        with pytest.raises(error, match="ONEX_WORKSPACE_CONFIG_ROOT"):
            call()


def test_public_contract_has_no_instances():
    import yaml

    raw = yaml.safe_load((ROOT / "config/deploy_lane_routing.yaml").read_text())
    assert len(raw.pop("overlay_sha256")) == 64
    assert raw == {
        "default_instance": None,
        "instances": {},
        "routes": [],
        "excluded_hosts": {},
    }


def test_corrupt_overlay_refuses(overlay):
    from deploy_agent.routing import RoutingTableError, load_routing_table

    overlay.write_text(TABLE + "# drift\n")
    with pytest.raises(RoutingTableError, match="digest mismatch"):
        load_routing_table()


def test_all_agent_config_readers_require_the_same_overlay(overlay):
    from deploy_agent.instance_lanes import (
        load_instance_lanes,
        load_instance_runtime_lanes,
    )
    from deploy_agent.load_gate import load_gate_for_instance

    assert load_instance_lanes(ROOT) == {}
    assert load_instance_runtime_lanes(ROOT) == {"primary": "example-receipt"}
    assert load_gate_for_instance(ROOT, "primary") is None
    overlay.unlink()
    from deploy_agent.routing import RoutingTableError

    for call in [
        lambda: load_instance_lanes(ROOT),
        lambda: load_instance_runtime_lanes(ROOT),
        lambda: load_gate_for_instance(ROOT, "primary"),
    ]:
        with pytest.raises(RoutingTableError, match="ONEX_WORKSPACE_CONFIG_ROOT"):
            call()
