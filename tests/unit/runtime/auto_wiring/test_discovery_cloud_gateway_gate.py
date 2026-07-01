# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the dormant cloud-gateway discovery gate (OMN-13809).

The bus forwarder node (``node_bus_forwarder_effect``) subscribes to
``onex.cmd.omnibase-infra.delegation-inference-request.v1``, whose real payload
is ``ModelInferenceIntent`` — a domain model, not the ``ModelGatewayEnvelope``
the forwarder handlers expect. On lanes with no hosted cloud Kafka edge (e.g.
the ``.201`` compose lanes) the forwarder has nothing to forward to, so wiring
it only produces a ``ValidationError`` on every delegation message.

These tests prove the forwarder contract is *skipped* by contract discovery
unless cloud mirroring is explicitly enabled, while remaining fully wired when
it is — without perturbing ordinary (non-gateway) contracts.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

import omnibase_infra.nodes.node_bus_forwarder_effect as bus_forwarder_pkg
from omnibase_infra.runtime.auto_wiring.discovery import (
    _contract_requires_cloud_gateway,
    _parse_contract,
    discover_contracts_from_paths,
)
from omnibase_infra.utils.util_runtime_packages import (
    ENV_GATEWAY_CLOUD_MIRRORING_ENABLED,
    is_gateway_cloud_mirroring_enabled,
)

pytestmark = pytest.mark.unit

_BUS_FORWARDER_CONTRACT = Path(bus_forwarder_pkg.__file__).parent / "contract.yaml"
_DELEGATION_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"


def _write_plain_contract(tmp_path: Path) -> Path:
    """Write a minimal non-gateway contract and return its path."""
    content = dedent("""\
        name: "node_plain_effect"
        node_type: "EFFECT_GENERIC"
        contract_version:
          major: 1
          minor: 0
          patch: 0
        node_version: "1.0.0"
        description: "A plain effect node with no cloud gateway leg"
        event_bus:
          subscribe_topics:
            - "onex.evt.platform.plain-input.v1"
          publish_topics:
            - "onex.evt.platform.plain-output.v1"
    """)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(content)
    return contract_path


class TestGatewayCloudMirroringEnabledHelper:
    """Fail-safe env parsing for the cloud mirroring gate."""

    def test_default_off_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_GATEWAY_CLOUD_MIRRORING_ENABLED, raising=False)
        assert is_gateway_cloud_mirroring_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_truthy_values_enable(self, value: str) -> None:
        assert is_gateway_cloud_mirroring_enabled(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  "])
    def test_falsy_values_disable(self, value: str) -> None:
        assert is_gateway_cloud_mirroring_enabled(value) is False


class TestContractRequiresCloudGatewayDetection:
    """Structural detection of a contract-declared cloud gateway leg."""

    def test_real_bus_forwarder_contract_requires_cloud_gateway(self) -> None:
        contract = _parse_contract(
            contract_path=_BUS_FORWARDER_CONTRACT,
            entry_point_name="node_bus_forwarder_effect",
            package_name="omnibase-infra",
            package_version="0.0.0",
        )
        assert contract.requires_cloud_gateway is True
        # Sanity: the forwarder really does subscribe to the delegation topic
        # whose payload collides with ModelGatewayEnvelope (the OMN-13809 bug).
        assert contract.event_bus is not None
        assert _DELEGATION_TOPIC in contract.event_bus.subscribe_topics

    def test_plain_contract_does_not_require_cloud_gateway(
        self, tmp_path: Path
    ) -> None:
        contract = _parse_contract(
            contract_path=_write_plain_contract(tmp_path),
            entry_point_name="node_plain_effect",
            package_name="local",
            package_version="0.0.0",
        )
        assert contract.requires_cloud_gateway is False

    def test_detection_helper_handles_missing_config(self) -> None:
        assert _contract_requires_cloud_gateway({}) is False
        assert _contract_requires_cloud_gateway({"config": "not-a-dict"}) is False
        assert _contract_requires_cloud_gateway({"config": {}}) is False
        assert (
            _contract_requires_cloud_gateway(
                {"config": {"gateway_forwarder": {"local_leg": {}}}}
            )
            is False
        )
        assert (
            _contract_requires_cloud_gateway(
                {"config": {"gateway_forwarder": {"cloud_leg": {"transport": "kafka"}}}}
            )
            is True
        )


class TestDiscoveryGate:
    """The gate that keeps the forwarder dormant on no-cloud-leg lanes."""

    def test_bus_forwarder_skipped_when_mirroring_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(ENV_GATEWAY_CLOUD_MIRRORING_ENABLED, raising=False)

        manifest = discover_contracts_from_paths(
            [_BUS_FORWARDER_CONTRACT, _write_plain_contract(tmp_path)]
        )

        discovered = {c.name for c in manifest.contracts}
        # The forwarder is skipped: nothing subscribes its gateway handlers to
        # the delegation topic, so ModelInferenceIntent no longer ValidationErrors.
        assert "node_bus_forwarder_effect" not in discovered
        # An ordinary contract on the same lane is unaffected.
        assert "node_plain_effect" in discovered
        assert manifest.errors == ()

    def test_bus_forwarder_wired_when_mirroring_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_GATEWAY_CLOUD_MIRRORING_ENABLED, "true")

        manifest = discover_contracts_from_paths([_BUS_FORWARDER_CONTRACT])

        discovered = {c.name for c in manifest.contracts}
        assert "node_bus_forwarder_effect" in discovered
