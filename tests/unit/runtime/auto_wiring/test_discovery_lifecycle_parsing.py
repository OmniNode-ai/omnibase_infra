# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for lifecycle hook parsing in contract discovery (OMN-7662).

Proves that:
1. discover_contracts_from_paths extracts lifecycle hooks from contract YAML.
2. Contracts without lifecycle sections have lifecycle_hooks=None.
3. Partially defined lifecycle sections parse correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths


class TestDiscoveryLifecycleParsing:
    """Tests for lifecycle hook extraction during contract discovery."""

    def test_contract_with_lifecycle_hooks(self, tmp_path: Path) -> None:
        """Lifecycle section is parsed into ModelLifecycleHooks."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text(
            """
name: test-node-with-lifecycle
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
event_bus:
  subscribe_topics:
    - onex.cmd.test.my-command.v1
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerTest
        module: tests.unit.runtime.auto_wiring.test_discovery_lifecycle_parsing
lifecycle:
  on_start:
    callable_ref: mypackage.hooks.on_start
    timeout_seconds: 15.0
    required: false
    idempotent: true
  on_shutdown:
    callable_ref: mypackage.hooks.on_shutdown
    timeout_seconds: 5.0
    required: false
    idempotent: true
"""
        )

        manifest = discover_contracts_from_paths([contract_yaml])
        assert manifest.total_discovered == 1
        assert manifest.total_errors == 0

        contract = manifest.contracts[0]
        assert contract.lifecycle_hooks is not None
        assert contract.lifecycle_hooks.has_hooks()

        assert contract.lifecycle_hooks.on_start is not None
        assert (
            contract.lifecycle_hooks.on_start.callable_ref == "mypackage.hooks.on_start"
        )
        assert contract.lifecycle_hooks.on_start.timeout_seconds == 15.0
        assert contract.lifecycle_hooks.on_start.required is False

        assert contract.lifecycle_hooks.on_shutdown is not None
        assert (
            contract.lifecycle_hooks.on_shutdown.callable_ref
            == "mypackage.hooks.on_shutdown"
        )

        assert contract.lifecycle_hooks.validate_handshake is None

    def test_contract_without_lifecycle(self, tmp_path: Path) -> None:
        """Contracts without lifecycle section have lifecycle_hooks=None."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text(
            """
name: test-node-no-lifecycle
node_type: COMPUTE_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerTest
        module: tests.unit.runtime.auto_wiring.test_discovery_lifecycle_parsing
"""
        )

        manifest = discover_contracts_from_paths([contract_yaml])
        assert manifest.total_discovered == 1
        contract = manifest.contracts[0]
        assert contract.lifecycle_hooks is None

    def test_contract_with_all_lifecycle_hooks(self, tmp_path: Path) -> None:
        """All three lifecycle hooks (on_start, validate_handshake, on_shutdown)."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text(
            """
name: test-node-full-lifecycle
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
lifecycle:
  on_start:
    callable_ref: pkg.hooks.start
    idempotent: true
  validate_handshake:
    callable_ref: pkg.hooks.validate
    timeout_seconds: 20.0
    idempotent: true
  on_shutdown:
    callable_ref: pkg.hooks.stop
    idempotent: true
"""
        )

        manifest = discover_contracts_from_paths([contract_yaml])
        contract = manifest.contracts[0]
        assert contract.lifecycle_hooks is not None

        assert contract.lifecycle_hooks.on_start is not None
        assert contract.lifecycle_hooks.validate_handshake is not None
        assert contract.lifecycle_hooks.validate_handshake.timeout_seconds == 20.0
        assert contract.lifecycle_hooks.on_shutdown is not None

    def test_contract_with_only_on_start(self, tmp_path: Path) -> None:
        """A lifecycle section with only on_start."""
        contract_yaml = tmp_path / "contract.yaml"
        contract_yaml.write_text(
            """
name: test-node-start-only
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
lifecycle:
  on_start:
    callable_ref: pkg.hooks.start
    idempotent: true
"""
        )

        manifest = discover_contracts_from_paths([contract_yaml])
        contract = manifest.contracts[0]
        assert contract.lifecycle_hooks is not None
        assert contract.lifecycle_hooks.has_hooks()
        assert contract.lifecycle_hooks.on_start is not None
        assert contract.lifecycle_hooks.on_shutdown is None


# Stub handler class for auto-wiring tests
class HandlerTest:
    async def handle(self, envelope: object) -> None:
        pass
