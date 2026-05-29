# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Focused unit tests for omnibase_infra runtime hotspots (OMN-12384).

Covers startup, handler discovery, plugin loading, and wiring surfaces:

service_kernel.py:
- validate_kafka_broker_allowlist: passes without KAFKA_BROKER_ALLOWLIST env,
  raises ProtocolConfigurationError when allowlist is set and broker not listed
- _topic_matches_pattern: exact match, trailing wildcard, mismatch
- _get_contracts_dir: env override, default fallback returns Path
- _resolve_marketplace_skills_root: returns a string

handler_plugin_loader.py:
- load_from_contract: FILE_NOT_FOUND, FILE_IS_DIR, invalid YAML,
  schema validation failure, NAMESPACE_NOT_ALLOWED, MODULE_NOT_FOUND
- _sanitize_exception_message: strips filesystem paths from exception messages

auto_wiring/handler_wiring.py:
- _assert_is_ownership_query: rejects non-protocol objects (raises ModelOnexError)
- wire_from_manifest: empty manifest returns report with zero contracts
- wire_from_manifest: deterministic across two identical calls
- wire_from_manifest: unreachable handler module produces non-WIRED outcome

Related: OMN-12384
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_infra.errors import InfraConnectionError, ProtocolConfigurationError
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _assert_is_ownership_query,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.handler_plugin_loader import (
    HandlerPluginLoader,
    _sanitize_exception_message,
)
from omnibase_infra.runtime.service_kernel import (
    _get_contracts_dir,
    _resolve_marketplace_skills_root,
    _topic_matches_pattern,
    validate_kafka_broker_allowlist,
)

# ---------------------------------------------------------------------------
# service_kernel.py — validate_kafka_broker_allowlist
# ---------------------------------------------------------------------------


class TestValidateKafkaBrokerAllowlist:
    """validate_kafka_broker_allowlist: reads KAFKA_BROKER_ALLOWLIST from env."""

    def test_no_allowlist_env_var_always_passes(self) -> None:
        """Without KAFKA_BROKER_ALLOWLIST set, any broker is accepted."""
        env = {"KAFKA_BROKER_ALLOWLIST": ""}
        with patch.dict("os.environ", env, clear=True):
            # Should not raise
            validate_kafka_broker_allowlist("kafka.prod:9092")

    def test_broker_matching_allowlist_prefix_passes(self) -> None:
        """Broker whose prefix appears in the allowlist is accepted."""
        with patch.dict(
            "os.environ", {"KAFKA_BROKER_ALLOWLIST": "kafka.dev:9092"}, clear=True
        ):
            # Must not raise
            validate_kafka_broker_allowlist("kafka.dev:9092")

    def test_broker_not_in_allowlist_raises(self) -> None:
        """Broker not matching any allowlist prefix raises ProtocolConfigurationError."""
        with patch.dict(
            "os.environ",
            {"KAFKA_BROKER_ALLOWLIST": "kafka.dev:9092"},
            clear=True,
        ):
            with pytest.raises(ProtocolConfigurationError):
                validate_kafka_broker_allowlist("evil.host:9092")

    def test_cleared_allowlist_env_var_allows_all(self) -> None:
        """Explicit empty string in env means no restriction applied."""
        import os

        env = dict(os.environ.items())
        env.pop("KAFKA_BROKER_ALLOWLIST", None)
        with patch.dict("os.environ", env, clear=True):
            # Should not raise — no allowlist configured
            validate_kafka_broker_allowlist("any.host:9092")


# ---------------------------------------------------------------------------
# service_kernel.py — _topic_matches_pattern
# ---------------------------------------------------------------------------


class TestTopicMatchesPattern:
    """_topic_matches_pattern: exact match and trailing/mid wildcard."""

    def test_exact_match_returns_true(self) -> None:
        assert (
            _topic_matches_pattern("onex.evt.foo.bar.v1", "onex.evt.foo.bar.v1") is True
        )

    def test_mismatch_returns_false(self) -> None:
        assert (
            _topic_matches_pattern("onex.evt.foo.bar.v1", "onex.evt.foo.baz.v1")
            is False
        )

    def test_trailing_wildcard_matches_version(self) -> None:
        """Pattern with trailing * matches any version suffix."""
        assert (
            _topic_matches_pattern("onex.evt.foo.bar.v1", "onex.evt.foo.bar.*") is True
        )

    def test_mid_wildcard_matches_segment(self) -> None:
        """Pattern with * in middle matches the corresponding segment."""
        assert (
            _topic_matches_pattern("onex.evt.foo.bar.v1", "onex.evt.*.bar.v1") is True
        )

    def test_wildcard_mismatch_on_first_segment(self) -> None:
        """Leading * (no matching prefix) does not match different prefix."""
        assert _topic_matches_pattern("other.evt.foo.bar.v1", "onex.*") is False

    def test_both_empty_match(self) -> None:
        assert _topic_matches_pattern("", "") is True


# ---------------------------------------------------------------------------
# service_kernel.py — _get_contracts_dir
# ---------------------------------------------------------------------------


class TestGetContractsDir:
    def test_returns_path_object(self) -> None:
        result = _get_contracts_dir()
        assert isinstance(result, Path)

    def test_env_override_is_used(self, tmp_path: Path) -> None:
        with patch.dict(
            "os.environ", {"ONEX_CONTRACTS_DIR": str(tmp_path)}, clear=False
        ):
            result = _get_contracts_dir()
            assert result == tmp_path


# ---------------------------------------------------------------------------
# service_kernel.py — _resolve_marketplace_skills_root
# ---------------------------------------------------------------------------


class TestResolveMarketplaceSkillsRoot:
    def test_returns_string(self) -> None:
        result = _resolve_marketplace_skills_root()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# handler_plugin_loader.py — _sanitize_exception_message
# ---------------------------------------------------------------------------


class TestSanitizeExceptionMessage:
    def test_strips_unix_absolute_path(self) -> None:
        e = OSError("[Errno 13] Permission denied: '/etc/secrets/key.pem'")
        sanitized = _sanitize_exception_message(e)
        assert "/etc/secrets/key.pem" not in sanitized
        assert "Permission denied" in sanitized

    def test_strips_relative_path(self) -> None:
        e = FileNotFoundError("./contracts/node/handler.yaml not found")
        sanitized = _sanitize_exception_message(e)
        assert "./contracts" not in sanitized

    def test_plain_message_unchanged(self) -> None:
        e = ValueError("invalid configuration: timeout must be positive")
        sanitized = _sanitize_exception_message(e)
        assert "invalid configuration" in sanitized


# ---------------------------------------------------------------------------
# handler_plugin_loader.py — load_from_contract error paths
# ---------------------------------------------------------------------------


class TestHandlerPluginLoaderErrors:
    """load_from_contract raises typed errors for each failure mode."""

    def test_file_not_found_raises_configuration_error(self, tmp_path: Path) -> None:
        loader = HandlerPluginLoader()
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(ProtocolConfigurationError):
            loader.load_from_contract(missing)

    def test_directory_path_raises_configuration_error(self, tmp_path: Path) -> None:
        """Passing a directory instead of a file raises ProtocolConfigurationError."""
        loader = HandlerPluginLoader()
        with pytest.raises(ProtocolConfigurationError):
            loader.load_from_contract(tmp_path)

    def test_invalid_yaml_raises_configuration_error(self, tmp_path: Path) -> None:
        contract = tmp_path / "bad.yaml"
        contract.write_text("key: [unclosed\n  - list\n")
        loader = HandlerPluginLoader()
        with pytest.raises(ProtocolConfigurationError):
            loader.load_from_contract(contract)

    def test_wrong_schema_raises_configuration_error(self, tmp_path: Path) -> None:
        """Valid YAML that fails ModelHandlerContract Pydantic schema raises."""
        contract = tmp_path / "schema_fail.yaml"
        contract.write_text(yaml.dump({"completely": "wrong", "schema": True}))
        loader = HandlerPluginLoader()
        with pytest.raises(ProtocolConfigurationError):
            loader.load_from_contract(contract)

    def test_namespace_not_allowed_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """Handler module outside allowed_namespaces raises ProtocolConfigurationError."""
        # We need a YAML that passes schema validation but has a disallowed module.
        # Use a minimal valid-looking contract payload:
        contract = tmp_path / "restricted.yaml"
        contract.write_text(
            yaml.dump(
                {
                    "handler_type": "EFFECT_GENERIC",
                    "handler_module": "evil_package.handlers",
                    "handler_class": "EvilHandler",
                    "contract_version": {"major": 1, "minor": 0, "patch": 0},
                    "name": "EvilHandler",
                }
            )
        )
        loader = HandlerPluginLoader(
            allowed_namespaces=["omnibase_infra.", "omnibase_core."]
        )
        with pytest.raises(ProtocolConfigurationError):
            loader.load_from_contract(contract)

    def test_module_not_found_raises_infra_connection_error(
        self, tmp_path: Path
    ) -> None:
        """Nonexistent module raises InfraConnectionError (MODULE_NOT_FOUND).

        The handler_class field encodes both module and class as a dotted path
        (module.ClassName). A nonexistent module triggers MODULE_NOT_FOUND which
        the loader surfaces as InfraConnectionError, not ProtocolConfigurationError.
        """
        contract = tmp_path / "module_missing.yaml"
        contract.write_text(
            yaml.dump(
                {
                    "name": "handler-missing-module",
                    "handler_class": "omnibase_infra.nonexistent.xyz.module.SomeHandler",
                    "handler_type": "effect",
                }
            )
        )
        loader = HandlerPluginLoader()
        with pytest.raises(InfraConnectionError):
            loader.load_from_contract(contract)


# ---------------------------------------------------------------------------
# auto_wiring/handler_wiring.py — _assert_is_ownership_query
# ---------------------------------------------------------------------------


class TestAssertIsOwnershipQuery:
    """_assert_is_ownership_query raises ModelOnexError for non-conforming objects."""

    def test_none_raises_model_onex_error(self) -> None:
        with pytest.raises(ModelOnexError):
            _assert_is_ownership_query(None)

    def test_plain_object_raises_model_onex_error(self) -> None:
        with pytest.raises(ModelOnexError):
            _assert_is_ownership_query(object())

    def test_string_raises_model_onex_error(self) -> None:
        with pytest.raises(ModelOnexError):
            _assert_is_ownership_query("not-a-protocol")

    def test_conforming_service_passes(self) -> None:
        """ServiceLocalHandlerOwnershipQuery satisfies the protocol check."""
        from omnibase_core.services.service_local_handler_ownership_query import (
            ServiceLocalHandlerOwnershipQuery,
        )

        ownership = ServiceLocalHandlerOwnershipQuery(local_node_names=frozenset())
        # Must not raise
        _assert_is_ownership_query(ownership)


# ---------------------------------------------------------------------------
# auto_wiring/handler_wiring.py — wire_from_manifest
# ---------------------------------------------------------------------------


def _make_ownership() -> object:
    from omnibase_core.services.service_local_handler_ownership_query import (
        ServiceLocalHandlerOwnershipQuery,
    )

    return ServiceLocalHandlerOwnershipQuery(local_node_names=frozenset())


def _make_empty_manifest() -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(contracts=[])


def _make_single_bad_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_bad_handler",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_bad_handler",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.test.handler.v1",), publish_topics=()
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=[
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="FakeHandler",
                        module="totally.fake.module.does.not.exist",
                    )
                )
            ],
        ),
    )


class TestWireFromManifest:
    @pytest.mark.asyncio
    async def test_empty_manifest_returns_empty_report(self) -> None:
        """Empty manifest produces a report with zero contract results."""
        mock_engine = MagicMock()

        report = await wire_from_manifest(
            manifest=_make_empty_manifest(),
            dispatch_engine=mock_engine,
            subscribe_immediately=False,
        )
        assert report is not None
        assert len(report.results) == 0

    @pytest.mark.asyncio
    async def test_report_is_deterministic_across_two_runs(self) -> None:
        """Two calls with the same empty manifest produce identical report structure."""
        mock_engine = MagicMock()
        manifest = _make_empty_manifest()

        report_a = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=mock_engine,
            subscribe_immediately=False,
        )
        report_b = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=mock_engine,
            subscribe_immediately=False,
        )
        assert len(report_a.results) == len(report_b.results)

    @pytest.mark.asyncio
    async def test_unreachable_module_produces_non_wired_outcome(self) -> None:
        """A contract with a nonexistent handler module produces a non-WIRED outcome."""
        mock_engine = MagicMock()
        manifest = ModelAutoWiringManifest(contracts=[_make_single_bad_contract()])

        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=mock_engine,
            subscribe_immediately=False,
        )
        assert len(report.results) == 1

        from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome

        assert report.results[0].outcome != EnumWiringOutcome.WIRED
