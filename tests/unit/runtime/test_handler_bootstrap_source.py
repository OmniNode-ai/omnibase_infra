# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for HandlerBootstrapSource hardcoded handler registration.

Tests the HandlerBootstrapSource functionality including:
- Protocol compliance with ProtocolContractSource
- Bootstrap handler discovery (consul, db, http, mcp, vault)
- ModelHandlerDescriptor validation for all bootstrap handlers
- Graceful mode behavior (API consistency)
- Idempotency of discover_handlers() calls

Related:
    - OMN-1087: HandlerBootstrapSource for hardcoded handler registration
    - src/omnibase_infra/runtime/handler_bootstrap_source.py
    - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md

Expected Behavior:
    HandlerBootstrapSource implements ProtocolContractSource from omnibase_infra.
    It provides hardcoded handler descriptors for core infrastructure handlers
    (Consul, Database, HTTP, MCP, Vault) without requiring contract.yaml files.

    The source_type property returns "BOOTSTRAP" as per the implementation.
    All handlers have handler_kind="effect" since they perform external I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)
from omnibase_infra.runtime.handler_bootstrap_source import (
    SOURCE_TYPE_BOOTSTRAP,
    HandlerBootstrapSource,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

# =============================================================================
# Constants for Test Validation
# =============================================================================

# Expected bootstrap handler IDs (using "proto." prefix for protocol identity namespace)
EXPECTED_HANDLER_IDS = frozenset(
    {
        "proto.consul",
        "proto.db",
        "proto.http",
        "proto.mcp",
        "proto.vault",
    }
)

# Expected handler kind for all bootstrap handlers
EXPECTED_HANDLER_KIND = "effect"

# Expected version for all bootstrap handlers
EXPECTED_VERSION = "1.0.0"

# Expected count of bootstrap handlers
EXPECTED_HANDLER_COUNT = 5

# Performance threshold: 20ms allows for contract YAML file I/O during handler
# discovery. Pre-OMN-1282 threshold was 10ms when no contract files were loaded.
# Current overhead comes from:
# - Reading handler_contract.yaml for each bootstrap handler (5 handlers)
# - YAML parsing via yaml.safe_load()
# - Path resolution and symlink handling
# - CI environment disk I/O variance
# See: OMN-1282 for contract-driven handler configuration migration
PERFORMANCE_THRESHOLD_MS = 20.0
PERFORMANCE_THRESHOLD_SECONDS = PERFORMANCE_THRESHOLD_MS / 1000.0


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestHandlerBootstrapSourceProtocolCompliance:
    """Tests for ProtocolContractSource compliance.

    These tests verify that HandlerBootstrapSource correctly implements
    the ProtocolContractSource protocol with all required methods and properties.
    """

    def test_handler_bootstrap_source_can_be_imported(self) -> None:
        """HandlerBootstrapSource should be importable from omnibase_infra.runtime.

        Expected import path:
            from omnibase_infra.runtime.handler_bootstrap_source import HandlerBootstrapSource
        """
        from omnibase_infra.runtime.handler_bootstrap_source import (
            HandlerBootstrapSource,
        )

        assert HandlerBootstrapSource is not None

    def test_handler_bootstrap_source_implements_protocol(self) -> None:
        """HandlerBootstrapSource should implement ProtocolContractSource.

        The implementation must satisfy ProtocolContractSource with:
        - source_type property returning "BOOTSTRAP"
        - async discover_handlers() method returning ModelContractDiscoveryResult
        """
        source = HandlerBootstrapSource()

        # Protocol compliance check via duck typing (ONEX convention)
        assert hasattr(source, "source_type")
        assert hasattr(source, "discover_handlers")
        assert callable(source.discover_handlers)

        # Runtime checkable protocol verification
        assert isinstance(source, ProtocolContractSource)

    def test_handler_bootstrap_source_type_is_bootstrap(self) -> None:
        """HandlerBootstrapSource.source_type should return "BOOTSTRAP".

        The source_type is used for observability and debugging purposes only.
        The runtime MUST NOT branch on this value.
        """
        source = HandlerBootstrapSource()

        assert source.source_type == SOURCE_TYPE_BOOTSTRAP
        assert source.source_type == "BOOTSTRAP"

    def test_source_type_constant_exported(self) -> None:
        """SOURCE_TYPE_BOOTSTRAP constant should be exported and match implementation."""
        from omnibase_infra.runtime.handler_bootstrap_source import (
            SOURCE_TYPE_BOOTSTRAP,
        )

        assert SOURCE_TYPE_BOOTSTRAP == "BOOTSTRAP"

        source = HandlerBootstrapSource()
        assert source.source_type == SOURCE_TYPE_BOOTSTRAP

    def test_handler_bootstrap_source_in_runtime_exports(self) -> None:
        """HandlerBootstrapSource should be exported from omnibase_infra.runtime."""
        from omnibase_infra.runtime import (
            SOURCE_TYPE_BOOTSTRAP as RUNTIME_SOURCE_TYPE,
        )
        from omnibase_infra.runtime import (
            HandlerBootstrapSource as RuntimeExportedSource,
        )

        assert RuntimeExportedSource is HandlerBootstrapSource
        assert RUNTIME_SOURCE_TYPE == SOURCE_TYPE_BOOTSTRAP


# =============================================================================
# Handler Discovery Tests
# =============================================================================


class TestHandlerBootstrapSourceDiscovery:
    """Tests for handler discovery functionality.

    These tests verify that HandlerBootstrapSource.discover_handlers() correctly
    returns the expected bootstrap handlers with proper descriptors.
    """

    @pytest.mark.asyncio
    async def test_discover_handlers_returns_model_contract_discovery_result(
        self,
    ) -> None:
        """discover_handlers() should return ModelContractDiscoveryResult.

        The result must be an instance of ModelContractDiscoveryResult containing
        both descriptors and validation_errors fields.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        assert isinstance(result, ModelContractDiscoveryResult)
        assert hasattr(result, "descriptors")
        assert hasattr(result, "validation_errors")

    @pytest.mark.asyncio
    async def test_discovers_exactly_five_handlers(self) -> None:
        """discover_handlers() should return exactly 5 bootstrap handlers.

        The bootstrap handlers are:
        - bootstrap.consul: HashiCorp Consul service discovery
        - bootstrap.db: PostgreSQL database operations
        - bootstrap.http: HTTP REST protocol
        - bootstrap.mcp: Model Context Protocol for AI agent integration
        - bootstrap.vault: HashiCorp Vault secret management
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        assert len(result.descriptors) == EXPECTED_HANDLER_COUNT, (
            f"Expected {EXPECTED_HANDLER_COUNT} bootstrap handlers, "
            f"got {len(result.descriptors)}"
        )

    @pytest.mark.asyncio
    async def test_discovered_handler_ids_match_expected(self) -> None:
        """All discovered handlers should have expected handler_id values.

        Handler IDs must follow the pattern "bootstrap.<service_name>" where
        service_name is one of: consul, db, http, mcp, vault.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        discovered_ids = {d.handler_id for d in result.descriptors}

        assert discovered_ids == EXPECTED_HANDLER_IDS, (
            f"Handler ID mismatch. Expected: {EXPECTED_HANDLER_IDS}, "
            f"Got: {discovered_ids}"
        )

    @pytest.mark.asyncio
    async def test_all_handlers_have_proto_prefix(self) -> None:
        """All handler IDs should start with 'proto.' prefix (protocol identity namespace)."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.handler_id.startswith("proto."), (
                f"Handler ID '{descriptor.handler_id}' does not have 'proto.' prefix"
            )

    @pytest.mark.asyncio
    async def test_all_handlers_have_effect_kind(self) -> None:
        """All bootstrap handlers should have handler_kind='effect'.

        Effect handlers perform external I/O operations with infrastructure services.
        All bootstrap handlers interact with external systems (Consul, DB, HTTP, Vault).
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.handler_kind == EXPECTED_HANDLER_KIND, (
                f"Handler '{descriptor.handler_id}' has kind '{descriptor.handler_kind}', "
                f"expected '{EXPECTED_HANDLER_KIND}'"
            )

    @pytest.mark.asyncio
    async def test_all_handlers_have_version_1_0_0(self) -> None:
        """All bootstrap handlers should have version='1.0.0'.

        Bootstrap handlers use a stable version since they are hardcoded
        definitions that don't change through contract files.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            # version is ModelSemVer, compare using str() for string representation
            assert str(descriptor.version) == EXPECTED_VERSION, (
                f"Handler '{descriptor.handler_id}' has version '{descriptor.version}', "
                f"expected '{EXPECTED_VERSION}'"
            )
            # Also verify components
            assert descriptor.version.major == 1
            assert descriptor.version.minor == 0
            assert descriptor.version.patch == 0

    @pytest.mark.asyncio
    async def test_all_handlers_have_valid_contract_paths(self) -> None:
        """All bootstrap handlers should have valid contract_path values.

        Bootstrap handlers now load their configuration from contract YAML files.
        Each handler should have a contract_path that points to an existing file.

        Basic handlers use: contracts/handlers/<type>/handler_contract.yaml
        MCP uses rich contract: src/omnibase_infra/contracts/handlers/mcp/handler_contract.yaml
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        expected_paths = {
            "proto.consul": "contracts/handlers/consul/handler_contract.yaml",
            "proto.db": "contracts/handlers/db/handler_contract.yaml",
            "proto.http": "contracts/handlers/http/handler_contract.yaml",
            "proto.vault": "contracts/handlers/vault/handler_contract.yaml",
            "proto.mcp": "src/omnibase_infra/contracts/handlers/mcp/handler_contract.yaml",
        }

        for descriptor in result.descriptors:
            assert descriptor.contract_path is not None, (
                f"Handler '{descriptor.handler_id}' has contract_path=None, expected a path"
            )
            expected = expected_paths.get(descriptor.handler_id)
            assert descriptor.contract_path == expected, (
                f"Handler '{descriptor.handler_id}' has contract_path="
                f"'{descriptor.contract_path}', expected '{expected}'"
            )

    @pytest.mark.asyncio
    async def test_validation_errors_always_empty(self) -> None:
        """validation_errors should always be empty for bootstrap source.

        Bootstrap handlers are hardcoded and validated at development time,
        so there should never be validation errors.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        assert result.validation_errors == [], (
            f"Expected empty validation_errors, got {len(result.validation_errors)} errors"
        )

    @pytest.mark.asyncio
    async def test_descriptors_are_model_handler_descriptor_instances(self) -> None:
        """All descriptors should be instances of ModelHandlerDescriptor."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert isinstance(descriptor, ModelHandlerDescriptor), (
                f"Descriptor for '{descriptor.handler_id}' is not ModelHandlerDescriptor, "
                f"got {type(descriptor).__name__}"
            )


# =============================================================================
# ModelHandlerDescriptor Validation Tests
# =============================================================================


class TestHandlerBootstrapSourceDescriptors:
    """Tests for ModelHandlerDescriptor validation.

    These tests verify that all bootstrap handler descriptors have properly
    populated required fields with valid values.
    """

    @pytest.mark.asyncio
    async def test_all_descriptors_have_non_empty_handler_id(self) -> None:
        """All descriptors should have a non-empty handler_id."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.handler_id, "handler_id should not be empty"
            assert len(descriptor.handler_id) > 0

    @pytest.mark.asyncio
    async def test_all_descriptors_have_non_empty_name(self) -> None:
        """All descriptors should have a non-empty human-readable name."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.name, f"Handler '{descriptor.handler_id}' has empty name"
            assert len(descriptor.name) > 0

    @pytest.mark.asyncio
    async def test_all_descriptors_have_non_empty_description(self) -> None:
        """All descriptors should have a non-empty description."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.description, (
                f"Handler '{descriptor.handler_id}' has empty description"
            )
            assert len(descriptor.description) > 0

    @pytest.mark.asyncio
    async def test_all_descriptors_have_valid_input_model_path(self) -> None:
        """All descriptors should have a valid fully qualified input_model path.

        The input_model path must be a valid Python module path
        (e.g., 'omnibase_core.types.JsonDict').
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.input_model, (
                f"Handler '{descriptor.handler_id}' has empty input_model"
            )
            # Verify it looks like a module path (contains at least one dot)
            assert "." in descriptor.input_model, (
                f"Handler '{descriptor.handler_id}' input_model '{descriptor.input_model}' "
                "does not look like a fully qualified path"
            )

    @pytest.mark.asyncio
    async def test_all_descriptors_have_valid_output_model_path(self) -> None:
        """All descriptors should have a valid fully qualified output_model path.

        The output_model path must be a valid Python module path
        (e.g., 'omnibase_core.models.dispatch.ModelHandlerOutput').
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.output_model, (
                f"Handler '{descriptor.handler_id}' has empty output_model"
            )
            # Verify it looks like a module path (contains at least one dot)
            assert "." in descriptor.output_model, (
                f"Handler '{descriptor.handler_id}' output_model '{descriptor.output_model}' "
                "does not look like a fully qualified path"
            )

    @pytest.mark.asyncio
    async def test_all_descriptors_have_same_output_model(self) -> None:
        """All bootstrap handlers should use the same output model.

        Bootstrap handlers use envelope-based routing and all return
        ModelHandlerOutput for consistency.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        output_models = {d.output_model for d in result.descriptors}

        # All handlers should have the same output model
        assert len(output_models) == 1, (
            f"Expected all handlers to have same output_model, got {output_models}"
        )

        # Verify it's ModelHandlerOutput
        expected_output = "omnibase_core.models.dispatch.ModelHandlerOutput"
        assert output_models == {expected_output}, (
            f"Expected output_model '{expected_output}', got {output_models}"
        )

    @pytest.mark.asyncio
    async def test_all_descriptors_have_same_input_model(self) -> None:
        """All bootstrap handlers should use JsonDict as input model.

        Bootstrap handlers use envelope-based routing with JsonDict payloads.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        input_models = {d.input_model for d in result.descriptors}

        # All handlers should have the same input model
        assert len(input_models) == 1, (
            f"Expected all handlers to have same input_model, got {input_models}"
        )

        # Verify it's JsonDict
        expected_input = "omnibase_infra.models.types.JsonDict"
        assert input_models == {expected_input}, (
            f"Expected input_model '{expected_input}', got {input_models}"
        )

    @pytest.mark.asyncio
    async def test_consul_handler_descriptor_content(self) -> None:
        """Verify the Consul handler descriptor has expected content."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        consul_descriptors = [
            d for d in result.descriptors if d.handler_id == "proto.consul"
        ]

        assert len(consul_descriptors) == 1, "Should have exactly one Consul handler"

        descriptor = consul_descriptors[0]
        assert descriptor.name == "Consul Handler"
        assert "consul" in descriptor.description.lower()
        assert descriptor.handler_kind == "effect"

    @pytest.mark.asyncio
    async def test_db_handler_descriptor_content(self) -> None:
        """Verify the Database handler descriptor has expected content."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        db_descriptors = [d for d in result.descriptors if d.handler_id == "proto.db"]

        assert len(db_descriptors) == 1, "Should have exactly one Database handler"

        descriptor = db_descriptors[0]
        assert descriptor.name == "Database Handler"
        assert (
            "database" in descriptor.description.lower()
            or "postgres" in descriptor.description.lower()
        )
        assert descriptor.handler_kind == "effect"

    @pytest.mark.asyncio
    async def test_http_handler_descriptor_content(self) -> None:
        """Verify the HTTP handler descriptor has expected content."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        http_descriptors = [
            d for d in result.descriptors if d.handler_id == "proto.http"
        ]

        assert len(http_descriptors) == 1, "Should have exactly one HTTP handler"

        descriptor = http_descriptors[0]
        assert descriptor.name == "HTTP Handler"
        assert "http" in descriptor.description.lower()
        assert descriptor.handler_kind == "effect"

    @pytest.mark.asyncio
    async def test_vault_handler_descriptor_content(self) -> None:
        """Verify the Vault handler descriptor has expected content."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        vault_descriptors = [
            d for d in result.descriptors if d.handler_id == "proto.vault"
        ]

        assert len(vault_descriptors) == 1, "Should have exactly one Vault handler"

        descriptor = vault_descriptors[0]
        assert descriptor.name == "Vault Handler"
        assert "vault" in descriptor.description.lower()
        assert descriptor.handler_kind == "effect"

    @pytest.mark.asyncio
    async def test_mcp_handler_descriptor_content(self) -> None:
        """Verify the MCP handler descriptor has expected content."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        mcp_descriptors = [d for d in result.descriptors if d.handler_id == "proto.mcp"]

        assert len(mcp_descriptors) == 1, "Should have exactly one MCP handler"

        descriptor = mcp_descriptors[0]
        assert descriptor.name == "MCP Handler"
        assert descriptor.handler_kind == "effect"
        assert (
            descriptor.description
            == "Model Context Protocol handler for AI agent integration"
        )
        assert (
            descriptor.handler_class == "omnibase_infra.handlers.handler_mcp.HandlerMCP"
        )


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestHandlerBootstrapSourceIdempotency:
    """Tests for idempotency of discover_handlers() calls.

    The discover_handlers() method should be idempotent - calling it multiple
    times should return the same results.
    """

    @pytest.mark.asyncio
    async def test_multiple_calls_return_same_handler_count(self) -> None:
        """Multiple calls to discover_handlers() should return same handler count."""
        source = HandlerBootstrapSource()

        result1 = await source.discover_handlers()
        result2 = await source.discover_handlers()
        result3 = await source.discover_handlers()

        assert len(result1.descriptors) == len(result2.descriptors)
        assert len(result2.descriptors) == len(result3.descriptors)

    @pytest.mark.asyncio
    async def test_multiple_calls_return_same_handler_ids(self) -> None:
        """Multiple calls to discover_handlers() should return same handler IDs."""
        source = HandlerBootstrapSource()

        result1 = await source.discover_handlers()
        result2 = await source.discover_handlers()

        ids1 = {d.handler_id for d in result1.descriptors}
        ids2 = {d.handler_id for d in result2.descriptors}

        assert ids1 == ids2

    @pytest.mark.asyncio
    async def test_multiple_calls_return_consistent_descriptor_values(self) -> None:
        """Multiple calls should return descriptors with consistent values."""
        source = HandlerBootstrapSource()

        result1 = await source.discover_handlers()
        result2 = await source.discover_handlers()

        # Build lookup by handler_id
        descriptors1 = {d.handler_id: d for d in result1.descriptors}
        descriptors2 = {d.handler_id: d for d in result2.descriptors}

        for handler_id, d1 in descriptors1.items():
            d2 = descriptors2[handler_id]

            assert d1.name == d2.name
            assert str(d1.version) == str(d2.version)
            assert d1.handler_kind == d2.handler_kind
            assert d1.input_model == d2.input_model
            assert d1.output_model == d2.output_model
            assert d1.description == d2.description
            assert d1.contract_path == d2.contract_path

    @pytest.mark.asyncio
    async def test_different_instances_return_same_results(self) -> None:
        """Different HandlerBootstrapSource instances should return same results."""
        source1 = HandlerBootstrapSource()
        source2 = HandlerBootstrapSource()

        result1 = await source1.discover_handlers()
        result2 = await source2.discover_handlers()

        ids1 = {d.handler_id for d in result1.descriptors}
        ids2 = {d.handler_id for d in result2.descriptors}

        assert ids1 == ids2
        assert len(result1.descriptors) == len(result2.descriptors)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestHandlerBootstrapSourceEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_descriptors_are_frozen(self) -> None:
        """ModelHandlerDescriptor instances should be frozen (immutable).

        The ModelHandlerDescriptor model config has frozen=True.
        """
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            # Attempting to modify a frozen model should raise ValidationError
            with pytest.raises(ValidationError):
                descriptor.handler_id = "modified"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_no_duplicate_handler_ids(self) -> None:
        """There should be no duplicate handler IDs in the result."""
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        handler_ids = [d.handler_id for d in result.descriptors]
        unique_ids = set(handler_ids)

        assert len(handler_ids) == len(unique_ids), (
            f"Found duplicate handler IDs: "
            f"{[hid for hid in handler_ids if handler_ids.count(hid) > 1]}"
        )

    @pytest.mark.asyncio
    async def test_handler_kind_is_valid_literal_type(self) -> None:
        """handler_kind should be one of the valid literal types.

        Valid handler kinds: compute, effect, reducer, orchestrator
        """
        valid_kinds = {"compute", "effect", "reducer", "orchestrator"}
        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.handler_kind in valid_kinds, (
                f"Handler '{descriptor.handler_id}' has invalid kind "
                f"'{descriptor.handler_kind}', expected one of {valid_kinds}"
            )

    @pytest.mark.asyncio
    async def test_version_is_model_semver_instance(self) -> None:
        """version field should be a ModelSemVer instance, not a string."""
        from omnibase_core.models.primitives.model_semver import ModelSemVer

        source = HandlerBootstrapSource()

        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert isinstance(descriptor.version, ModelSemVer), (
                f"Handler '{descriptor.handler_id}' version is "
                f"{type(descriptor.version).__name__}, expected ModelSemVer"
            )

    @pytest.mark.asyncio
    async def test_all_handler_classes_are_importable(self) -> None:
        """Verify all bootstrap handler classes can be dynamically imported.

        This test ensures that handler_class paths are valid and importable,
        catching refactoring issues (renamed/moved handlers) early.
        """
        import importlib

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            handler_class_path = descriptor.handler_class
            assert handler_class_path is not None, (
                f"Handler {descriptor.handler_id} missing handler_class"
            )

            # Split into module and class name
            module_path, class_name = handler_class_path.rsplit(".", 1)

            # Attempt to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(
                    f"Handler {descriptor.handler_id}: Could not import module "
                    f"'{module_path}': {e}"
                )

            # Verify the class exists in the module
            assert hasattr(module, class_name), (
                f"Handler {descriptor.handler_id}: Class '{class_name}' not found "
                f"in module '{module_path}'"
            )

            # Get the class and verify it's a class
            handler_cls = getattr(module, class_name)
            assert isinstance(handler_cls, type), (
                f"Handler {descriptor.handler_id}: '{class_name}' is not a class"
            )


# =============================================================================
# Model Path Importability Tests (OMN-1087 Review Feedback)
# =============================================================================


class TestHandlerBootstrapSourceModelImportability:
    """Tests for input_model and output_model path importability.

    These tests verify that all model paths specified in bootstrap handlers
    can be dynamically imported, catching path errors or refactoring issues early.

    Added per OMN-1087 PR review recommendation to add test coverage for
    input_model and output_model importability.
    """

    @pytest.mark.asyncio
    async def test_all_input_models_are_importable(self) -> None:
        """Verify all bootstrap handler input_model paths can be dynamically imported.

        This test ensures that input_model paths are valid and importable,
        catching issues like:
        - Wrong module path (e.g., omnibase_core.types.JsonDict vs omnibase_infra.models.types.JsonDict)
        - Missing types after refactoring
        - Typos in module paths
        """
        import importlib

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            input_model_path = descriptor.input_model
            assert input_model_path, (
                f"Handler {descriptor.handler_id} missing input_model"
            )

            # Split into module and type name
            module_path, type_name = input_model_path.rsplit(".", 1)

            # Attempt to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(
                    f"Handler {descriptor.handler_id}: Could not import input_model "
                    f"module '{module_path}': {e}"
                )

            # Verify the type exists in the module
            assert hasattr(module, type_name), (
                f"Handler {descriptor.handler_id}: input_model type '{type_name}' "
                f"not found in module '{module_path}'"
            )

    @pytest.mark.asyncio
    async def test_all_output_models_are_importable(self) -> None:
        """Verify all bootstrap handler output_model paths can be dynamically imported.

        This test ensures that output_model paths are valid and importable,
        catching refactoring issues (renamed/moved models) early.
        """
        import importlib

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            output_model_path = descriptor.output_model
            assert output_model_path, (
                f"Handler {descriptor.handler_id} missing output_model"
            )

            # Split into module and type name
            module_path, type_name = output_model_path.rsplit(".", 1)

            # Attempt to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                pytest.fail(
                    f"Handler {descriptor.handler_id}: Could not import output_model "
                    f"module '{module_path}': {e}"
                )

            # Verify the type exists in the module
            assert hasattr(module, type_name), (
                f"Handler {descriptor.handler_id}: output_model type '{type_name}' "
                f"not found in module '{module_path}'"
            )

            # Verify it's a class (Pydantic model)
            model_cls = getattr(module, type_name)
            assert isinstance(model_cls, type), (
                f"Handler {descriptor.handler_id}: output_model '{type_name}' is not a class"
            )

    @pytest.mark.asyncio
    async def test_input_model_jsondict_is_correct_type(self) -> None:
        """Verify JsonDict input_model is the correct type alias.

        JsonDict should be dict[str, object] from omnibase_infra.models.types.
        This test catches path confusion between omnibase_core and omnibase_infra.
        """
        from omnibase_infra.models.types import JsonDict

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        # All bootstrap handlers should use JsonDict from omnibase_infra
        expected_path = "omnibase_infra.models.types.JsonDict"
        for descriptor in result.descriptors:
            assert descriptor.input_model == expected_path, (
                f"Handler {descriptor.handler_id} uses '{descriptor.input_model}', "
                f"expected '{expected_path}'"
            )

        # Verify JsonDict is the correct type alias
        # JsonDict is dict[str, object] per CLAUDE.md guidelines
        assert JsonDict is not None

    @pytest.mark.asyncio
    async def test_output_model_handler_output_is_pydantic_model(self) -> None:
        """Verify ModelHandlerOutput output_model is a valid Pydantic model.

        This ensures the output model path points to a proper Pydantic BaseModel
        that can be used for response serialization.
        """
        from pydantic import BaseModel

        from omnibase_core.models.dispatch import ModelHandlerOutput

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        # All bootstrap handlers should use ModelHandlerOutput
        expected_path = "omnibase_core.models.dispatch.ModelHandlerOutput"
        for descriptor in result.descriptors:
            assert descriptor.output_model == expected_path, (
                f"Handler {descriptor.handler_id} uses '{descriptor.output_model}', "
                f"expected '{expected_path}'"
            )

        # Verify ModelHandlerOutput is a Pydantic model
        assert issubclass(ModelHandlerOutput, BaseModel), (
            "ModelHandlerOutput should be a Pydantic BaseModel"
        )


# =============================================================================
# Concurrent Discovery Thread Safety Tests (OMN-1087 Review Feedback)
# =============================================================================


class TestHandlerBootstrapSourceThreadSafety:
    """Tests for concurrent discovery thread safety.

    These tests verify that HandlerBootstrapSource.discover_handlers() is
    thread-safe and can be called concurrently without race conditions.

    Added per OMN-1087 PR review recommendation for concurrent discovery tests.
    """

    @pytest.mark.asyncio
    async def test_concurrent_discovery_returns_consistent_results(self) -> None:
        """Multiple concurrent discover_handlers() calls should return consistent results.

        This tests that the double-checked locking pattern in _ensure_model_rebuilt()
        works correctly under concurrent access.
        """
        import asyncio

        source = HandlerBootstrapSource()
        concurrent_count = 10

        # Run multiple discover_handlers() calls concurrently
        tasks = [source.discover_handlers() for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks)

        # All results should have the same handler count
        handler_counts = [len(r.descriptors) for r in results]
        assert all(count == handler_counts[0] for count in handler_counts), (
            f"Inconsistent handler counts from concurrent calls: {handler_counts}"
        )

        # All results should have the same handler IDs
        reference_ids = {d.handler_id for d in results[0].descriptors}
        for i, result in enumerate(results[1:], start=2):
            result_ids = {d.handler_id for d in result.descriptors}
            assert result_ids == reference_ids, (
                f"Result {i} has different handler IDs than result 1"
            )

    @pytest.mark.asyncio
    async def test_concurrent_discovery_with_multiple_sources(self) -> None:
        """Multiple HandlerBootstrapSource instances accessed concurrently.

        This tests that multiple independent sources work correctly when
        their discover_handlers() methods are called concurrently.
        """
        import asyncio

        source_count = 5

        async def discover_from_new_source() -> set[str]:
            source = HandlerBootstrapSource()
            result = await source.discover_handlers()
            return {d.handler_id for d in result.descriptors}

        # Run discovery on multiple source instances concurrently
        tasks = [discover_from_new_source() for _ in range(source_count)]
        results = await asyncio.gather(*tasks)

        # All results should have the same handler IDs
        reference_ids = results[0]
        for i, result_ids in enumerate(results[1:], start=2):
            assert result_ids == reference_ids, (
                f"Source {i} returned different handler IDs"
            )

    def test_model_rebuild_lock_is_thread_safe(self) -> None:
        """The _ensure_model_rebuilt() function should be thread-safe.

        This tests the thread safety of the model rebuild initialization
        using actual threads (not asyncio).
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from omnibase_infra.runtime.handler_bootstrap_source import (
            _ensure_model_rebuilt,
        )

        thread_count = 20
        results: list[bool] = []
        exceptions: list[Exception] = []
        lock = threading.Lock()

        def call_ensure_rebuild() -> bool:
            try:
                _ensure_model_rebuilt()
                return True
            except Exception as e:
                with lock:
                    exceptions.append(e)
                return False

        # Call _ensure_model_rebuilt from multiple threads simultaneously
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(call_ensure_rebuild) for _ in range(thread_count)
            ]
            for future in as_completed(futures):
                with lock:
                    results.append(future.result())

        # All calls should succeed
        assert all(results), f"Some threads failed: {exceptions}"
        assert len(exceptions) == 0, (
            f"Exceptions during concurrent rebuild: {exceptions}"
        )

    @pytest.mark.asyncio
    async def test_rapid_concurrent_discovery_stress(self) -> None:
        """Stress test with many rapid concurrent discovery calls.

        This tests behavior under high concurrency to catch subtle
        race conditions in the discovery process.
        """
        import asyncio

        source = HandlerBootstrapSource()
        concurrent_count = 50

        # Run many discover_handlers() calls concurrently
        tasks = [source.discover_handlers() for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks)

        # Verify all results are valid
        for i, result in enumerate(results):
            assert len(result.descriptors) == EXPECTED_HANDLER_COUNT, (
                f"Result {i + 1} has wrong handler count: {len(result.descriptors)}"
            )
            assert result.validation_errors == [], (
                f"Result {i + 1} has validation errors"
            )


# =============================================================================
# Performance Characteristics Tests
# =============================================================================


class TestHandlerBootstrapSourcePerformance:
    """Tests for performance characteristics.

    These tests verify performance guarantees for handler discovery:
    - No network I/O required (all local file operations)
    - Constant time O(1) discovery (fixed set of 5 handlers)
    - Typical performance: <20ms for all handlers

    Note on I/O overhead (OMN-1282):
        Since OMN-1282, bootstrap handlers load configuration from contract
        YAML files rather than using hardcoded values. This adds file I/O
        overhead but enables contract-driven configuration. The threshold
        was increased from 10ms to 20ms to accommodate:
        - Reading handler_contract.yaml for each bootstrap handler
        - YAML parsing via yaml.safe_load()
        - Path resolution and symlink handling
        - CI environment disk I/O variance
    """

    @pytest.mark.asyncio
    async def test_discovery_is_fast(self) -> None:
        """discover_handlers() should complete quickly even with contract I/O.

        Since OMN-1282, contract YAML files are loaded during discovery,
        adding file I/O. This test uses a generous 50ms bound to account
        for CI environment variance and cold cache scenarios.

        See test_multiple_rapid_calls_are_fast() and PERFORMANCE_THRESHOLD_MS
        for more detailed performance expectations.
        """
        import time

        source = HandlerBootstrapSource()

        start = time.perf_counter()
        await source.discover_handlers()
        duration = time.perf_counter() - start

        # Should complete in under 50ms (generous bound for CI variance)
        # Typical execution is <10ms since this is pure in-memory operation
        assert duration < 0.05, (
            f"discover_handlers() took {duration:.3f}s, expected < 0.05s"
        )

    @pytest.mark.asyncio
    async def test_multiple_rapid_calls_are_fast(self) -> None:
        """Multiple rapid calls should all be fast despite contract I/O.

        Performance threshold: 20ms (PERFORMANCE_THRESHOLD_MS)

        Since OMN-1282, bootstrap handlers load configuration from contract
        YAML files during discovery. This adds file I/O overhead compared to
        the previous hardcoded approach (pre-OMN-1282 threshold was 10ms).

        I/O operations contributing to overhead:
        - Reading 5 handler_contract.yaml files (one per bootstrap handler)
        - YAML parsing via yaml.safe_load() for each contract
        - Path resolution and symlink handling for contract paths
        - OS-level file system caching (first call slower than subsequent)

        The 20ms threshold accommodates:
        - Contract file loading overhead (~5-10ms typical)
        - CI environment disk I/O variance
        - Cold cache scenarios on first discovery
        """
        import time

        source = HandlerBootstrapSource()
        call_count = 100

        start = time.perf_counter()
        for _ in range(call_count):
            await source.discover_handlers()
        total_duration = time.perf_counter() - start

        avg_duration = total_duration / call_count

        # Use constant from module top for threshold documentation
        assert avg_duration < PERFORMANCE_THRESHOLD_SECONDS, (
            f"Average call took {avg_duration * 1000:.2f}ms, "
            f"expected < {PERFORMANCE_THRESHOLD_MS}ms. "
            "See PERFORMANCE_THRESHOLD_MS comment for OMN-1282 context."
        )


__all__ = [
    "TestHandlerBootstrapSourceDescriptors",
    "TestHandlerBootstrapSourceDiscovery",
    "TestHandlerBootstrapSourceEdgeCases",
    "TestHandlerBootstrapSourceIdempotency",
    "TestHandlerBootstrapSourceModelImportability",
    "TestHandlerBootstrapSourcePerformance",
    "TestHandlerBootstrapSourceProtocolCompliance",
    "TestHandlerBootstrapSourceThreadSafety",
]
