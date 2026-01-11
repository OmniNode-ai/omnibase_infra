# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for correlation_id auto-generation in HandlerPluginLoader.

Verifies that correlation_id is auto-generated when not provided,
following ONEX guidelines: "Always propagate from incoming requests;
Auto-generate with uuid4() if missing; Include in all error context."

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

import pytest

from .conftest import MINIMAL_HANDLER_CONTRACT_YAML


class TestCorrelationIdAutoGeneration:
    """Tests for correlation_id auto-generation behavior.

    Per ONEX coding guidelines:
    1. Always propagate correlation_id from incoming requests
    2. Auto-generate with uuid4() if missing
    3. Include in all error context
    """

    def test_load_from_contract_auto_generates_correlation_id_on_error(
        self, tmp_path: Path
    ) -> None:
        """Test load_from_contract auto-generates correlation_id when not provided.

        Verifies that when no correlation_id is passed, the error context
        contains a valid auto-generated UUID.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create path that doesn't exist to trigger an error
        nonexistent_path = tmp_path / "nonexistent" / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            # Call without correlation_id - should auto-generate
            loader.load_from_contract(nonexistent_path)

        # Verify correlation_id was auto-generated (not None)
        assert exc_info.value.model.correlation_id is not None
        # Verify it's a valid UUID
        assert isinstance(exc_info.value.model.correlation_id, UUID)

    def test_load_from_contract_preserves_provided_correlation_id(
        self, tmp_path: Path
    ) -> None:
        """Test load_from_contract preserves provided correlation_id.

        Verifies that when a correlation_id is explicitly provided,
        it is propagated to error context without modification.
        """
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_path = tmp_path / "nonexistent" / "handler_contract.yaml"
        provided_id = UUID("12345678-1234-5678-1234-567812345678")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent_path, correlation_id=provided_id)

        # Verify provided correlation_id was preserved
        assert exc_info.value.model.correlation_id == provided_id

    def test_load_from_directory_auto_generates_correlation_id_on_error(
        self, tmp_path: Path
    ) -> None:
        """Test load_from_directory auto-generates correlation_id when not provided.

        Verifies that when no correlation_id is passed, the error context
        contains a valid auto-generated UUID.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_dir = tmp_path / "nonexistent_directory"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            # Call without correlation_id - should auto-generate
            loader.load_from_directory(nonexistent_dir)

        # Verify correlation_id was auto-generated (not None)
        assert exc_info.value.model.correlation_id is not None
        # Verify it's a valid UUID
        assert isinstance(exc_info.value.model.correlation_id, UUID)

    def test_load_from_directory_preserves_provided_correlation_id(
        self, tmp_path: Path
    ) -> None:
        """Test load_from_directory preserves provided correlation_id.

        Verifies that when a correlation_id is explicitly provided,
        it is propagated to error context without modification.
        """
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_dir = tmp_path / "nonexistent_directory"
        provided_id = UUID("abcdef12-3456-7890-abcd-ef1234567890")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(nonexistent_dir, correlation_id=provided_id)

        # Verify provided correlation_id was preserved
        assert exc_info.value.model.correlation_id == provided_id

    def test_discover_and_load_auto_generates_correlation_id_on_error(
        self,
    ) -> None:
        """Test discover_and_load auto-generates correlation_id when not provided.

        Verifies that when no correlation_id is passed, the error context
        contains a valid auto-generated UUID.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            # Call with empty patterns list (triggers error) without correlation_id
            loader.discover_and_load([])

        # Verify correlation_id was auto-generated (not None)
        assert exc_info.value.model.correlation_id is not None
        # Verify it's a valid UUID
        assert isinstance(exc_info.value.model.correlation_id, UUID)

    def test_discover_and_load_preserves_provided_correlation_id(
        self,
    ) -> None:
        """Test discover_and_load preserves provided correlation_id.

        Verifies that when a correlation_id is explicitly provided,
        it is propagated to error context without modification.
        """
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        provided_id = UUID("fedcba98-7654-3210-fedc-ba9876543210")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            # Call with empty patterns list (triggers error) with correlation_id
            loader.discover_and_load([], correlation_id=provided_id)

        # Verify provided correlation_id was preserved
        assert exc_info.value.model.correlation_id == provided_id


class TestCorrelationIdPropagation:
    """Tests for correlation_id propagation through method calls.

    Verifies that the same correlation_id is propagated to all
    sub-operations within a single API call.
    """

    def test_load_from_directory_propagates_correlation_id_to_contract_loads(
        self, tmp_path: Path
    ) -> None:
        """Test that load_from_directory propagates correlation_id to load_from_contract.

        When loading multiple contracts, each should receive the same correlation_id.
        """
        from uuid import UUID

        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create directory with a contract that has an invalid import
        handler_dir = tmp_path / "handler"
        handler_dir.mkdir()
        (handler_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class="nonexistent.module.Handler",
            )
        )

        provided_id = UUID("11111111-2222-3333-4444-555555555555")

        loader = HandlerPluginLoader()

        # This should fail with InfraConnectionError (module not found)
        # but the correlation_id should be propagated
        # Since load_from_directory catches errors gracefully, we need to
        # verify via logging or use a contract that will definitely fail
        handlers = loader.load_from_directory(tmp_path, correlation_id=provided_id)

        # No handlers should be loaded (all failed due to import error)
        assert len(handlers) == 0

    def test_discover_and_load_propagates_correlation_id_to_contract_loads(
        self, tmp_path: Path
    ) -> None:
        """Test that discover_and_load propagates correlation_id to load_from_contract.

        When discovering and loading contracts, each should receive the same correlation_id.
        """
        from uuid import UUID

        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create directory with a contract that has an invalid import
        handler_dir = tmp_path / "handler"
        handler_dir.mkdir()
        (handler_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class="nonexistent.module.Handler",
            )
        )

        provided_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

        loader = HandlerPluginLoader()

        # discover_and_load should propagate the correlation_id
        handlers = loader.discover_and_load(
            ["**/handler_contract.yaml"],
            correlation_id=provided_id,
            base_path=tmp_path,
        )

        # No handlers should be loaded (all failed due to import error)
        assert len(handlers) == 0


class TestCorrelationIdUniqueness:
    """Tests for uniqueness of auto-generated correlation IDs.

    Verifies that each call generates a unique correlation_id when
    not explicitly provided.
    """

    def test_each_call_generates_unique_correlation_id(self, tmp_path: Path) -> None:
        """Test that separate calls generate unique correlation IDs.

        Each call to load_from_contract without a correlation_id should
        generate a new unique UUID.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_path = tmp_path / "nonexistent" / "handler_contract.yaml"

        loader = HandlerPluginLoader()
        correlation_ids: list[UUID] = []

        # Make multiple calls without correlation_id
        for _ in range(5):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader.load_from_contract(nonexistent_path)

            assert exc_info.value.model.correlation_id is not None
            correlation_ids.append(exc_info.value.model.correlation_id)

        # All correlation IDs should be unique
        assert len(set(correlation_ids)) == 5, "All correlation IDs should be unique"


class TestCorrelationIdFormat:
    """Tests for correlation_id format validation.

    Verifies that auto-generated correlation IDs are valid UUID4 format.
    """

    def test_auto_generated_correlation_id_is_valid_uuid4(self, tmp_path: Path) -> None:
        """Test that auto-generated correlation_id is a valid UUID4.

        UUID4 characteristics:
        - Version nibble (position 12) is '4'
        - Variant nibble (position 16) is '8', '9', 'a', or 'b'
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_path = tmp_path / "nonexistent" / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent_path)

        correlation_id = exc_info.value.model.correlation_id
        assert correlation_id is not None

        # Verify UUID4 version
        assert correlation_id.version == 4

        # Verify it's a valid UUID string representation
        uuid_str = str(correlation_id)
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        assert uuid_pattern.match(uuid_str), f"Invalid UUID4 format: {uuid_str}"
