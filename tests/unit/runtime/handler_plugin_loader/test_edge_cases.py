# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader edge cases, file size limits, idempotency, and case sensitivity.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .conftest import MINIMAL_HANDLER_CONTRACT_YAML, MockValidHandler

# =============================================================================
# File Size Limit Tests
# =============================================================================


class TestHandlerPluginLoaderFileSizeLimit:
    """Tests for 10MB file size limit enforcement."""

    def test_rejects_file_exceeding_10mb_limit(self, tmp_path: Path) -> None:
        """Test that files exceeding MAX_CONTRACT_SIZE are rejected."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import (
            MAX_CONTRACT_SIZE,
            HandlerPluginLoader,
        )

        # Create a valid contract file
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="oversized.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create mock stat result with oversized file size
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = MAX_CONTRACT_SIZE + 1
        mock_stat_result.st_mode = 0o100644  # Regular file mode for is_file() check

        loader = HandlerPluginLoader()

        with patch.object(Path, "stat", return_value=mock_stat_result):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader.load_from_contract(contract_file)

        # Verify error message indicates size limit exceeded
        error_msg = str(exc_info.value).lower()
        assert "size" in error_msg or "limit" in error_msg or "exceeds" in error_msg

    def test_accepts_file_under_10mb_limit(self, tmp_path: Path) -> None:
        """Test that files under MAX_CONTRACT_SIZE are accepted."""
        from omnibase_infra.runtime.handler_plugin_loader import (
            MAX_CONTRACT_SIZE,
            HandlerPluginLoader,
        )

        # Create a valid contract file (small, under limit)
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="normal.size.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Verify file is under limit
        actual_size = contract_file.stat().st_size
        assert actual_size < MAX_CONTRACT_SIZE

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.handler_name == "normal.size.handler"


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestHandlerPluginLoaderIdempotency:
    """Tests for idempotency of load operations."""

    def test_load_from_contract_is_idempotent(self, valid_contract_path: Path) -> None:
        """Test that loading the same contract multiple times works correctly."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Load the same contract multiple times
        result1 = loader.load_from_contract(valid_contract_path)
        result2 = loader.load_from_contract(valid_contract_path)
        result3 = loader.load_from_contract(valid_contract_path)

        # All should return equivalent results
        assert result1.handler_name == result2.handler_name == result3.handler_name
        assert result1.handler_class == result2.handler_class == result3.handler_class
        assert result1.handler_type == result2.handler_type == result3.handler_type

    def test_load_from_directory_is_idempotent(
        self, valid_contract_directory: Path
    ) -> None:
        """Test loading from directory multiple times returns consistent results."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        result1 = loader.load_from_directory(valid_contract_directory)
        result2 = loader.load_from_directory(valid_contract_directory)
        result3 = loader.load_from_directory(valid_contract_directory)

        # All should return same number of handlers
        assert len(result1) == len(result2) == len(result3) == 3

        # Handler names should be consistent
        names1 = {h.handler_name for h in result1}
        names2 = {h.handler_name for h in result2}
        names3 = {h.handler_name for h in result3}

        assert names1 == names2 == names3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestHandlerPluginLoaderEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handles_single_tag_as_string(self, tmp_path: Path) -> None:
        """Test that single tag specified as string (not list) is handled."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: single.tag.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
capability_tags: single-tag
"""
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.capability_tags == ["single-tag"]

    def test_filters_non_string_tags(self, tmp_path: Path) -> None:
        """Test that non-string tags are filtered out."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: mixed.tags.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
capability_tags:
  - valid-tag
  - 123
  - true
  - another-valid
"""
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        # Only string tags should be included
        assert "valid-tag" in handler.capability_tags
        assert "another-valid" in handler.capability_tags

    def test_handler_name_whitespace_stripped(self, tmp_path: Path) -> None:
        """Test that whitespace in handler_name is stripped."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: "  whitespace.handler  "
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.handler_name == "whitespace.handler"

    def test_accepts_name_field_as_alternative_to_handler_name(
        self, tmp_path: Path
    ) -> None:
        """Test that 'name' field can be used instead of 'handler_name'."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
name: alternative.name.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.handler_name == "alternative.name.handler"

    def test_accepts_tags_field_as_alternative_to_capability_tags(
        self, tmp_path: Path
    ) -> None:
        """Test that 'tags' field can be used instead of 'capability_tags'."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: alternative.tags.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
tags:
  - alt-tag-1
  - alt-tag-2
"""
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert "alt-tag-1" in handler.capability_tags
        assert "alt-tag-2" in handler.capability_tags

    def test_loaded_at_timestamp_is_set(self, valid_contract_path: Path) -> None:
        """Test that loaded_at timestamp is set during load."""
        from datetime import UTC, datetime

        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        before_load = datetime.now(UTC)

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(valid_contract_path)

        after_load = datetime.now(UTC)

        assert handler.loaded_at >= before_load
        assert handler.loaded_at <= after_load

    def test_contract_path_is_resolved_to_absolute(self, tmp_path: Path) -> None:
        """Test that contract_path in result is resolved to absolute path."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="absolute.path.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        # Path should be absolute
        assert handler.contract_path.is_absolute()
        assert handler.contract_path == contract_file.resolve()


# =============================================================================
# Case Sensitivity Tests
# =============================================================================


class TestHandlerPluginLoaderCaseSensitivity:
    """Tests for case-sensitive file discovery."""

    def test_only_discovers_exact_filename_match(self, tmp_path: Path) -> None:
        """Test that only exact filename matches are discovered."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create correctly named file (should be discovered)
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="valid.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create incorrectly named files (should NOT be discovered)
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / "HANDLER_CONTRACT.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="uppercase.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        wrong_ext_dir = tmp_path / "wrong_ext"
        wrong_ext_dir.mkdir()
        (wrong_ext_dir / "handler_contract.yml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="wrong.ext.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(tmp_path)

        # Should only find the correctly named file
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"
