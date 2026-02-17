# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the no-direct-adapter-usage architecture rule."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from omnibase_infra.validation.validator_no_direct_adapter import (
    ModelAdapterViolation,
    check_no_direct_adapter_usage,
)


@pytest.fixture
def temp_src() -> Path:
    """Create a temporary source directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_root = Path(tmpdir) / "src" / "omnibase_infra"
        src_root.mkdir(parents=True)
        yield src_root


class TestNoDirectAdapterRule:
    """Test the architecture rule validator."""

    def test_no_violations_clean_code(self, temp_src: Path) -> None:
        """Test no violations in code that doesn't import _internal."""
        (temp_src / "my_module.py").write_text(
            "from omnibase_infra.handlers import HandlerInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 0

    def test_violation_direct_import(self, temp_src: Path) -> None:
        """Test violation when directly importing _internal adapter."""
        (temp_src / "bad_module.py").write_text(
            "from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 1
        assert (
            violations[0].module_imported
            == "omnibase_infra.adapters._internal.adapter_infisical"
        )

    def test_handler_allowed(self, temp_src: Path) -> None:
        """Test handler modules are allowed to import adapters."""
        handler_dir = temp_src / "handlers"
        handler_dir.mkdir()
        (handler_dir / "handler_infisical.py").write_text(
            "from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 0

    def test_test_dir_files_allowed(self, temp_src: Path) -> None:
        """Test files inside a tests/ directory are allowed to import adapters."""
        tests_dir = temp_src / "tests" / "unit"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_adapter_infisical.py").write_text(
            "from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 0

    def test_test_prefix_in_production_code_not_allowed(self, temp_src: Path) -> None:
        """Test that a test_* file outside a test directory IS flagged."""
        (temp_src / "test_adapter_infisical.py").write_text(
            "from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 1

    def test_internal_module_allowed(self, temp_src: Path) -> None:
        """Test _internal modules themselves are allowed."""
        internal_dir = temp_src / "adapters" / "_internal"
        internal_dir.mkdir(parents=True)
        (internal_dir / "__init__.py").write_text(
            "from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 0

    def test_violation_import_statement(self, temp_src: Path) -> None:
        """Test violation with import statement (not from...import)."""
        (temp_src / "bad2.py").write_text(
            "import omnibase_infra.adapters._internal.adapter_infisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 1

    def test_violation_details(self, temp_src: Path) -> None:
        """Test violation has correct details."""
        (temp_src / "violator.py").write_text(
            "from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical\n"
        )
        violations = check_no_direct_adapter_usage(temp_src)
        assert len(violations) == 1
        v = violations[0]
        assert isinstance(v, ModelAdapterViolation)
        assert "violator.py" in v.file_path
        assert v.line_number == 1
        assert "not allowed" in v.message
