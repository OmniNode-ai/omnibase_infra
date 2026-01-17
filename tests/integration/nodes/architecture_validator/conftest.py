# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for architecture validator integration tests.

This module provides common fixtures used across architecture validator
integration tests.

Fixtures Provided:
    - container: ModelONEXContainer instance
    - validator: NodeArchitectureValidator instance
    - all_rules: Tuple of all three rule class instances
    - project_temp_dir: Temporary directory within project root

Related:
    - tests/unit/nodes/architecture_validator/conftest.py: Unit test fixtures
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.nodes.architecture_validator.node import NodeArchitectureValidator
from omnibase_infra.nodes.architecture_validator.validators import (
    RuleNoDirectDispatch,
    RuleNoHandlerPublishing,
    RuleNoOrchestratorFSM,
)


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create ONEX container for tests.

    Returns:
        A fresh ModelONEXContainer instance.
    """
    return ModelONEXContainer()


@pytest.fixture
def validator(container: ModelONEXContainer) -> NodeArchitectureValidator:
    """Create NodeArchitectureValidator instance.

    Args:
        container: ONEX container for dependency injection.

    Returns:
        A configured NodeArchitectureValidator instance.
    """
    return NodeArchitectureValidator(container)


@pytest.fixture
def all_rules() -> tuple[
    RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
]:
    """Create instances of all three rule classes.

    Returns:
        Tuple containing one instance of each rule class:
        - RuleNoDirectDispatch (ARCH-001)
        - RuleNoHandlerPublishing (ARCH-002)
        - RuleNoOrchestratorFSM (ARCH-003)
    """
    return (
        RuleNoDirectDispatch(),
        RuleNoHandlerPublishing(),
        RuleNoOrchestratorFSM(),
    )


@pytest.fixture
def project_temp_dir() -> Iterator[Path]:
    """Create temporary directory within project for testing.

    The NodeArchitectureValidator has security validation that rejects
    absolute paths outside the working directory. This fixture creates
    temp files within the project directory (but NOT in tests/ to avoid
    exemptions from ARCH-001 rule).

    Yields:
        Path to temporary directory within project root.

    Note:
        Directory is automatically cleaned up after test completion.

    Raises:
        pytest.fail: If pyproject.toml cannot be found (project root not detected).
    """
    current = Path(__file__).resolve()
    project_root = current.parent
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    if not (project_root / "pyproject.toml").exists():
        pytest.fail(
            "Could not find pyproject.toml in any parent directory. "
            "Ensure tests are run from within the project repository."
        )

    temp_dir = project_root / f"_test_tmp_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def create_temp_file(project_temp_dir: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary Python files.

    Args:
        project_temp_dir: Temporary directory within project.

    Returns:
        A callable that takes (filename, content) and returns the Path
        to the created file.

    Example::

        def test_something(create_temp_file):
            path = create_temp_file("service.py", "class MyService: pass")
            result = rule.check(str(path))
            assert result.passed

    """

    def _create(filename: str, content: str) -> Path:
        """Create a temporary Python file with the given content.

        Args:
            filename: Name of the file to create.
            content: Python source code to write.

        Returns:
            Path to the created file.
        """
        file_path = project_temp_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create
