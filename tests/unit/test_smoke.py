"""Smoke tests for omnibase_infra package.

These tests verify that the package structure is correct and basic imports work.
All smoke tests should be fast (<100ms) and require no external dependencies.
"""

from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_package_import() -> None:
    """Verify omnibase_infra package can be imported.

    Basic package import smoke test - this is intentionally lightweight.
    We test deeper functionality in dedicated test modules. The purpose here
    is simply to ensure the package structure is valid and can be imported.
    """
    import omnibase_infra

    assert omnibase_infra is not None


@pytest.mark.smoke
def test_cli_module_import() -> None:
    """Verify CLI module can be imported and CLI entry point is callable."""
    from omnibase_infra.cli import commands

    assert commands is not None
    assert hasattr(commands, "cli")
    assert callable(commands.cli), "CLI entry point must be callable"


@pytest.mark.smoke
def test_validation_module_import() -> None:
    """Verify validation module can be imported and validators are callable.

    Validates that core validator functions are importable and callable.
    Actual validation logic is tested in dedicated validation test modules.
    This smoke test ensures the validation API surface is intact.
    """
    from omnibase_infra.validation import (
        validate_infra_all,
        validate_infra_architecture,
        validate_infra_contracts,
        validate_infra_patterns,
    )

    # Verify validators are not None and are callable
    assert validate_infra_all is not None
    assert callable(validate_infra_all), "validate_infra_all must be callable"

    assert validate_infra_architecture is not None
    assert callable(validate_infra_architecture), (
        "validate_infra_architecture must be callable"
    )

    assert validate_infra_contracts is not None
    assert callable(validate_infra_contracts), (
        "validate_infra_contracts must be callable"
    )

    assert validate_infra_patterns is not None
    assert callable(validate_infra_patterns), "validate_infra_patterns must be callable"


@pytest.mark.smoke
def test_submodule_structure() -> None:
    """Verify all committed submodules can be imported.

    Cross-subsystem import test - ensures all major submodules are importable
    without circular dependencies or missing __init__.py files.

    NOTE: If this test becomes slow (>100ms) as modules grow heavier with
    more infrastructure adapters, consider splitting into separate tests
    or using lazy imports for heavyweight components.
    """
    from omnibase_infra import (
        clients,
        enums,
        infrastructure,
        models,
        nodes,
        shared,
        utils,
    )

    # All committed modules should be importable
    assert clients is not None
    assert enums is not None
    assert infrastructure is not None
    assert models is not None
    assert nodes is not None
    assert shared is not None
    assert utils is not None


@pytest.mark.smoke
def test_validation_constants_exist() -> None:
    """Verify validation constants are defined with expected values.

    Path constant validation - these are hardcoded string checks which are
    somewhat brittle, but acceptable for smoke tests. They ensure validators
    are configured with correct default paths for infrastructure code.

    If project structure changes, these assertions will fail fast, which is
    the intended behavior (smoke test failing indicates configuration drift).
    """
    from omnibase_infra.validation.infra_validators import (
        INFRA_NODES_PATH,
        INFRA_SRC_PATH,
    )

    assert INFRA_SRC_PATH == "src/omnibase_infra/"
    assert INFRA_NODES_PATH == "src/omnibase_infra/nodes/"
