"""Smoke tests for omnibase_infra package.

These tests verify that the package structure is correct and basic imports work.
All smoke tests should be fast (<100ms) and require no external dependencies.
"""

from __future__ import annotations

import pytest


@pytest.mark.smoke
def test_package_import() -> None:
    """Verify omnibase_infra package can be imported."""
    import omnibase_infra

    assert omnibase_infra is not None


@pytest.mark.smoke
def test_cli_module_import() -> None:
    """Verify CLI module can be imported."""
    from omnibase_infra.cli import commands

    assert commands is not None
    assert hasattr(commands, "cli")
    assert callable(commands.cli)


@pytest.mark.smoke
def test_validation_module_import() -> None:
    """Verify validation module can be imported."""
    from omnibase_infra.validation import (
        validate_infra_all,
        validate_infra_architecture,
        validate_infra_contracts,
        validate_infra_patterns,
    )

    assert validate_infra_all is not None
    assert validate_infra_architecture is not None
    assert validate_infra_contracts is not None
    assert validate_infra_patterns is not None


@pytest.mark.smoke
def test_submodule_structure() -> None:
    """Verify all committed submodules can be imported."""
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
    """Verify validation constants are defined."""
    from omnibase_infra.validation.infra_validators import (
        INFRA_NODES_PATH,
        INFRA_SRC_PATH,
    )

    assert INFRA_SRC_PATH == "src/omnibase_infra/"
    assert INFRA_NODES_PATH == "src/omnibase_infra/nodes/"
