# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12162: verify register_handlers_from_config stub is fully removed."""

import pytest


@pytest.mark.unit
def test_register_handlers_from_config_not_in_module() -> None:
    import omnibase_infra.runtime.handler_registry as registry_module

    assert not hasattr(registry_module, "register_handlers_from_config"), (
        "stub register_handlers_from_config must be removed (OMN-12162)"
    )


@pytest.mark.unit
def test_register_handlers_from_config_not_in_runtime_init() -> None:
    import omnibase_infra.runtime as runtime_module

    assert not hasattr(runtime_module, "register_handlers_from_config"), (
        "stub register_handlers_from_config must not be re-exported from runtime (OMN-12162)"
    )


@pytest.mark.unit
def test_register_handlers_from_config_not_in_all() -> None:
    import omnibase_infra.runtime.handler_registry as registry_module

    all_exports: list[str] = getattr(registry_module, "__all__", [])
    assert "register_handlers_from_config" not in all_exports, (
        "register_handlers_from_config must not appear in handler_registry.__all__ (OMN-12162)"
    )
