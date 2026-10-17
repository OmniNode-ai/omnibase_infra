# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime's core wheel supplies the Claude Code binding handler models."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_runtime_core_can_round_trip_binding_install_request() -> None:
    """Exercise the types imported by the render, install and hook handlers."""
    from omnibase_core.enums.enum_binding_kind import EnumBindingKind
    from omnibase_core.enums.enum_client_max_mode import EnumClientMaxMode
    from omnibase_core.models.claude_code_binding import (
        ModelBindingInstallRequest,
        ModelBindingInstallResult,
        ModelClientArgument,
        ModelClientBinding,
    )

    binding = ModelClientBinding(
        name="session_start",
        kind=EnumBindingKind.ADDITIVE,
        hook_event="SessionStart",
        command_topic="binding-command",
        command_type="BindingCommand",
        arguments=(ModelClientArgument(name="session_id", source="$.session_id"),),
        verdict_compute="binding_verdict",
        deadline_ms=1000,
    )
    request = ModelBindingInstallRequest(
        bindings=(binding,),
        max_mode=EnumClientMaxMode.OBSERVE,
        mods_dir="mods",
        settings_path="settings.json",
        supported_probe_rows=(),
    )

    assert (
        ModelBindingInstallRequest.model_validate_json(request.model_dump_json())
        == request
    )
    assert request.bindings[0].kind is EnumBindingKind.ADDITIVE
    assert "installed_bindings" in ModelBindingInstallResult.model_fields
