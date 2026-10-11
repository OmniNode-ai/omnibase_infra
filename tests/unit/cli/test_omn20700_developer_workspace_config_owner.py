# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A fresh developer workspace finds its config owner without a private repo (OMN-20700).

THE DEFECT. OMN-19743 moved the workspace config owner out of the registry root:
``ONEX_WORKSPACE_CONFIG_ROOT`` names it, and with the variable unset the owner is
the sibling ``../omnibase_internal`` -- a private repository the public
onboarding never clones. The onboarding script (omniclaude
``plugins/onex/skills/_bin/lab-onboarding.sh``, phase 2) writes the developer
workspace's tier-1 config under ``<workspace>/.onex/workspace-config`` and exports
the variable, but only in its own process and in the shell profile it writes. A
shell that did not read that profile still binds the workspace root (the ``onex``
wrapper derives ``OMNIBASE_PATH`` itself), so ``onex delegate`` looked for
``../omnibase_internal``, found nothing, and refused with a remedy that pointed at
the private repository.

THE FIX. With the variable unset, a workspace whose sibling operator repository
is absent takes the developer config directory the onboarding writes. A registry
workspace -- one whose sibling exists -- is unchanged, and a bound root with no
config anywhere is still refused (OMN-19193), now with a typed error naming the
variable, the developer location and the onboarding step that writes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime import service_kernel
from omnibase_infra.runtime.service_kernel import (
    WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH,
    resolve_embedded_runtime_config,
    workspace_runtime_config_root,
)

pytestmark = pytest.mark.unit

#: The file the onboarding script writes in phase 2, byte for byte in its keys.
_DEVELOPER_TIER1 = """\
# Written by omninode-dev-setup.
description: "Developer workspace tier-1 runtime configuration: in-memory bus, local profile"
input_topic: "requests"
output_topic: "responses"
group_id: "onex-runtime"
event_bus:
  type: "inmemory"
  profile: "local"
  environment: "local"
  max_history: 1000
  circuit_breaker_threshold: 5
"""

_REGISTRY_TIER1 = """
description: "registry tier-1 runtime config (test)"
event_bus:
  type: "kafka"
  profile: "local"
  lane: "dev"
"""

_RUNTIME_CONFIG = (
    WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH / "runtime" / ("runtime_config.yaml")
)


def _write(owner: Path, text: str) -> Path:
    path = owner / _RUNTIME_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A developer workspace: public clones only, no sibling operator repository."""
    root = tmp_path / "code" / "omni"
    root.mkdir(parents=True)
    return root


def test_a_fresh_developer_workspace_resolves_the_config_onboarding_wrote(
    workspace: Path,
) -> None:
    written = _write(workspace / ".onex" / "workspace-config", _DEVELOPER_TIER1)
    config, source = resolve_embedded_runtime_config(workspace_root=workspace)
    assert config.event_bus.type is EnumEventBusType.INMEMORY
    assert str(written) in source
    assert "omnibase_internal" not in source


def test_a_registry_workspace_still_takes_its_sibling_owner(workspace: Path) -> None:
    sibling = workspace.parent / "omnibase_internal"
    owner_config = _write(sibling, _REGISTRY_TIER1)
    _write(workspace / ".onex" / "workspace-config", _DEVELOPER_TIER1)
    assert workspace_runtime_config_root(workspace) == sibling.resolve()
    config, source = resolve_embedded_runtime_config(workspace_root=workspace)
    assert config.event_bus.lane == "dev"
    assert str(owner_config) in source


def test_a_registry_sibling_without_config_is_still_refused(workspace: Path) -> None:
    """OMN-19193 is not weakened: a developer file never answers for a registry."""
    (workspace.parent / "omnibase_internal").mkdir()
    _write(workspace / ".onex" / "workspace-config", _DEVELOPER_TIER1)
    with pytest.raises(ProtocolConfigurationError):
        resolve_embedded_runtime_config(workspace_root=workspace)


def test_the_variable_still_outranks_both(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = tmp_path / "named-owner"
    owner_config = _write(named, _REGISTRY_TIER1)
    _write(workspace.parent / "omnibase_internal", _DEVELOPER_TIER1)
    _write(workspace / ".onex" / "workspace-config", _DEVELOPER_TIER1)
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(named))
    config, source = resolve_embedded_runtime_config(workspace_root=workspace)
    assert config.event_bus.lane == "dev"
    assert str(owner_config) in source


def test_no_owner_anywhere_is_a_typed_refusal_that_names_the_public_fix(
    workspace: Path,
) -> None:
    error_type = getattr(service_kernel, "WorkspaceRuntimeConfigMissingError", None)
    assert error_type is not None, "the refusal has no type of its own"
    assert issubclass(error_type, ProtocolConfigurationError)
    with pytest.raises(error_type) as exc:
        resolve_embedded_runtime_config(workspace_root=workspace)
    message = str(exc.value)
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in message
    assert str(workspace.resolve() / ".onex" / "workspace-config") in message
    assert "lab-onboarding.sh" in message
    assert "--bus inmemory" in message
