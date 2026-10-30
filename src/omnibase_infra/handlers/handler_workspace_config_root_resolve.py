# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve a workspace's config owner under the caller's directory policy."""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.workspace import (
    ModelWorkspaceConfigRootRequest,
    ModelWorkspaceConfigRootResolution,
)
from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

# Bootstrap location for the workspace's tier-1 config owner.
ENV_WORKSPACE_CONFIG_ROOT = "ONEX_WORKSPACE_CONFIG_ROOT"
# OMN-20700: public lab-onboarding.sh phase 2 writes developer tier-1 config
# here, relative to the workspace root.
DEVELOPER_WORKSPACE_CONFIG_RELATIVE_PATH = Path(".onex") / "workspace-config"


class HandlerWorkspaceConfigRootResolve:
    """Resolve the config owner, independently of the registry's state root.

    ONEX_WORKSPACE_CONFIG_ROOT is a bootstrap location, never a transport
    selector, and the deployment supplies it (its overlay or host installer).
    Resolution order is a non-blank environment override, then
    ``.onex/workspace-config`` if it is a directory. Otherwise no root is
    declared and ``config_root`` is None with source ``unset``: no directory
    name is guessed, and each caller applies its own neutral default or refuses
    (fail-fast environment). A developer install never needs a private
    repository (OMN-20700). Resolved results are absolute.

    A supplied environment replaces os.environ, including when it is empty.
    Setting allow_developer_config to false skips the developer directory.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify this infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Directory checks perform filesystem I/O."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelWorkspaceConfigRootRequest
    ) -> ModelWorkspaceConfigRootResolution:
        """Resolve the config root and report the winning rule."""
        configured = (
            expand_contract_env_refs(f"${{env.{ENV_WORKSPACE_CONFIG_ROOT}}}")
            if request.environment is None
            else request.environment.get(ENV_WORKSPACE_CONFIG_ROOT, "")
        ).strip()
        if configured:
            return ModelWorkspaceConfigRootResolution(
                config_root=Path(configured).resolve(), source="environment"
            )
        if request.allow_developer_config:
            developer = (
                request.workspace_root / DEVELOPER_WORKSPACE_CONFIG_RELATIVE_PATH
            )
            if developer.is_dir():
                return ModelWorkspaceConfigRootResolution(
                    config_root=developer.resolve(), source="developer"
                )
        return ModelWorkspaceConfigRootResolution(config_root=None, source="unset")

    def require(self, request: ModelWorkspaceConfigRootRequest) -> Path:
        """Resolve the config root, or raise FileNotFoundError naming the variable.

        For callers that have no neutral default and so refuse when none is
        declared (fail-fast environment).
        """
        root = self.handle(request).config_root
        if root is None:
            raise FileNotFoundError(
                f"no workspace config root is declared: set {ENV_WORKSPACE_CONFIG_ROOT}"
            )
        return root
