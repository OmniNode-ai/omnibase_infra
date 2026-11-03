# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Select the kernel plugins a lane enables from the node catalog (OMN-21591).

The kernel used to find its two built-in plugins by scanning an entry-point group
for a name written in code. Boot step 0 now announces every installed plugin as an
entry of kind ``kernel_plugin``; the lane overlay's ``runtime.kernel_plugins`` names
the ones this lane enables. This handler is the join: an announced entry the overlay
names is selected, one announced and not named is left out, and a name the catalog
does not hold as an admitted ``kernel_plugin`` is a typed boot error. The group name
appears only in the announcer's metadata reader; this module reads the catalog.
"""

from __future__ import annotations

from importlib import import_module
from uuid import UUID, uuid4

from omnibase_core.enums.enum_announced_node_kind import EnumAnnouncedNodeKind
from omnibase_core.enums.enum_catalog_entry_state import EnumCatalogEntryState
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.node_announce.model_kernel_plugin_selection_request import (
    ModelKernelPluginSelectionRequest,
)
from omnibase_infra.models.node_announce.model_kernel_plugin_selection_result import (
    ModelKernelPluginSelectionResult,
)
from omnibase_infra.models.node_announce.model_selected_kernel_plugin import (
    ModelSelectedKernelPlugin,
)


class KernelPluginNotAnnouncedError(RuntimeHostError):
    """The lane overlay enables a kernel plugin the catalog does not hold (boot error).

    Raised before the container exists, so a lane that names a plugin no installed
    package announces never starts.

    Attributes:
        plugin_names: The enabled names that are not admitted ``kernel_plugin`` entries.
    """

    def __init__(
        self,
        message: str,
        *,
        plugin_names: tuple[str, ...] = (),
        correlation_id: UUID | None = None,
    ) -> None:
        self.plugin_names = plugin_names
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONFIGURATION_ERROR,
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="kernel_plugin_selection",
                correlation_id=correlation_id or uuid4(),
            ),
            plugin_names=",".join(plugin_names),
        )


def load_kernel_plugin(selected: ModelSelectedKernelPlugin) -> object:
    """Import and instantiate one selected plugin from its announced ``module:attribute``."""
    module_name, _, attribute = selected.target.partition(":")
    target: object = import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Kernel plugin target {selected.target!r} is not callable")
    return target()


class HandlerKernelPluginSelection:
    """Pure join of the overlay's enabled names with the catalog's ``kernel_plugin`` entries."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelKernelPluginSelectionRequest
    ) -> ModelKernelPluginSelectionResult:
        """Return the enabled plugins in the overlay's order; refuse any the catalog lacks."""
        announced = request.announced
        targets = (
            {(p.name, p.distribution): p.target for p in announced.inventory.plugins}
            if announced is not None
            else {}
        )
        selected: list[ModelSelectedKernelPlugin] = []
        missing: list[str] = []
        for name in request.enabled:
            entry = announced.catalog.lookup(name) if announced is not None else None
            target = (
                targets.get((name, entry.package.distribution))
                if entry is not None
                else None
            )
            if (
                entry is None
                or entry.node_kind is not EnumAnnouncedNodeKind.KERNEL_PLUGIN
                or entry.state is not EnumCatalogEntryState.ADMITTED
                or target is None
            ):
                missing.append(name)
                continue
            selected.append(
                ModelSelectedKernelPlugin(
                    name=name, distribution=entry.package.distribution, target=target
                )
            )
        if missing:
            raise KernelPluginNotAnnouncedError(
                "The lane overlay's runtime.kernel_plugins names "
                f"{', '.join(missing)}, which the node catalog does not hold as an "
                "admitted kernel_plugin entry; install the package that announces it "
                "or remove the name from the overlay",
                plugin_names=tuple(missing),
            )
        return ModelKernelPluginSelectionResult(selected=tuple(selected))


__all__ = [
    "HandlerKernelPluginSelection",
    "KernelPluginNotAnnouncedError",
    "load_kernel_plugin",
]
