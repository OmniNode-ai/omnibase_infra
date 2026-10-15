# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply a planned diff under policy, a fresh-generation check and the lease."""

from collections.abc import Mapping

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.declared_state.enum_declared_state_apply_status import (
    EnumDeclaredStateApplyStatus,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_request import (
    ModelDeclaredStateApplyRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_result import (
    ModelDeclaredStateApplyResult,
)
from omnibase_infra.protocols.protocol_declared_state_lease import (
    ProtocolDeclaredStateLease,
)
from omnibase_infra.protocols.protocol_declared_state_surface import (
    ProtocolDeclaredStateSurface,
)


class HandlerDeclaredStateApply:
    """Change a surface only when policy allows, it was readable and it has not moved.

    Mirrors formal/declared_state_reconcile: the lease is taken before the first
    change, the generation is checked before every change, and the lease is
    released on every exit.
    """

    def __init__(
        self,
        surfaces: Mapping[EnumDeclaredStateSurfaceKind, ProtocolDeclaredStateSurface],
        lease: ProtocolDeclaredStateLease,
    ) -> None:
        self._surfaces = surfaces
        self._lease = lease

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the effect role."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the side-effecting category."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelDeclaredStateApplyRequest
    ) -> ModelDeclaredStateApplyResult:
        """Apply the diff's operations in order, stopping at the first refusal."""
        diff = request.diff
        kind = diff.surface_kind
        if kind not in request.policy.allowed_kinds:
            allowed = sorted(k.value for k in request.policy.allowed_kinds)
            return self._refused(
                EnumDeclaredStateApplyStatus.REFUSED_POLICY,
                0,
                f"surface kind '{kind.value}' is not allowed by the apply policy "
                f"(allowed: {allowed}); it stays observe-only",
            )
        if diff.unobservable_reason is not None:
            return self._refused(
                EnumDeclaredStateApplyStatus.REFUSED_UNOBSERVABLE,
                0,
                f"surface {kind.value}/{diff.surface_key} is unobservable: "
                f"{diff.unobservable_reason}",
            )
        surface = self._surfaces.get(kind)
        if surface is None:
            raise ProtocolConfigurationError(
                f"no surface adapter is bound for kind '{kind.value}'"
            )

        await self._lease.acquire(kind, diff.surface_key, request.round_key)
        try:
            expected = diff.generation
            applied = 0
            for operation in diff.operations:
                current = await surface.read_generation(kind, diff.surface_key)
                if current != expected:
                    return self._refused(
                        EnumDeclaredStateApplyStatus.REFUSED_STALE,
                        applied,
                        f"surface {kind.value}/{diff.surface_key} moved from "
                        f"{expected} to {current} since it was observed; "
                        f"{len(diff.operations) - applied} operations not applied",
                    )
                expected = await surface.apply_operation(operation, request.round_key)
                applied += 1
            return ModelDeclaredStateApplyResult(
                status=EnumDeclaredStateApplyStatus.APPLIED,
                applied_count=applied,
                refusal=None,
            )
        finally:
            await self._lease.release(kind, diff.surface_key, request.round_key)

    @staticmethod
    def _refused(
        status: EnumDeclaredStateApplyStatus, applied: int, reason: str
    ) -> ModelDeclaredStateApplyResult:
        return ModelDeclaredStateApplyResult(
            status=status, applied_count=applied, refusal=reason
        )


__all__ = ["HandlerDeclaredStateApply"]
