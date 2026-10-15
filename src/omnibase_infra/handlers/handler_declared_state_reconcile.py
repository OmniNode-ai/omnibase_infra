# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One reconcile round: observe, diff, then apply when the mode and policy allow."""

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.handler_declared_state_apply import (
    HandlerDeclaredStateApply,
)
from omnibase_infra.handlers.handler_declared_state_diff import (
    HandlerDeclaredStateDiff,
)
from omnibase_infra.models.declared_state.enum_declared_state_reconcile_mode import (
    EnumDeclaredStateReconcileMode,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_request import (
    ModelDeclaredStateApplyRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_diff_request import (
    ModelDeclaredStateDiffRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_reconcile_request import (
    ModelDeclaredStateReconcileRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_reconcile_result import (
    ModelDeclaredStateReconcileResult,
)
from omnibase_infra.protocols.protocol_declared_state_observer import (
    ProtocolDeclaredStateObserver,
)


class HandlerDeclaredStateReconcile:
    """Every round observes first and plans from what it observed.

    Nothing is remembered between rounds, so a hand edit made after a
    successful apply is repaired by the next round.
    """

    def __init__(
        self,
        observer: ProtocolDeclaredStateObserver,
        diff: HandlerDeclaredStateDiff,
        apply: HandlerDeclaredStateApply,
    ) -> None:
        self._observer = observer
        self._diff = diff
        self._apply = apply

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the orchestrating role."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the side-effecting category."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelDeclaredStateReconcileRequest
    ) -> ModelDeclaredStateReconcileResult:
        """Observe, plan, and apply only in apply mode with something to settle."""
        snapshot = await self._observer.observe(
            request.surface_kind,
            request.surface_key,
            request.round_key,
            request.declared,
        )
        diff = self._diff.handle(
            ModelDeclaredStateDiffRequest(
                surface_kind=request.surface_kind,
                surface_key=request.surface_key,
                generation=snapshot.generation,
                declared=request.declared,
                observations=snapshot.observations,
            )
        )
        apply = None
        settles = bool(diff.operations) or diff.unobservable_reason is not None
        if request.mode is EnumDeclaredStateReconcileMode.APPLY and settles:
            apply = await self._apply.handle(
                ModelDeclaredStateApplyRequest(
                    diff=diff, policy=request.policy, round_key=request.round_key
                )
            )
        return ModelDeclaredStateReconcileResult(
            round_key=request.round_key,
            mode=request.mode,
            operation_count=len(diff.operations),
            unobservable_reason=diff.unobservable_reason,
            apply=apply,
        )


__all__ = ["HandlerDeclaredStateReconcile"]
