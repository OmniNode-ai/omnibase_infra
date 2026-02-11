# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for auth gate — 10-step authorization decision cascade.

Pure COMPUTE handler: receives auth state + tool request, returns
allow/deny/soft_deny. No I/O, no side effects.

Decision Cascade (evaluated top-to-bottom, first match wins):
     1. Whitelisted paths -> allow (plans, memory)
     2. Emergency override active -> allow (with reason_code) / deny if no reason
     3. No run_id determinable -> deny
     4. Run context not found (no authorization) -> deny
     5. Auth not granted (authorization exists but run_id mismatch) -> deny
     6. Tool not in allowed_tools -> deny
     7. Path not matching allowed_paths glob -> deny
     8. Repo not in repo_scopes -> deny
     9. Auth expired -> deny
    10. All checks pass -> allow

Ticket: OMN-2125
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_auth_decision import EnumAuthDecision
from omnibase_infra.nodes.node_auth_gate_compute.models.model_auth_gate_decision import (
    ModelAuthGateDecision,
)
from omnibase_infra.nodes.node_auth_gate_compute.models.model_auth_gate_request import (
    ModelAuthGateRequest,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

logger = logging.getLogger(__name__)

HANDLER_ID_AUTH_GATE: str = "auth-gate-handler"

# Paths that are always permitted regardless of authorization state.
# Plans and memory files are safe to read/write without explicit auth.
WHITELISTED_PATH_PATTERNS: tuple[str, ...] = (
    "*.plan.md",
    "*/.claude/memory/*",
    "*/.claude/projects/*/memory/*",
    "*/MEMORY.md",
)

EMERGENCY_BANNER: str = (
    "EMERGENCY OVERRIDE ACTIVE — "
    "All tool invocations are permitted under emergency override. "
    "This override expires in 10 minutes and cannot be renewed "
    "without manual /authorize."
)


class HandlerAuthGate:
    """Pure COMPUTE handler for authorization decisions.

    Implements a 10-step cascade that evaluates authorization state against
    a tool invocation request. Each step either returns a decision (early exit)
    or falls through to the next step. The final step (10) is the success case.

    CRITICAL INVARIANTS:
    - Pure computation: no I/O, no side effects, no event bus access
    - Deterministic: ``evaluate()`` always produces same output for same input.
      ``execute()`` generates envelope metadata (correlation_id, envelope_id)
      which may differ across calls.
    - Cascade order is fixed and must not be reordered

    Attributes:
        handler_type: EnumHandlerType.COMPUTE_HANDLER
        handler_category: EnumHandlerTypeCategory.COMPUTE
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the auth gate handler.

        Args:
            container: ONEX dependency injection container.
        """
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.COMPUTE_HANDLER
        """
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.COMPUTE
        """
        return EnumHandlerTypeCategory.COMPUTE

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler.

        Args:
            config: Configuration dict (currently unused).
        """
        self._initialized = True
        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={"handler": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False
        logger.info("HandlerAuthGate shutdown complete")

    def evaluate(self, request: ModelAuthGateRequest) -> ModelAuthGateDecision:
        """Evaluate the 10-step authorization cascade.

        This is the core pure function. Each step returns a decision or
        falls through.

        Args:
            request: Authorization gate request with tool context and auth state.

        Returns:
            Authorization decision with step, reason, and optional banner.
        """
        # Step 1: Whitelisted paths -> allow
        if self._is_whitelisted_path(request.target_path):
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.ALLOW,
                step=1,
                reason=f"Path '{request.target_path}' matches whitelisted pattern.",
                reason_code="whitelisted_path",
            )

        # Step 2: Emergency override
        if request.emergency_override_active:
            if not request.emergency_override_reason:
                return ModelAuthGateDecision(
                    decision=EnumAuthDecision.DENY,
                    step=2,
                    reason=(
                        "Emergency override active but ONEX_UNSAFE_REASON is empty. "
                        "A reason is required for emergency overrides."
                    ),
                    reason_code="emergency_no_reason",
                )
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.SOFT_DENY,
                step=2,
                reason=(
                    f"Emergency override active. "
                    f"Reason: {request.emergency_override_reason}"
                ),
                reason_code="emergency_override",
                banner=EMERGENCY_BANNER,
            )

        # Step 3: No run_id determinable -> deny
        if request.run_id is None:
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=3,
                reason="No run_id determinable from context.",
                reason_code="no_run_id",
            )

        # Step 4: Run context not found (no authorization) -> deny
        if request.authorization is None:
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=4,
                reason="No authorization contract found for this run.",
                reason_code="no_authorization",
            )

        auth = request.authorization

        # Step 5: Auth not granted (run_id mismatch) -> deny
        if auth.run_id != request.run_id:
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=5,
                reason=(
                    f"Authorization run_id mismatch: "
                    f"auth={auth.run_id}, request={request.run_id}."
                ),
                reason_code="run_id_mismatch",
            )

        # Step 6: Tool not in allowed_tools -> deny
        if request.tool_name not in auth.allowed_tools:
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=6,
                reason=(
                    f"Tool '{request.tool_name}' not in allowed_tools: "
                    f"{list(auth.allowed_tools)}."
                ),
                reason_code="tool_not_allowed",
            )

        # Step 7: Path not matching allowed_paths glob -> deny
        if request.target_path and not self._path_matches_globs(
            request.target_path, auth.allowed_paths
        ):
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=7,
                reason=(
                    f"Path '{request.target_path}' does not match any allowed_paths: "
                    f"{list(auth.allowed_paths)}."
                ),
                reason_code="path_not_allowed",
            )

        # Step 8: Repo not in repo_scopes -> deny
        if (
            request.target_repo
            and auth.repo_scopes
            and (request.target_repo not in auth.repo_scopes)
        ):
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=8,
                reason=(
                    f"Repository '{request.target_repo}' not in repo_scopes: "
                    f"{list(auth.repo_scopes)}."
                ),
                reason_code="repo_not_in_scope",
            )

        # Step 9: Auth expired -> deny
        if auth.is_expired(now=request.now):
            return ModelAuthGateDecision(
                decision=EnumAuthDecision.DENY,
                step=9,
                reason=(f"Authorization expired at {auth.expires_at.isoformat()}."),
                reason_code="auth_expired",
            )

        # Step 10: All checks pass -> allow
        return ModelAuthGateDecision(
            decision=EnumAuthDecision.ALLOW,
            step=10,
            reason="All authorization checks passed.",
            reason_code="all_checks_passed",
        )

    async def execute(
        self,
        envelope: dict[str, object],
    ) -> ModelHandlerOutput[ModelAuthGateDecision]:
        """Execute auth gate from envelope (ProtocolHandler interface).

        Args:
            envelope: Request envelope containing:
                - operation: "auth_gate.evaluate"
                - payload: ModelAuthGateRequest as dict
                - correlation_id: Optional correlation ID

        Returns:
            ModelHandlerOutput wrapping ModelAuthGateDecision.
        """
        correlation_id_raw = envelope.get("correlation_id")
        correlation_id = (
            UUID(str(correlation_id_raw)) if correlation_id_raw else uuid4()
        )
        input_envelope_id = uuid4()

        payload_raw = envelope.get("payload")
        if payload_raw is None:
            msg = "Envelope missing required 'payload' key for auth gate evaluation."
            raise ValueError(msg)
        if isinstance(payload_raw, ModelAuthGateRequest):
            request = payload_raw
        else:
            request = ModelAuthGateRequest.model_validate(payload_raw)

        decision = self.evaluate(request)

        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_AUTH_GATE,
            result=decision,
        )

    @staticmethod
    def _is_whitelisted_path(path: str) -> bool:
        """Check if a path matches any whitelisted pattern.

        Uses ``fnmatch`` where ``*`` matches across directory separators.
        This is intentionally more permissive than ``_path_matches_globs``
        (which treats ``*`` as non-``/`` matching) because whitelisted
        paths are safe-by-definition and broader matching is desired.

        Args:
            path: File path to check.

        Returns:
            True if the path matches a whitelisted pattern.
        """
        if not path:
            return False
        for pattern in WHITELISTED_PATH_PATTERNS:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    @staticmethod
    def _glob_to_regex(pattern: str) -> re.Pattern[str]:
        """Convert a glob pattern (supporting ``**``) to a compiled regex.

        Standard ``fnmatch`` does not treat ``**`` as recursive directory
        match. This function converts glob patterns to regex where:
        - ``**`` matches zero or more path components (including ``/``)
        - ``*`` matches any characters except ``/``
        - ``?`` matches a single non-``/`` character

        Args:
            pattern: Glob pattern, e.g. ``src/**/*.py``.

        Returns:
            Compiled regex pattern.
        """
        regex = ""
        i = 0
        n = len(pattern)
        while i < n:
            c = pattern[i]
            if c == "*":
                if i + 1 < n and pattern[i + 1] == "*":
                    # ** matches zero or more path components
                    if i + 2 < n and pattern[i + 2] == "/":
                        regex += "(?:.*/)?"
                        i += 3
                    else:
                        regex += ".*"
                        i += 2
                    continue
                regex += "[^/]*"
            elif c == "?":
                regex += "[^/]"
            else:
                regex += re.escape(c)
            i += 1
        return re.compile(f"^{regex}$")

    @staticmethod
    def _path_matches_globs(path: str, globs: tuple[str, ...]) -> bool:
        """Check if a path matches any of the provided glob patterns.

        All patterns are routed through ``_glob_to_regex`` to ensure ``*``
        never matches ``/``. This is stricter than ``fnmatch`` (used only
        for whitelisted paths) and is the correct behavior for authorization.

        Args:
            path: File path to check.
            globs: Glob patterns to match against.

        Returns:
            True if the path matches at least one glob pattern.
        """
        for pattern in globs:
            if HandlerAuthGate._glob_to_regex(pattern).match(path):
                return True
        return False


__all__: list[str] = ["HandlerAuthGate"]
