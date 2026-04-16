# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI subprocess handler for LLM inference via Gemini CLI and Codex CLI.

Generalizes the proven subprocess dispatch pattern from the hostile reviewer
aggregator (aggregate_reviews.py) into an ONEX handler that accepts
ModelLlmInferenceRequest and returns ModelLlmInferenceResponse.

CLI subprocess handlers distinguish these failure classes:
- UNAVAILABLE: CLI binary not found on PATH
- INVALID_REQUEST: empty prompt or malformed input
- TIMEOUT: subprocess exceeded deadline
- SUBPROCESS_ERROR: non-zero exit code with stderr
- EMPTY_RESPONSE: process succeeded but stdout was empty
- SUCCESS: valid response returned

Related:
    - OMN-7106: Add Gemini CLI and Codex CLI as subprocess LLM handlers
    - OMN-7103: Node-Based LLM Delegation Workflow
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from datetime import UTC
from enum import Enum
from typing import TYPE_CHECKING

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)

if TYPE_CHECKING:
    from omnibase_infra.models.llm.model_llm_inference_request import (
        ModelLlmInferenceRequest,
    )
    from omnibase_infra.models.llm.model_llm_inference_response import (
        ModelLlmInferenceResponse,
    )

logger = logging.getLogger(__name__)


class EnumCliBackendStatus(str, Enum):
    """Structured failure classes for CLI subprocess handlers."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    SUBPROCESS_ERROR = "subprocess_error"
    EMPTY_RESPONSE = "empty_response"


class HandlerLlmCliSubprocess:
    """Dispatch LLM inference to a CLI tool (gemini, codex) via subprocess.

    This handler spawns the CLI in headless/non-interactive mode (-p flag),
    passes the user prompt, captures stdout as the response, and wraps it
    in a ModelLlmInferenceResponse.

    The handler returns a tuple of (response, status, detail) to preserve
    structured failure information for fallback routing and metrics.

    Example:
        >>> handler = HandlerLlmCliSubprocess(cli="gemini", cli_args=["-p"])
        >>> response, status, detail = handler.execute_cli_inference(request)
        >>> if status == EnumCliBackendStatus.SUCCESS:
        ...     print(response.generated_text)
    """

    def __init__(
        self,
        cli: str = "claude",
        cli_args: list[str] | None = None,
        timeout: int = 120,
    ) -> None:
        self._cli = cli
        self._cli_args = cli_args or ["-p"]
        self._timeout = timeout

    @property
    def cli_name(self) -> str:
        """Return the CLI binary name."""
        return self._cli

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification."""
        return EnumHandlerTypeCategory.EFFECT

    def execute_cli_inference(
        self,
        request: ModelLlmInferenceRequest,
    ) -> tuple[ModelLlmInferenceResponse | None, EnumCliBackendStatus, str]:
        """Execute inference via CLI subprocess.

        Returns:
            Tuple of (response, status, detail). Status is always set even
            on failure, preserving structured failure information.
        """
        from omnibase_infra.models.llm.model_llm_inference_response import (
            ModelLlmInferenceResponse,
        )

        if not shutil.which(self._cli):
            return (
                None,
                EnumCliBackendStatus.UNAVAILABLE,
                f"{self._cli} not found on PATH",
            )

        # Extract last user message as prompt
        prompt = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                prompt = msg.content or ""
                break

        if not prompt:
            return (
                None,
                EnumCliBackendStatus.INVALID_REQUEST,
                "no user message in request",
            )

        try:
            start = time.monotonic()
            result = subprocess.run(
                [self._cli, *self._cli_args, prompt],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            latency_ms = (time.monotonic() - start) * 1000

            if result.returncode != 0:
                stderr_preview = result.stderr[:200] if result.stderr else "(no stderr)"
                logger.debug(
                    "%s exited %d: %s", self._cli, result.returncode, stderr_preview
                )
                return (
                    None,
                    EnumCliBackendStatus.SUBPROCESS_ERROR,
                    f"exit {result.returncode}: {stderr_preview}",
                )

            content = result.stdout.strip()
            if not content:
                return (
                    None,
                    EnumCliBackendStatus.EMPTY_RESPONSE,
                    "stdout empty after successful exit",
                )

            # Build response with all required fields
            from datetime import datetime
            from uuid import uuid4

            from omnibase_infra.enums import (
                EnumLlmFinishReason,
                EnumLlmOperationType,
            )
            from omnibase_infra.models.llm.model_llm_usage import ModelLlmUsage
            from omnibase_infra.models.model_backend_result import ModelBackendResult

            # Rough token estimate: ~1.3 tokens per word
            prompt_tokens = int(len(prompt.split()) * 1.3)
            completion_tokens = int(len(content.split()) * 1.3)

            response = ModelLlmInferenceResponse(
                status="success",
                generated_text=content,
                model_used=f"{self._cli}-cli",
                operation_type=EnumLlmOperationType.CHAT_COMPLETION,
                finish_reason=EnumLlmFinishReason.STOP,
                usage=ModelLlmUsage(
                    tokens_input=prompt_tokens,
                    tokens_output=completion_tokens,
                    tokens_total=prompt_tokens + completion_tokens,
                ),
                latency_ms=latency_ms,
                backend_result=ModelBackendResult(success=True, duration_ms=latency_ms),
                correlation_id=getattr(request, "correlation_id", uuid4()),
                execution_id=uuid4(),
                timestamp=datetime.now(tz=UTC),
            )

            logger.info(
                "%s-cli: completed in %.0fms (~%d tokens)",
                self._cli,
                latency_ms,
                completion_tokens,
            )

            return (response, EnumCliBackendStatus.SUCCESS, "")

        except subprocess.TimeoutExpired:
            return (
                None,
                EnumCliBackendStatus.TIMEOUT,
                f"{self._cli} exceeded {self._timeout}s deadline",
            )


__all__ = [
    "EnumCliBackendStatus",
    "HandlerLlmCliSubprocess",
]
