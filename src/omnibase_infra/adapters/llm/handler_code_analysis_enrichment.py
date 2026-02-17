# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Code analysis enrichment handler for ProtocolContextEnrichment.

Reads git diff of modified files, calls the Coder-14B LLM to analyze
changes, and returns a ContractEnrichmentResult with structured markdown
summarizing affected functions, dependency changes, and potential issues.

Architecture:
    - Implements ProtocolContextEnrichment from omnibase_spi
    - Reads git diff via asyncio.create_subprocess_exec (no shell)
    - Delegates LLM inference to HandlerLlmOpenaiCompatible via
      TransportHolderLlmHttp
    - Returns ContractEnrichmentResult with enrichment_type="code_analysis"

Git Diff Strategy:
    The ``context`` parameter is treated as the raw git diff text to analyze.
    When empty, the handler falls back to running ``git diff HEAD`` in the
    current working directory.  The ``prompt`` parameter provides the user
    query context so the LLM can focus its analysis on relevant aspects.

Token Estimation:
    Token count is estimated at 4 characters per token (rough heuristic).
    Actual counts depend on the model tokenizer but this is sufficient
    for budget accounting purposes.

Related Tickets:
    - OMN-2260: Code analysis enrichment handler
    - OMN-2252: ProtocolContextEnrichment SPI contract
    - OMN-2257: LLM endpoint configuration
    - OMN-2107: HandlerLlmOpenaiCompatible
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    TransportHolderLlmHttp,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLlmOperationType,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_request import (
    ModelLlmInferenceRequest,
)
from omnibase_spi.contracts.enrichment.contract_enrichment_result import (
    ContractEnrichmentResult,
)

logger = logging.getLogger(__name__)

# Prompt version -- bump this when the system/user prompt template changes.
_PROMPT_VERSION: str = "v1.0"

# Default model identifier sent to the Coder-14B endpoint.
_DEFAULT_MODEL: str = "qwen2.5-coder-14b"

# Default maximum tokens for the analysis response.
_DEFAULT_MAX_TOKENS: int = 1024

# Default temperature -- low value for deterministic code analysis.
_DEFAULT_TEMPERATURE: float = 0.1

# Rough token estimation: 4 characters per token.
_CHARS_PER_TOKEN: int = 4

# Maximum characters of diff sent to the LLM to avoid context overflow.
# Coder-14B supports up to 32 K tokens; at ~4 chars/token that is ~128 K
# characters. We cap conservatively at 32 000 chars (≈8 000 tokens) to
# leave room for the prompt, system message, and completion.
_MAX_DIFF_CHARS: int = 32_000

# Timeout for the git subprocess (seconds).
_GIT_TIMEOUT_SECONDS: float = 15.0

# Relevance score assigned when the diff is non-empty and analysis succeeds.
_ANALYSIS_RELEVANCE_SCORE: float = 0.85

# Relevance score when no diff is found (nothing changed / empty context).
_EMPTY_DIFF_RELEVANCE_SCORE: float = 0.0

# System prompt sent to the model.
_SYSTEM_PROMPT: str = (
    "You are an expert software engineer performing code change analysis. "
    "Your task is to analyze git diffs and produce a concise, structured "
    "summary in Markdown. Focus on: affected functions/methods, API or "
    "interface changes, potential side effects or regressions, and "
    "dependency changes. Be precise and technical. Avoid filler text."
)

# User prompt template -- {prompt} is the caller's query, {diff} is the diff.
_USER_PROMPT_TEMPLATE: str = """\
Analyze the following git diff in the context of this query: {prompt}

## Git Diff

```diff
{diff}
```

Produce a Markdown report with these sections:
1. **Affected Functions / Methods** - list each changed function/method with a one-line description of what changed.
2. **Dependency Changes** - note any import additions, removals, or modifications.
3. **Potential Issues** - highlight risks, regressions, or breaking changes.
4. **Summary** - one paragraph summary of the overall change.
"""


class HandlerCodeAnalysisEnrichment:
    """Context enrichment handler that analyzes git diffs via Coder-14B.

    Implements ``ProtocolContextEnrichment``.  The ``context`` parameter
    is treated as raw git diff text.  When empty, the handler runs
    ``git diff HEAD`` in the current working directory as a fallback.

    Attributes:
        handler_type: ``INFRA_HANDLER`` -- infrastructure-level handler.
        handler_category: ``EFFECT`` -- performs external I/O (git + LLM HTTP).

    Example:
        >>> handler = HandlerCodeAnalysisEnrichment()
        >>> result = await handler.enrich(
        ...     prompt="What functions were affected?",
        ...     context="diff --git a/foo.py ...",
        ... )
        >>> print(result.summary_markdown)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            base_url: Base URL of the Coder-14B endpoint.  Defaults to the
                ``LLM_CODER_URL`` environment variable, falling back to
                ``http://localhost:8000``.
            model: Model identifier string sent in inference requests.
            max_tokens: Maximum tokens for the LLM completion.
            temperature: Sampling temperature (lower = more deterministic).
            api_key: Optional Bearer token for authenticated endpoints.
        """
        self._base_url: str = base_url or os.environ.get(
            "LLM_CODER_URL", "http://localhost:8000"
        )
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._api_key: str | None = api_key

        self._transport = TransportHolderLlmHttp(
            target_name="coder-14b-enrichment",
            max_timeout_seconds=120.0,
        )
        self._handler = HandlerLlmOpenaiCompatible(self._transport)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: EFFECT (git subprocess + HTTP call)."""
        return EnumHandlerTypeCategory.EFFECT

    async def enrich(
        self,
        prompt: str,
        context: str,
    ) -> ContractEnrichmentResult:
        """Enrich a prompt by analyzing git diff via Coder-14B.

        When ``context`` is non-empty, it is used as-is as the diff text.
        When ``context`` is empty, ``git diff HEAD`` is run in the current
        working directory to obtain the diff.

        Args:
            prompt: User query or description of the change context.
                Used to focus the LLM analysis.
            context: Raw git diff text.  When empty, falls back to
                running ``git diff HEAD`` automatically.

        Returns:
            ``ContractEnrichmentResult`` with:

            - ``enrichment_type="code_analysis"``
            - ``summary_markdown``: Structured Markdown analysis
            - ``token_count``: Estimated token count of the summary
            - ``relevance_score``: 0.85 for non-empty diff, 0.0 otherwise
            - ``model_used``: Model identifier from the handler
            - ``prompt_version``: Template version (``"v1.0"``)
            - ``latency_ms``: End-to-end wall time in milliseconds

        Raises:
            RuntimeHostError: Propagated from ``HandlerLlmOpenaiCompatible``
                on connection failures, timeouts, or authentication errors.
        """
        start = time.perf_counter()

        # Resolve diff text: use provided context or fall back to git diff HEAD.
        diff_text = context.strip()
        if not diff_text:
            diff_text = await self._read_git_diff()

        if not diff_text:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "No git diff found; returning empty enrichment result. latency_ms=%.1f",
                latency_ms,
            )
            return ContractEnrichmentResult(
                summary_markdown="## No Changes Detected\n\nNo git diff found to analyze.",
                token_count=0,
                relevance_score=_EMPTY_DIFF_RELEVANCE_SCORE,
                enrichment_type="code_analysis",
                latency_ms=latency_ms,
                model_used=self._model,
                prompt_version=_PROMPT_VERSION,
            )

        # Truncate diff to avoid context overflow.
        if len(diff_text) > _MAX_DIFF_CHARS:
            logger.debug(
                "Truncating diff from %d to %d chars to fit context window.",
                len(diff_text),
                _MAX_DIFF_CHARS,
            )
            diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n... [diff truncated]"

        user_message = _USER_PROMPT_TEMPLATE.format(
            prompt=prompt,
            diff=diff_text,
        )

        request = ModelLlmInferenceRequest(
            base_url=self._base_url,
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            model=self._model,
            messages=({"role": "user", "content": user_message},),
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            api_key=self._api_key,
        )

        response = await self._handler.handle(request)
        latency_ms = (time.perf_counter() - start) * 1000

        summary = response.generated_text or ""
        if not summary:
            summary = "## Analysis Unavailable\n\nThe model did not return a response."
            logger.warning(
                "Coder-14B returned empty generated_text. model=%s latency_ms=%.1f",
                self._model,
                latency_ms,
            )

        token_count = max(0, len(summary) // _CHARS_PER_TOKEN)

        logger.debug(
            "Code analysis enrichment complete. "
            "model=%s token_count=%d latency_ms=%.1f",
            self._model,
            token_count,
            latency_ms,
        )

        return ContractEnrichmentResult(
            summary_markdown=summary,
            token_count=token_count,
            relevance_score=_ANALYSIS_RELEVANCE_SCORE,
            enrichment_type="code_analysis",
            latency_ms=latency_ms,
            model_used=self._model,
            prompt_version=_PROMPT_VERSION,
        )

    async def close(self) -> None:
        """Close the HTTP transport client."""
        await self._transport.close()

    @staticmethod
    async def _read_git_diff(repo_path: str = ".") -> str:
        """Run ``git diff HEAD`` and return the output as a string.

        Uses ``asyncio.create_subprocess_exec`` (not shell) for safety.
        Arguments are passed as a list, preventing shell injection.

        Args:
            repo_path: Directory in which to run the git command.
                Defaults to the current working directory.

        Returns:
            Raw diff text, or an empty string on any error.
        """
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                repo_path,
                "diff",
                "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_GIT_TIMEOUT_SECONDS
            )
            if proc.returncode != 0:
                logger.debug(
                    "git diff HEAD failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace").strip()[:200],
                )
                return ""
            return stdout.decode(errors="replace")
        except TimeoutError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            logger.debug("git diff HEAD timed out after %.0fs", _GIT_TIMEOUT_SECONDS)
            return ""
        except asyncio.CancelledError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            raise
        except OSError as exc:
            logger.debug("git diff HEAD failed (OSError): %s", exc)
            return ""


__all__: list[str] = ["HandlerCodeAnalysisEnrichment"]
