# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""LLM-based resolution summary generator.

Calls Qwen3-14B (port 8001) to generate a 2-3 sentence summary of what
an agent did to resolve a problem, based on the session's tool trace.
"""

from __future__ import annotations

_MAX_ERROR_CHARS = 500
_MAX_SUMMARY_CHARS = 2000
_FALLBACK_SUMMARY = "Session completed successfully (no summary generated)."

_SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Given a summary of an agent's "
    "session (repo, files touched, errors encountered, tools used), write a 2-3 "
    "sentence resolution summary explaining what the agent did and why it worked. "
    "Be specific and actionable. Do not include pleasantries or filler."
)


def build_summary_prompt(
    *,
    repo: str,
    file_paths: list[str],
    error_signatures: list[str],
    tool_names: list[str],
) -> str:
    """Build the user prompt for the LLM summary call."""
    truncated_errors = [e[:_MAX_ERROR_CHARS] for e in error_signatures[:5]]
    truncated_files = file_paths[:20]
    # Deduplicate preserving first-occurrence order (sequence matters for trace)
    seen: set[str] = set()
    unique_tools = [t for t in tool_names if t not in seen and not seen.add(t)]  # type: ignore[func-returns-value]
    tool_summary = ", ".join(unique_tools) if unique_tools else "none"

    parts = [
        f"Repo: {repo}",
        f"Files touched: {', '.join(truncated_files) if truncated_files else 'none'}",
        f"Errors encountered: {'; '.join(truncated_errors) if truncated_errors else 'none'}",
        f"Tools used: {tool_summary}",
        "",
        "Write a 2-3 sentence summary of what this agent session accomplished and how it resolved any errors.",
    ]
    return "\n".join(parts)


def parse_summary_response(response: dict[str, object]) -> str:
    """Parse the LLM response into a clean summary string."""
    try:
        choices = response.get("choices", [])
        if not choices or not isinstance(choices, list):
            return _FALLBACK_SUMMARY
        message = choices[0]  # type: ignore[index]
        if isinstance(message, dict):
            content = message.get("message", {})
            if isinstance(content, dict):
                text = str(content.get("content", "")).strip()
            else:
                text = ""
        else:
            text = ""
    except (IndexError, KeyError, TypeError):
        return _FALLBACK_SUMMARY

    if not text:
        return _FALLBACK_SUMMARY

    return text[:_MAX_SUMMARY_CHARS]
