# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Utility: MCP API key injection from environment variables.

Extracted from ``service_runtime_host_process.py`` so the injection logic can
be exercised by unit tests against the **real** production code path (OMN-1419
CR thread fix — "tests don't execute the production injection path").

Public API
----------
``parse_mcp_api_keys(env)`` — parse MCP_API_KEYS / MCP_API_KEY env vars.
``inject_mcp_api_keys(effective_config, env)`` — apply parsed keys to config dict.
"""

from __future__ import annotations

import os


def parse_mcp_api_keys(
    env: dict[str, str] | None = None,
) -> tuple[str, ...] | None:
    """Parse MCP API key env vars and return a normalised key tuple.

    Returns:
        ``None`` if neither env var is set (var was absent).
        ``()`` if the var is set but yields no usable tokens (malformed).
        ``("k1", "k2", ...)`` with one or more non-empty keys otherwise.

    Args:
        env: Optional env mapping to use instead of ``os.environ``. Accepts a
            plain ``dict[str, str]`` so callers can supply a test fixture
            without monkeypatching the real environment.
    """
    lookup: dict[str, str] = env if env is not None else dict(os.environ)

    mcp_api_keys_csv = lookup.get("MCP_API_KEYS") or lookup.get("ONEX_MCP_API_KEYS")
    mcp_api_key_single = lookup.get("MCP_API_KEY") or lookup.get("ONEX_MCP_API_KEY")

    # parsed_keys=None  → env var absent (not set at all)
    # parsed_keys=()    → env var set but empty / whitespace-only
    parsed_keys: tuple[str, ...] | None = None
    if mcp_api_keys_csv is not None:
        parsed_keys = tuple(k.strip() for k in mcp_api_keys_csv.split(",") if k.strip())
    elif mcp_api_key_single is not None:
        parsed_keys = (
            (mcp_api_key_single.strip(),) if mcp_api_key_single.strip() else ()
        )

    return parsed_keys


def inject_mcp_api_keys(
    effective_config: dict[str, object],
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Apply MCP key injection rules to *effective_config* (mutates in place).

    Rules (same semantics as the original inline block in
    ``service_runtime_host_process.py``):

    1. If env-derived ``parsed_keys`` is not None AND ``effective_config``
       does not already have ``api_keys``, inject ``parsed_keys``.
    2. If ``auth_enabled`` is not already set AND config has no ``api_keys``
       AND ``parsed_keys`` is None (env var absent), default to
       ``auth_enabled=False`` for local-dev startup without secrets.
    3. Existing ``api_keys`` in config are never overwritten by env vars.

    Returns the (possibly mutated) config dict.
    """
    parsed_keys = parse_mcp_api_keys(env)

    if parsed_keys is not None and "api_keys" not in effective_config:
        effective_config["api_keys"] = parsed_keys
    elif (
        "auth_enabled" not in effective_config
        and not effective_config.get("api_keys")
        and parsed_keys is None
    ):
        effective_config["auth_enabled"] = False

    return effective_config
