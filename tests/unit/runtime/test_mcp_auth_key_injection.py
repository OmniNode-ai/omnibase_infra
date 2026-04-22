# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for MCP API key injection logic in RuntimeHostProcess.

Tests the key-injection branch at the point where the runtime merges env-var
derived keys into effective_config for MCP handlers (OMN-1419).

CR-thread fixes verified here:
- Critical: existing api_keys in effective_config must NOT be overwritten by
  env-derived keys, and must NOT trigger auth_enabled=False fallback.
- Critical: malformed MCP_API_KEYS (only whitespace/commas) must not silently
  disable auth.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Isolated helper that mirrors the injection logic in
# service_runtime_host_process.py so we can unit-test it without booting a
# full RuntimeHostProcess.  Keep in sync with the production code path.
# ---------------------------------------------------------------------------


def _simulate_mcp_key_injection(
    effective_config: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    """Mirror of the MCP key-injection block in _initialize_handler_instance.

    Takes an initial effective_config dict and a simulated env mapping, applies
    the same injection logic, and returns the (possibly mutated) config.
    """
    mcp_api_keys_csv = env.get("MCP_API_KEYS") or env.get("ONEX_MCP_API_KEYS")
    mcp_api_key_single = env.get("MCP_API_KEY") or env.get("ONEX_MCP_API_KEY")

    parsed_keys: tuple[str, ...] | None = None
    if mcp_api_keys_csv is not None:
        parsed_keys = tuple(k.strip() for k in mcp_api_keys_csv.split(",") if k.strip())
    elif mcp_api_key_single is not None:
        parsed_keys = (
            (mcp_api_key_single.strip(),) if mcp_api_key_single.strip() else ()
        )

    if parsed_keys is not None and "api_keys" not in effective_config:
        effective_config["api_keys"] = parsed_keys
    elif (
        "auth_enabled" not in effective_config
        and not effective_config.get("api_keys")
        and parsed_keys is None
    ):
        effective_config["auth_enabled"] = False

    return effective_config


# ---------------------------------------------------------------------------
# Tests: env-derived keys injected when config has no api_keys
# ---------------------------------------------------------------------------


def test_csv_env_injects_keys() -> None:
    """MCP_API_KEYS CSV injects parsed tuple into effective_config."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {"MCP_API_KEYS": "alpha,beta,gamma"})
    assert result["api_keys"] == ("alpha", "beta", "gamma")
    assert "auth_enabled" not in result


def test_single_env_injects_single_key() -> None:
    """MCP_API_KEY (single) injects a one-element tuple."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {"MCP_API_KEY": "solo-token"})
    assert result["api_keys"] == ("solo-token",)


def test_no_env_no_config_disables_auth() -> None:
    """With no env vars and no existing config, auth defaults to disabled (local dev)."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {})
    assert result.get("auth_enabled") is False
    assert "api_keys" not in result


# ---------------------------------------------------------------------------
# Tests: existing api_keys in effective_config are preserved (CR thread fix)
# ---------------------------------------------------------------------------


def test_existing_contract_api_keys_not_overwritten_by_env() -> None:
    """If effective_config already has api_keys, env vars must not override them.

    Covers the critical CR-thread finding: a config pre-populated with api_keys
    (from contract/runtime config) must not be silently replaced by env-derived keys.
    """
    cfg: dict[str, Any] = {"api_keys": ("contract-key-a", "contract-key-b")}
    result = _simulate_mcp_key_injection(cfg, {"MCP_API_KEYS": "injected-key"})
    # Contract keys must remain intact
    assert result["api_keys"] == ("contract-key-a", "contract-key-b")
    # auth_enabled must not have been set to False either
    assert "auth_enabled" not in result


def test_existing_api_keys_prevent_auth_disabled_fallback() -> None:
    """Existing api_keys in config must block the auth_enabled=False fallback.

    The fail-open path (auth_enabled=False) must only trigger when both
    env vars are absent AND effective_config has no api_keys.
    """
    cfg: dict[str, Any] = {"api_keys": ("existing-key",)}
    result = _simulate_mcp_key_injection(cfg, {})  # no env vars
    # auth_enabled must NOT be set to False since api_keys already present
    assert "auth_enabled" not in result
    assert result["api_keys"] == ("existing-key",)


# ---------------------------------------------------------------------------
# Tests: malformed MCP_API_KEYS values (CR thread fix — fail-safe, not fail-open)
# ---------------------------------------------------------------------------


def test_whitespace_only_csv_yields_empty_tuple() -> None:
    """MCP_API_KEYS with only whitespace/commas yields an empty parsed_keys tuple.

    The var is set (not None), so parsed_keys != None, but contains no usable tokens.
    This must NOT trigger the auth_enabled=False fallback.
    """
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {"MCP_API_KEYS": " , , "})
    # api_keys set to empty tuple — the Pydantic validator on ModelMcpHandlerConfig
    # will reject this at initialization time (validator enforces non-empty when
    # auth_enabled=True), which is the correct fail-fast behavior.
    assert result.get("api_keys") == ()
    # auth_enabled must NOT be set to False (env var was present, just malformed)
    assert "auth_enabled" not in result


def test_whitespace_only_single_key_yields_empty_tuple() -> None:
    """MCP_API_KEY set to whitespace-only yields empty parsed_keys, not auth bypass."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {"MCP_API_KEY": "   "})
    assert result.get("api_keys") == ()
    assert "auth_enabled" not in result


# ---------------------------------------------------------------------------
# Tests: onex alias env vars
# ---------------------------------------------------------------------------


def test_onex_prefix_csv_alias_works() -> None:
    """ONEX_MCP_API_KEYS (alias) is accepted when MCP_API_KEYS is absent."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {"ONEX_MCP_API_KEYS": "x,y"})
    assert result["api_keys"] == ("x", "y")


def test_onex_prefix_single_alias_works() -> None:
    """ONEX_MCP_API_KEY (alias) is accepted when MCP_API_KEY is absent."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(cfg, {"ONEX_MCP_API_KEY": "z-token"})
    assert result["api_keys"] == ("z-token",)


def test_csv_takes_precedence_over_single() -> None:
    """When both MCP_API_KEYS and MCP_API_KEY are set, CSV wins (it's checked first)."""
    cfg: dict[str, Any] = {}
    result = _simulate_mcp_key_injection(
        cfg, {"MCP_API_KEYS": "key-a,key-b", "MCP_API_KEY": "single"}
    )
    assert result["api_keys"] == ("key-a", "key-b")
