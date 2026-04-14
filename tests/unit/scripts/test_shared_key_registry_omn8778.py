# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry completeness test for OMN-8778: 6 previously-missing secrets.

Validates that shared_key_registry.yaml declares every key identified in the
OMN-8778 audit so future removals fail loudly rather than silently regressing
Infisical coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "shared_key_registry.yaml"
)

# Keys added by OMN-8778 and their expected Infisical paths.
_REQUIRED_KEYS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "/shared/auth/",
    "LOCAL_LLM_SHARED_SECRET": "/shared/llm/",
    "LINEAR_API_KEY": "/shared/env/",
    "CI_CALLBACK_TOKEN": "/shared/env/",
    "MCP_API_KEY": "/shared/env/",
    "LOGFIRE_TOKEN": "/shared/env/",
}


@pytest.fixture(scope="module")
def registry() -> dict[str, object]:
    return yaml.safe_load(_REGISTRY_PATH.read_text())  # type: ignore[no-any-return]


class TestOMN8778SecretRegistration:
    def test_registry_file_exists(self) -> None:
        assert _REGISTRY_PATH.exists(), (
            f"shared_key_registry.yaml not found at {_REGISTRY_PATH}"
        )

    def test_registry_version_at_least_1_1(self, registry: dict[str, object]) -> None:
        version = str(registry.get("version", "0.0"))
        major, minor = (int(p) for p in version.split("."))
        assert (major, minor) >= (1, 1), (
            f"Registry version {version} predates OMN-8778 additions; expected >= 1.1"
        )

    @pytest.mark.parametrize(("key", "expected_path"), list(_REQUIRED_KEYS.items()))
    def test_key_registered_at_correct_path(
        self, registry: dict[str, object], key: str, expected_path: str
    ) -> None:
        shared: dict[str, list[str]] = registry.get("shared", {})  # type: ignore[assignment]
        keys_at_path: list[str] = shared.get(expected_path, [])
        assert key in keys_at_path, (
            f"Secret '{key}' missing from shared_key_registry.yaml under '{expected_path}'. "
            f"OMN-8778 requires it to be registered there for Infisical seeding coverage."
        )

    def test_no_required_key_in_bootstrap_only(
        self, registry: dict[str, object]
    ) -> None:
        bootstrap: list[str] = registry.get("bootstrap_only", [])  # type: ignore[assignment]
        for key in _REQUIRED_KEYS:
            assert key not in bootstrap, (
                f"Secret '{key}' must not be in bootstrap_only — it is a regular Infisical secret."
            )
