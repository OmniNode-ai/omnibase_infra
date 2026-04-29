# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry coverage test: every consumed secret must appear in shared_key_registry.yaml.

OMN-8778: Asserts that secrets referenced in source code via os.environ.get() or
os.environ[] have a corresponding entry in the shared_key_registry.yaml.

The test greps the source tree for known secret access patterns and cross-references
against all keys declared under the `shared`, `bootstrap_only`, and `identity_defaults`
sections of the registry YAML.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "config" / "shared_key_registry.yaml"

# Source roots to scan (relative to repo root).
_SOURCE_ROOTS = [
    _REPO_ROOT / "src",
    _REPO_ROOT / "scripts",
]

# Regex that captures the key name from os.environ.get("KEY") or os.environ["KEY"].
_ENV_ACCESS_RE = re.compile(r'os\.environ(?:\.get)?\(\s*["\']([A-Z][A-Z0-9_]+)["\']')


def _all_registry_keys(registry: dict) -> set[str]:
    """Collect every key name declared in the registry across all sections."""
    keys: set[str] = set()

    # shared: dict of {path: [KEY, ...]}
    for path_keys in registry.get("shared", {}).values():
        if isinstance(path_keys, list):
            keys.update(k for k in path_keys if isinstance(k, str))

    # bootstrap_only: flat list
    for k in registry.get("bootstrap_only", []):
        if isinstance(k, str):
            keys.add(k)

    # identity_defaults: flat list
    for k in registry.get("identity_defaults", []):
        if isinstance(k, str):
            keys.add(k)

    return keys


def _consumed_keys_in_source() -> set[str]:
    """Scan source roots for os.environ access patterns and return all found keys."""
    found: set[str] = set()
    for root in _SOURCE_ROOTS:
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8", errors="replace")
            found.update(_ENV_ACCESS_RE.findall(text))
    return found


# Keys consumed in source that are intentionally NOT in the registry.
# These are per-service overrides, local-dev-only vars, or test fixtures.
_KNOWN_EXCLUSIONS: frozenset[str] = frozenset(
    {
        # Standard Python / CI env vars — not ONEX secrets.
        "HOME",
        "PATH",
        "USER",
        "SHELL",
        "CI",
        "GITHUB_TOKEN",
        "GITHUB_WORKSPACE",
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_STEP_SUMMARY",
        "GITHUB_RUN_ID",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_REPOSITORY",
        "GITHUB_EVENT_NAME",
        "GITHUB_ACTIONS",
        "RUNNER_TEMP",
        "RUNNER_OS",
        # Per-service overrides (not global shared secrets).
        "POST_MERGE_LINEAR_API_KEY",
        "ONEX_MCP_API_KEY",
        # Test / local dev only.
        "PYTEST_CURRENT_TEST",
        "ONEX_ENVIRONMENT",
        "ONEX_LOG_LEVEL",
        "ONEX_LOG_FORMAT",
        "RUNTIME_VERSION",
        "OMNI_HOME",
        "LINEAR_TEAM_ID",
        "LINEAR_PROJECT_ID",
        "CONSUL_HOST",
        "CONSUL_PORT",
        "CONSUL_TOKEN",
        "CONSUL_DC",
        "CONSUL_SCHEME",
        "ONEX_CONSUL_ENABLED",
        "ONEX_VAULT_ENABLED",
        "ONEX_SKIP_INFISICAL",
        "ONEX_SKIP_CONSUL",
        "ONEX_RUNNER_ID",
        "ONEX_NODE_TYPE",
        "ONEX_INSTANCE_ID",
        "ONEX_ENABLE_OTEL",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "OTEL_RESOURCE_ATTRIBUTES",
        "LOGFIRE_SEND_TO_LOGFIRE",
        "LOGFIRE_PYDANTIC_PLUGIN_RECORD",
        "VAULT_TOKEN",
        "VAULT_NAMESPACE",
        "VAULT_MOUNT_PATH",
        "VAULT_ROLE_ID",
        "VAULT_SECRET_ID",
        "VAULT_TLS_CA_CERT",
        "MEMGRAPH_HOST",
        "MEMGRAPH_PORT",
        "MEMGRAPH_USER",
        "MEMGRAPH_PASSWORD",
        "MEMGRAPH_ENCRYPTED",
        "OMNIMEMORY_HOST",
        "OMNIMEMORY_PORT",
        # Script-level helpers (not seeded to Infisical).
        "DRY_RUN",
        "VERBOSE",
        "NO_COLOR",
        "FORCE",
        "SKIP_CONFIRM",
        "GITHUB_BASE_URL",
        "ONEX_GH_REPO",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "KAFKA_TOPIC",
        "CI_RELAY_HOST",
        "CI_RELAY_PORT",
    }
)

# The 6 keys added by OMN-8778 — always assert these are present regardless of grep.
_OMN_8778_REQUIRED: frozenset[str] = frozenset(
    {
        "LINEAR_API_KEY",
        "ANTHROPIC_API_KEY",
        "CI_CALLBACK_TOKEN",
        "LOGFIRE_TOKEN",
        "LOCAL_LLM_SHARED_SECRET",
        "MCP_API_KEY",
    }
)


class TestSharedKeyRegistryCoverage:
    """Every consumed secret in source must be declared in shared_key_registry.yaml."""

    @pytest.fixture(scope="class")
    def registry(self) -> dict:
        assert _REGISTRY_PATH.exists(), f"Registry not found: {_REGISTRY_PATH}"
        with _REGISTRY_PATH.open() as fh:
            return yaml.safe_load(fh)

    @pytest.fixture(scope="class")
    def registry_keys(self, registry: dict) -> set[str]:
        return _all_registry_keys(registry)

    @pytest.fixture(scope="class")
    def consumed_keys(self) -> set[str]:
        return _consumed_keys_in_source()

    @pytest.mark.unit
    def test_omn_8778_secrets_present(self, registry_keys: set[str]) -> None:
        """The 6 secrets from OMN-8778 must all be in the registry."""
        missing = _OMN_8778_REQUIRED - registry_keys
        assert not missing, (
            f"OMN-8778 secrets missing from shared_key_registry.yaml: {sorted(missing)}"
        )

    @pytest.mark.unit
    def test_consumed_secrets_have_registry_entries(
        self, registry_keys: set[str], consumed_keys: set[str]
    ) -> None:
        """Every secret accessed via os.environ in source must be in the registry
        or explicitly excluded."""
        candidates = consumed_keys - _KNOWN_EXCLUSIONS
        missing = candidates - registry_keys
        assert not missing, (
            "Secrets consumed in source but absent from shared_key_registry.yaml "
            "(add to registry or _KNOWN_EXCLUSIONS if intentionally omitted):\n"
            + "\n".join(f"  - {k}" for k in sorted(missing))
        )

    @pytest.mark.unit
    def test_registry_version_is_string(self, registry: dict) -> None:
        """Registry must declare a version string."""
        assert isinstance(registry.get("version"), str), (
            "shared_key_registry.yaml must have a top-level 'version' string field"
        )

    @pytest.mark.unit
    def test_registry_has_required_top_level_keys(self, registry: dict) -> None:
        """Registry must contain the canonical top-level sections."""
        for section in ("shared", "bootstrap_only", "identity_defaults"):
            assert section in registry, (
                f"shared_key_registry.yaml is missing required section: {section!r}"
            )
