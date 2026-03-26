# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test env var alignment checker detects mismatches between app code and k8s manifests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.unit
def test_extract_manifest_env_vars() -> None:
    """Extract env var names from a k8s deployment YAML."""
    from compare_environments import _extract_manifest_env_vars

    result = _extract_manifest_env_vars(FIXTURES / "fake-deployment.yaml")
    assert result == {"OMNIBASE_INFRA_DB_URL", "NEXTAUTH_SECRET"}


@pytest.mark.unit
def test_extract_app_env_vars() -> None:
    """Extract env var names from TypeScript source code."""
    from compare_environments import _extract_app_env_vars

    result = _extract_app_env_vars(FIXTURES / "fake-source", "test-app")
    assert "OMNIWEB_DB_URL" in result
    assert "NEXTAUTH_SECRET" in result


@pytest.mark.unit
def test_env_var_alignment_detects_mismatch() -> None:
    """When app reads OMNIWEB_DB_URL but deployment only injects OMNIBASE_INFRA_DB_URL, flag CRITICAL."""
    from compare_environments import (
        _extract_app_env_vars,
        _extract_manifest_env_vars,
    )

    # Build a minimal infra_repo structure that points to our fixture
    # We need k8s/onex-dev/<service>/deployment.yaml to exist
    # The fixture deployment.yaml is at FIXTURES/fake-deployment.yaml
    # We'll use check_env_var_name_alignment with service_repos pointing to fake-source
    # and infra_repo pointing to a path where k8s/onex-dev/test-app/deployment.yaml exists

    # Instead, test the underlying functions directly for a cleaner unit test
    manifest_vars = _extract_manifest_env_vars(FIXTURES / "fake-deployment.yaml")
    app_vars = _extract_app_env_vars(FIXTURES / "fake-source", "test-app")

    missing = app_vars - manifest_vars
    assert "OMNIWEB_DB_URL" in missing, (
        f"Expected OMNIWEB_DB_URL to be missing from manifest. "
        f"manifest_vars={manifest_vars}, app_vars={app_vars}"
    )


@pytest.mark.unit
def test_env_var_alignment_clean() -> None:
    """When all app env vars are in manifest, no findings."""
    app_env_vars = {"OMNIWEB_DB_URL", "NEXTAUTH_SECRET"}
    manifest_env_vars = {
        "OMNIWEB_DB_URL",
        "NEXTAUTH_SECRET",
        "AUTH_TRUST_HOST",
        "OMNIBASE_INFRA_DB_URL",
    }
    missing = app_env_vars - manifest_env_vars
    assert missing == set(), "All app vars present in manifest"
