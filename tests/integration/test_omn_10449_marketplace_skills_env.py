# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime marketplace skill-manifest env wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.service_kernel import ENV_MARKETPLACE_SKILLS_ROOT

pytestmark = pytest.mark.integration


def test_runtime_compose_env_matches_kernel_marketplace_skills_env() -> None:
    """Compose and kernel agree on the marketplace skill-manifest env name."""
    repo_root = Path(__file__).resolve().parents[2]
    compose_path = repo_root / "docker" / "docker-compose.infra.yml"
    compose = yaml.safe_load(compose_path.read_text())

    runtime_env = compose["services"]["omninode-runtime"]["environment"]

    assert ENV_MARKETPLACE_SKILLS_ROOT == "ONEX_MARKETPLACE_SKILLS_ROOT"
    assert ENV_MARKETPLACE_SKILLS_ROOT in runtime_env
    assert "OMNICLAUDE_SKILLS_ROOT" not in runtime_env
