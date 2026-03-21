# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI guard: every :?-required env var in docker-compose.infra.yml must appear
in the test_compose_config_valid fixture dict.

Catches: PRs that add a new service with a required :? env var to compose
without updating the integration test fixture, which previously caused
cascading failures in #886, #890, #895 (OMN-5240 root cause analysis).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

COMPOSE_FILE = (
    Path(__file__).parent.parent.parent / "docker" / "docker-compose.infra.yml"
)
FIXTURE_FILE = (
    Path(__file__).parent.parent
    / "integration"
    / "docker"
    / "test_docker_integration.py"
)


def extract_required_compose_vars(compose_path: Path) -> set[str]:
    """Return all variable names that use :? fail-fast syntax in the compose file."""
    text = compose_path.read_text()
    return set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):\?", text))


def extract_fixture_vars(fixture_path: Path) -> set[str]:
    """Return all string keys in the env.update({...}) dict in test_compose_config_valid."""
    text = fixture_path.read_text()
    # Narrow to the env.update block inside test_compose_config_valid
    # Strategy: find the function, then extract all "KEY": patterns within it
    match = re.search(
        r"def test_compose_config_valid.*?env\.update\s*\(\s*\{(.*?)\}\s*\)",
        text,
        re.DOTALL,
    )
    if not match:
        return set()
    block = match.group(1)
    return set(re.findall(r'"([A-Z_][A-Z0-9_]*)"\s*:', block))


@pytest.mark.unit
def test_all_required_compose_vars_in_fixture() -> None:
    """Every :?-required var in compose must be present in the test fixture dict.

    This is the CI twin for the contract: 'if you add a :? var to compose,
    you must also add it to the test_compose_config_valid fixture dict'.
    Fails on the PR that introduces the gap, not three PRs later.
    """
    required = extract_required_compose_vars(COMPOSE_FILE)
    provided = extract_fixture_vars(FIXTURE_FILE)
    missing = required - provided
    assert not missing, (
        "These :?-required env vars are in docker-compose.infra.yml but "
        "NOT in the test_compose_config_valid fixture dict:\n"
        + "\n".join(f"  - {v}" for v in sorted(missing))
        + "\n\nFix: add each missing var to the env.update({...}) dict in "
        "tests/integration/docker/test_docker_integration.py "
        "(around the 'test_compose_config_valid' method)."
    )
