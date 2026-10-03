# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Keep host Bifrost paths out of lane containers (OMN-12864, OMN-19076)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
_COMPOSE_FILES = (
    _DOCKER_DIR / "docker-compose.dogfood.yml",
    _DOCKER_DIR / "docker-compose.judge.yml",
)
_BIFROST_INTERPOLATION_RE = re.compile(
    r"^[ \t]*BIFROST_CONTRACT_PATH:[^\n#]*\$\{[^}\n]+\}", re.MULTILINE
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("compose_file", _COMPOSE_FILES, ids=lambda p: p.name)
def test_no_bifrost_contract_interpolation(compose_file: Path) -> None:
    """Assert the rule on raw compose text without inheriting the host shell."""
    violations = [
        f"line {lineno}: {line.strip()}"
        for lineno, line in enumerate(
            compose_file.read_text(encoding="utf-8").splitlines(), start=1
        )
        if _BIFROST_INTERPOLATION_RE.search(line)
    ]
    assert not violations, (
        f"{compose_file.name}: BIFROST_CONTRACT_PATH must use a container literal "
        f"(OMN-12864, OMN-19076); host interpolation found:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "line",
    [
        "  BIFROST_CONTRACT_PATH: "
        "${BIFROST_CONTRACT_PATH:-/app/data/delegation/bifrost_delegation.yaml}",
        '  BIFROST_CONTRACT_PATH: "${BIFROST_CONTRACT_PATH}"',
        "  BIFROST_CONTRACT_PATH: '/prefix/${OTHER_PATH}'",
    ],
)
def test_bifrost_interpolation_regex_positive_control(line: str) -> None:
    """Prove the guard recognizes interpolated assignments, including quotes."""
    assert _BIFROST_INTERPOLATION_RE.search(line) is not None
