# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime image dependency pin tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def test_omnibase_compat_pin_includes_delegation_wire_contracts(
    dockerfile_path: Path,
) -> None:
    """Guard the runtime image against an omnimarket delegation import crash."""
    dockerfile = dockerfile_path.read_text(encoding="utf-8")

    match = re.search(
        r'ARG OMNIBASE_COMPAT_SOURCE="https://github\.com/OmniNode-ai/'
        r'omnibase_compat/archive/(?P<sha>[0-9a-f]{40})\.tar\.gz"',
        dockerfile,
    )

    assert match is not None, (
        "OMNIBASE_COMPAT_SOURCE ARG not found in Dockerfile.runtime"
    )
    sha = match.group("sha")
    # Commits that contain src/omnibase_compat/contracts/delegation/wire/:
    # 3e34ab9 feat(OMN-11024): add delegation wire DTOs
    # c1a878f chore: release v0.4.0 (includes delegation wire, current pin)
    delegation_commits = {
        "3e34ab94fad0a9db1c3b59f0e100c5da659b0792",
        "c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1",
    }
    assert sha in delegation_commits, (
        f"OMNIBASE_COMPAT_SOURCE pin {sha!r} predates the delegation wire contracts. "
        "Update the pin to a commit that includes src/omnibase_compat/contracts/delegation/wire/."
    )
