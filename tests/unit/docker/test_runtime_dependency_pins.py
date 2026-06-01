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
    # Commits verified to contain src/omnibase_compat/contracts/delegation/wire/.
    # This set is a lower-bound allowlist: every listed commit contains the delegation
    # wire DTOs.  When the Dockerfile pin advances to a newer commit, add it here.
    # Do NOT remove entries; older commits remain valid lower-bound anchors.
    #
    # Ancestry cannot be checked offline (no git clone in unit tests), so we maintain
    # this verified set instead of a computed ancestry check.
    #
    # 3e34ab9 feat(OMN-11024): add delegation wire DTOs          (first introduction)
    # c1a878f chore: release v0.4.0 (pin as of OMN-12421 baseline)
    # 4d887307 fix(OMN-12245): release delegation escalation DTOs (pin as of OMN-12421 advance)
    _DELEGATION_WIRE_VERIFIED_COMMITS = {
        "3e34ab94fad0a9db1c3b59f0e100c5da659b0792",
        "c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1",
        "4d887307aae34d9d40d389ba91070cb411ce3df5",
    }
    assert sha in _DELEGATION_WIRE_VERIFIED_COMMITS, (
        f"OMNIBASE_COMPAT_SOURCE pin {sha!r} is not in the verified set of commits "
        "known to include src/omnibase_compat/contracts/delegation/wire/. "
        "If this is a forward pin, add the new SHA to _DELEGATION_WIRE_VERIFIED_COMMITS "
        "in this test after confirming the commit contains delegation wire contracts."
    )
