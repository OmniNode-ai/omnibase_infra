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
        r'omnibase_compat/archive/(?P<ref>[0-9a-f]{40}|v\d+\.\d+\.\d+)\.tar\.gz"',
        dockerfile,
    )

    assert match is not None, (
        "OMNIBASE_COMPAT_SOURCE ARG not found in Dockerfile.runtime"
    )
    ref = match.group("ref")
    # Refs verified to contain src/omnibase_compat/contracts/delegation/wire/.
    # This set is a lower-bound allowlist: every listed commit contains the delegation
    # wire DTOs.  When the Dockerfile pin advances to a newer commit or immutable
    # release tag, add it here. Do NOT remove entries; older commits remain valid
    # lower-bound anchors.
    #
    # Ancestry cannot be checked offline (no git clone in unit tests), so we maintain
    # this verified set instead of a computed ancestry check.
    #
    # 3e34ab9 feat(OMN-11024): add delegation wire DTOs          (first introduction)
    # c1a878f chore: release v0.4.0 (pin as of OMN-12421 baseline)
    # 4d887307 fix(OMN-12245): release delegation escalation DTOs (pin as of OMN-12421 advance)
    # v0.5.4  release tag -> 11325233b202449a216b5c862084be3b5e0cae2c
    _DELEGATION_WIRE_VERIFIED_REFS = {
        "3e34ab94fad0a9db1c3b59f0e100c5da659b0792",
        "c1a878f1339d396a7f11dee42ea9f0e30fb9c1d1",
        "4d887307aae34d9d40d389ba91070cb411ce3df5",
        "v0.5.4",
    }
    assert ref in _DELEGATION_WIRE_VERIFIED_REFS, (
        f"OMNIBASE_COMPAT_SOURCE pin {ref!r} is not in the verified set of refs "
        "known to include src/omnibase_compat/contracts/delegation/wire/. "
        "If this is a forward pin, add the new ref to _DELEGATION_WIRE_VERIFIED_REFS "
        "in this test after confirming the commit contains delegation wire contracts."
    )


def test_runtime_image_includes_session_orchestrator_probe_toolchain(
    dockerfile_path: Path,
) -> None:
    """The deployed runtime owns dry-run health and repo probes."""
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    runtime_match = re.search(
        r"^FROM\s+.+\s+AS\s+runtime\s*$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert runtime_match is not None, "Runtime stage not found in Dockerfile.runtime"
    runtime_stage = dockerfile[runtime_match.start() :]
    runtime_marker_length = len(runtime_match.group(0))
    next_stage_match = re.search(
        r"^FROM\s+",
        runtime_stage[runtime_marker_length:],
        flags=re.MULTILINE,
    )
    if next_stage_match is not None:
        runtime_stage = runtime_stage[
            : runtime_marker_length + next_stage_match.start()
        ]

    assert "OMNI_HOME=/app" in runtime_stage
    assert "COPY --from=uv-bin /uv /uvx /usr/local/bin/" in runtime_stage
    assert re.search(r"^\s+git \\\s*$", runtime_stage, flags=re.MULTILINE) is not None
    assert (
        re.search(r"^\s+openssh-client \\\s*$", runtime_stage, flags=re.MULTILINE)
        is not None
    )
