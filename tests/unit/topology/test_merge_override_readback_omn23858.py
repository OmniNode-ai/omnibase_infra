# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The checked-in dev-lane readback names the host, the revision and the grants (OMN-23858).

AC4 of the contract is done only when running and observed where it runs: the
dev lane's grant read-back is checked in at docs/readbacks/OMN-23858_dev_grants.txt.
These tests hold that file to its own shape: the host, image revision and
release lines must be present and well formed, the four merge override tables
must each carry the runtime's INSERT, SELECT and UPDATE grant row, and the
release tag must name the same revision the runtime container above reports.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_READBACK = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "readbacks"
    / "OMN-23858_dev_grants.txt"
)
_TABLES = (
    "merge_override_open_findings",
    "merge_override_pending",
    "merge_override_receipts",
    "merge_override_verdicts",
)
_IMAGE_REVISION_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_RE: re.Pattern[str] = re.compile(r"^dev-\d{14}-[0-9a-f]{12}$")


def _readback() -> dict[str, str]:
    lines = _READBACK.read_text(encoding="utf-8").splitlines()
    readback: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        readback[key] = value
    return readback


def test_the_readback_names_the_host_the_image_revision_and_the_release() -> None:
    readback = _readback()
    for key in ("host", "image_revision", "omnibase_infra_release"):
        assert readback.get(key), key
    revision = readback["image_revision"]
    assert _IMAGE_REVISION_RE.match(revision), revision
    release = readback["omnibase_infra_release"]
    assert _RELEASE_RE.match(release), release


def test_the_readback_lists_the_runtime_grants_on_the_four_tables() -> None:
    lines = _READBACK.read_text(encoding="utf-8").splitlines()
    for table in _TABLES:
        row = f"omninode_runtime|{table}|INSERT,SELECT,UPDATE"
        assert row in lines, row


def test_the_release_tag_ends_with_the_running_image_revision_prefix() -> None:
    readback = _readback()
    assert readback["omnibase_infra_release"].endswith(readback["image_revision"][:12])
