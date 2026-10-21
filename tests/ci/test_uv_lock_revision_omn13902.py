# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-13902: ``uv.lock`` is written in lockfile revision 5 with upload metadata.

The sibling lock refresh runs a current uv, which writes revision 5 and
records ``upload-time`` on registry artifacts. A lock an older uv wrote
(revision 1, no upload times) makes every refresh rewrite the whole file, so
the refresh pull request carries format churn instead of a sibling delta.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

UV_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"

MIN_REVISION = 5


def _lock() -> dict[str, object]:
    with UV_LOCK.open("rb") as handle:
        return tomllib.load(handle)


def test_uv_lock_is_at_least_revision_5() -> None:
    revision = _lock().get("revision")
    assert isinstance(revision, int)
    assert revision >= MIN_REVISION, (
        f"uv.lock is revision {revision}; the sibling lock refresh writes "
        f"revision {MIN_REVISION}"
    )


def test_uv_lock_records_registry_upload_times() -> None:
    packages = _lock()["package"]
    assert isinstance(packages, list)
    artifacts = [
        artifact
        for package in packages
        if "registry" in package.get("source", {})
        for artifact in [package.get("sdist", {}), *package.get("wheels", [])]
        if artifact
    ]
    assert artifacts, "uv.lock has no registry artifacts"
    timed = [artifact for artifact in artifacts if "upload-time" in artifact]
    assert len(timed) * 2 > len(artifacts), (
        f"only {len(timed)} of {len(artifacts)} registry artifacts in uv.lock "
        "carry upload-time"
    )
