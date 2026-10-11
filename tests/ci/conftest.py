# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for CI tests.  # ai-slop-ok: pre-existing

This module provides common fixtures used across CI test modules,
promoting code reuse and consistent test patterns.

Ticket: OMN-255
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def create_test_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary Python test files.

    Uses pytest's tmp_path fixture for automatic cleanup after tests.
    This eliminates the need for try/finally cleanup patterns.

    Args:
        tmp_path: Pytest fixture providing a temporary directory unique to each test.

    Returns:
        A callable that takes content and optional filename, returns the file path.

    Example:
        def test_something(create_test_file):
            test_file = create_test_file("import kafka\\n")
            # File is automatically cleaned up after test
    """

    def _create(content: str, filename: str = "test_module.py") -> Path:
        """Create a temporary Python file with given content.

        Args:
            content: The content to write to the file.
            filename: Optional filename (default: test_module.py).

        Returns:
            Path to the created temporary file.
        """
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


@pytest.fixture
def forbidden_patterns() -> list[str]:
    """Standard forbidden import patterns for architecture compliance testing.

    Returns:
        List containing the standard test pattern ["kafka"].
    """
    return ["kafka"]


@pytest.fixture
def recorded_response(request: pytest.FixtureRequest) -> dict[str, object]:
    """Load the exact artifact named by a live_contact test (OMN-18648).

    These are recorded API bytes or incident replays with source provenance,
    not dicts recreating what a fetch seam is assumed to return.
    """
    marker = request.node.get_closest_marker("live_contact")
    if marker is None or len(marker.args) != 1 or not isinstance(marker.args[0], str):
        pytest.fail("recorded_response requires live_contact with one artifact path")
    root = Path(__file__).resolve().parents[2]
    path = (root / marker.args[0]).resolve()
    if not path.is_relative_to(root):
        pytest.fail("recorded_response artifact must be inside the repository")
    document = json.loads(path.read_text(encoding="utf-8"))
    provenance = document.get("_provenance") if isinstance(document, dict) else None
    if (
        not isinstance(provenance, dict)
        or not provenance.get("source")
        or not any(provenance.get(key) for key in ("captured_utc", "source_sha256"))
    ):
        pytest.fail("recorded_response lacks source and capture provenance")
    if provenance.get("source_file") and provenance.get("source_sha256"):
        import hashlib

        source = (root / provenance["source_file"]).resolve()
        if (
            not source.is_relative_to(root)
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != provenance["source_sha256"]
        ):
            pytest.fail(
                "recorded_response source bytes disagree with their recorded hash"
            )
    return document
