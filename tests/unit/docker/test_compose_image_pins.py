# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose image pin tests for Docker infrastructure.

Asserts that no service image in docker-compose.infra.yml uses the :latest tag.
Using :latest makes docker compose pull non-deterministic across runs and can
silently change versions, breaking dashboards or causing schema incompatibilities.

OMN-4303: Pin Phoenix image away from :latest.
"""

from __future__ import annotations

import re

import pytest

from tests.unit.docker.conftest import COMPOSE_FILE_PATH

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_LINE_RE = re.compile(
    r"^\s+image:\s+(?P<image>\S+)",
    re.MULTILINE,
)


def _extract_image_lines(content: str) -> list[str]:
    """Return all image values found in the compose file."""
    return [m.group("image") for m in _IMAGE_LINE_RE.finditer(content)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_latest_image_tags(compose_file_content: str) -> None:
    """Assert that no service in docker-compose.infra.yml uses :latest.

    :latest is non-deterministic: docker compose pull can silently change
    the running version between runs. All images must be pinned to a
    specific tag (e.g. arizephoenix/phoenix:13.11.0).

    OMN-4303: Added after phoenix was found using :latest at line 907.
    """
    images = _extract_image_lines(compose_file_content)
    latest_images = [img for img in images if img.endswith(":latest")]

    assert not latest_images, (
        f"Found {len(latest_images)} image(s) using ':latest' tag in "
        f"{COMPOSE_FILE_PATH}. Pin each to a specific version:\n"
        + "\n".join(f"  {img}" for img in latest_images)
    )


def test_phoenix_image_is_pinned(compose_file_content: str) -> None:
    """Assert that the phoenix service image is pinned to a specific tag.

    OMN-4303: Previously arizephoenix/phoenix:latest; now pinned to 13.11.0.
    """
    images = _extract_image_lines(compose_file_content)
    phoenix_images = [img for img in images if "arizephoenix/phoenix" in img]

    assert phoenix_images, "Phoenix service image not found in compose file."

    for img in phoenix_images:
        assert not img.endswith(":latest"), (
            f"Phoenix image '{img}' must be pinned to a specific tag, not ':latest'."
        )
        tag = img.split(":")[-1] if ":" in img else ""
        assert tag, (
            f"Phoenix image '{img}' has no tag — must be pinned (e.g. :13.11.0)."
        )
