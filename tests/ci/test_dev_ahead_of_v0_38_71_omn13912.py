# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev's live ``[project].version`` is strictly ahead of the published v0.38.71.

Publishing ``v0.38.71`` (``cbcfb93cf``, OMN-18595) left dev's
``pyproject.toml`` at ``0.38.71``: equal to the newly published tag, which arms
the OMN-13412 release-identity gate against dev. The post-release bump
(OMN-13912) moves dev to ``0.38.72``.

The test drives the real ``decide()`` over the live ``pyproject.toml`` and
requires it to call dev disarmed against ``v0.38.71``. Versions only rise, so it
stays green after later releases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.post_release_dev_bump import (
    ACTION_NOOP,
    decide,
    parse_final_version,
    read_project_version,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

RELEASED_TAG = "v0.38.71"


def test_dev_is_strictly_ahead_of_the_published_v0_38_71() -> None:
    dev_version = read_project_version(PYPROJECT)
    assert parse_final_version(dev_version, label="dev") > parse_final_version(
        RELEASED_TAG, label="released"
    ), f"dev {dev_version} is not ahead of published {RELEASED_TAG}"


def test_the_release_train_sees_dev_disarmed_after_v0_38_71() -> None:
    decision = decide(
        dev_version=read_project_version(PYPROJECT), released_version=RELEASED_TAG
    )
    assert decision.action == ACTION_NOOP, decision.reason
