# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared fixtures for deploy-agent unit tests.

OMN-12626 (R1): release-mode ``_compose_build`` now runs the prod
promotion-lineage guard, which inspects the git state of ``REPO_DIR`` (a deploy
HOST path that does not exist in the unit-test sandbox). Unit tests that
exercise unrelated build-arg / staging concerns are not testing lineage, so by
default the guard is stubbed to a no-op here.

This is explicit and visible (not a hidden bypass): the guard's own behavior is
covered by ``scripts/test_check_prod_promotion_lineage.py`` (the single source
of truth), and ``test_executor_promotion_lineage.py`` re-stubs
``_load_promotion_guard`` to assert the deploy-agent enforcement path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod


class _NoopPromotionGuard:
    """No-op stand-in for the scripts/ promotion-lineage guard module."""

    class ProdLineageError(RuntimeError):
        pass

    def assert_prod_build_promoted(self, repo_dir: Path) -> str:
        return "0" * 40


@pytest.fixture(autouse=True)
def _stub_promotion_guard(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub the promotion-lineage guard by default for deploy-agent unit tests.

    Tests that explicitly verify the guard (marked by overriding
    ``_load_promotion_guard`` themselves) opt out via the ``promotion_guard``
    marker so they control the stub.
    """
    if request.node.get_closest_marker("promotion_guard") is not None:
        return
    monkeypatch.setattr(
        executor_mod, "_load_promotion_guard", lambda: _NoopPromotionGuard()
    )
