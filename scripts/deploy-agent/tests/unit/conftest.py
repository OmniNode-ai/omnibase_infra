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


@pytest.fixture(autouse=True)
def _declare_lane_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMN-16939: DEPLOY_AGENT_ALLOWED_LANES is required and has no default.

    Tests that construct a ``DeployAgent`` are not testing the fence, so they
    get an explicit permissive one here rather than each re-declaring it. This
    is visible, not a bypass: the fence's own behaviour — including that an
    unset variable aborts startup — is asserted in
    ``test_lane_policy.py``, which deletes the variable via monkeypatch and so
    is unaffected by this fixture.
    """
    monkeypatch.setenv("DEPLOY_AGENT_ALLOWED_LANES", "dev,stability-test,prod")


@pytest.fixture(autouse=True)
def _declare_control_bus_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMN-18012: KAFKA_SECURITY_PROTOCOL is required and has no default.

    The agent refuses to start on an undeclared control-bus transport rather
    than inferring one from whether SASL credentials happen to be in the
    environment. Tests that construct a ``DeployAgent`` are not testing that
    declaration, so they get an explicit plaintext one here rather than each
    re-declaring it — the same visible arrangement as the lane fence above.

    This is not a bypass: the declaration's own behaviour, including that an
    unset variable refuses startup and that credential presence never selects a
    protocol, is asserted in ``test_kafka_config.py``, which deletes the
    variable in its own autouse fixture and so is unaffected by this one.
    """
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
