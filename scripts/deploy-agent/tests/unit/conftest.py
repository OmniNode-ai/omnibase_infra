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


@pytest.fixture(autouse=True)
def _declare_tracking_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMN-16442: DEPLOY_AGENT_TRACKING_REF is required and has no default.

    The branch this agent tracks — for its own self-update, for the default
    git ref of a rebuild command that omits one, and for the sibling-repo
    build-arg fallback — is a property of the deployment, so the package
    refuses to guess it. Tests that exercise unrelated concerns are not
    testing that declaration, so they get an explicit ``dev`` here rather than
    each re-declaring it; this is the same visible arrangement the lane fence
    and the control-bus transport already use above.

    This is not a bypass: the declaration's own behaviour — including that an
    unset variable raises — is asserted in ``test_tracking_ref.py``, which
    deletes the variable in its own autouse fixture and so is unaffected by
    this one.
    """
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")


@pytest.fixture(autouse=True)
def _derive_runtime_budget_from_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMN-18057: derive the runtime compose-up ceiling from THIS checkout.

    ``_compose_up`` now reads the lane's compose files to derive its ceiling
    (``deploy_agent.compose_budget``) and refuses fail-closed when it cannot --
    a ceiling that silently reverts to its floor because the model was
    unreadable is the same undetectable wrongness as the bare ``300`` this
    replaced. ``REPO_DIR`` is a deploy-HOST path that does not exist in the
    unit-test sandbox, so the files are repointed at the repository under test.

    This is a path repoint, not a stub: the real derivation runs, against the
    real ``docker/docker-compose.infra.yml`` + dev-lane overlay, so a compose
    change that moves ``start_period`` moves what these tests observe. The
    derivation's own behaviour -- including the fail-closed read and the
    ``service_healthy`` gating rule -- is asserted directly in
    ``test_compose_budget_omn18057.py``.
    """
    from deploy_agent import compose_budget
    from deploy_agent import executor as executor_mod

    docker_dir = Path(__file__).resolve().parents[4] / "docker"
    compose_files = (
        str(docker_dir / "docker-compose.infra.yml"),
        str(docker_dir / "docker-compose.dev-lane.yml"),
    )

    def _budget(
        lane: object, expected_services: list[str]
    ) -> compose_budget.ModelPhaseBudget:
        return compose_budget.derive_runtime_phase_budget(
            compose_files,
            expected_services,
            margin_seconds=executor_mod.RUNTIME_COMPOSE_UP_MARGIN_SECONDS,
            floor_seconds=executor_mod.RUNTIME_COMPOSE_UP_FLOOR_SECONDS,
        )

    monkeypatch.setattr(executor_mod, "runtime_compose_up_budget", _budget)


@pytest.fixture(autouse=True)
def _derive_image_build_budget_from_this_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18072: derive the runtime image-build ceiling from THIS checkout.

    ``_compose_build`` now reads the compose file it is about to invoke, and the
    Dockerfile that file names, to derive its ceiling
    (``deploy_agent.build_budget``) and refuses fail-closed when it cannot.
    ``REPO_DIR`` is a deploy-HOST path absent from the unit-test sandbox, so the
    compose file is repointed at the repository under test -- the same visible
    path repoint the compose-up budget already takes above.

    This is a repoint, not a stub: the real derivation runs against the real
    ``docker/docker-compose.infra.yml`` and ``docker/Dockerfile.runtime``, so
    adding a runtime service or a Dockerfile step moves what these tests
    observe. The derivation's own behaviour is asserted directly in
    ``test_build_budget_omn18072.py``.
    """
    from deploy_agent import build_budget
    from deploy_agent import executor as executor_mod

    docker_dir = Path(__file__).resolve().parents[4] / "docker"
    compose_file = str(docker_dir / "docker-compose.infra.yml")

    def _budget(
        profile: str, compose_files: tuple[str, ...] = (compose_file,)
    ) -> build_budget.ModelBuildBudget:
        # OMN-18108: the DEV addendum build passes the lane overlay as well, so
        # the repoint is per-file by basename rather than one fixed path.
        repointed = tuple(
            str(docker_dir / Path(candidate).name) for candidate in compose_files
        )
        return build_budget.derive_image_build_budget(
            repointed,
            profile,
            per_step_seconds=executor_mod.RUNTIME_IMAGE_BUILD_PER_STEP_SECONDS,
            per_image_seconds=executor_mod.RUNTIME_IMAGE_BUILD_PER_IMAGE_SECONDS,
            floor_seconds=executor_mod.RUNTIME_IMAGE_BUILD_FLOOR_SECONDS,
        )

    monkeypatch.setattr(executor_mod, "runtime_image_build_budget", _budget)
