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

from collections.abc import Callable
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import agent as agent_mod
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
    monkeypatch.setattr(executor_mod, "_load_promotion_guard", _NoopPromotionGuard)


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
def _derive_deps_budget_from_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMN-18692: derive the DEPS compose-up ceiling from THIS checkout.

    Same path repoint, same reason, as the runtime budget above: ``REPO_DIR``
    is a deploy-HOST path absent from the unit-test sandbox, and the deps
    derivation refuses fail-closed when it cannot read the compose model.

    A REPOINT, NOT A STUB. The real derivation runs against the real
    ``docker/docker-compose.infra.yml`` and the dev-lane overlay, so an edit
    that moves ``postgres``'s ``start_period`` or removes the ``service_healthy``
    gate on it moves what these tests observe. Like the image-build and gateway
    repoints above, it re-implements the production call and therefore must
    carry EVERY argument that call passes -- including the host term, reached
    through ``executor_mod`` so a test that pins a machine still reaches this
    closure. Omitting it would silently revert the ceiling to a machine-blind
    number for every test while still reporting green, which is the failure
    mode those two docstrings record happening twice.
    """
    from deploy_agent import compose_budget, recreate_supervisor
    from deploy_agent import executor as executor_mod

    docker_dir = Path(__file__).resolve().parents[4] / "docker"
    compose_files = (
        str(docker_dir / "docker-compose.infra.yml"),
        str(docker_dir / "docker-compose.dev-lane.yml"),
    )

    def _budget(
        lane: object, expected_services: list[str]
    ) -> recreate_supervisor.ModelDepsRecreateBudget:
        model = compose_budget.derive_runtime_phase_budget(
            compose_files,
            expected_services,
            margin_seconds=recreate_supervisor.DEPS_COMPOSE_UP_MARGIN_SECONDS,
            floor_seconds=recreate_supervisor.DEPS_COMPOSE_UP_FLOOR_SECONDS,
        )
        return recreate_supervisor.derive_deps_recreate_budget(
            model, executor_mod.probe_host_conditions()
        )

    monkeypatch.setattr(executor_mod, "deps_compose_up_budget", _budget)


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

    OMN-18615: it re-implements the production function's argument list, so it
    must carry EVERY argument that function passes -- including the host
    conditions. A repoint that quietly omits one silently reverts the
    derivation for every executor test while still calling itself a repoint,
    which is the failure mode this docstring is asserting it does not have.
    ``probe_host_conditions`` is reached through ``executor_mod`` rather than
    imported here so a test that pins a host still reaches this closure.
    ``test_executor_build_ceiling_host_omn18615.py`` fails if the host term
    goes missing from this fixture again.
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
            host=executor_mod.probe_host_conditions(),
        )

    monkeypatch.setattr(executor_mod, "runtime_image_build_budget", _budget)


@pytest.fixture(autouse=True)
def _resolve_preflight_script_from_this_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18123: point the required-compose-env preflight at THIS checkout.

    ``_preflight_required_compose_env`` now refuses with its own error class
    when the script is not a file, because a preflight that could not RUN and a
    preflight that RAN and found unset variables are different facts and were
    reported as the same one. ``REPO_DIR`` is a deploy-HOST path absent from the
    unit-test sandbox, so every test that drives compose generation would hit
    the new refusal instead of the behaviour it is asserting.

    This is a path repoint, not a stub -- the same one the compose-up and
    image-build budget fixtures above take. The script really is on disk in the
    repository under test, so deleting or renaming it moves what these tests
    observe. The refusal's own behaviour is asserted directly in
    ``test_preflight_script_missing_omn18123.py``, which repoints it the other
    way.
    """
    from deploy_agent import executor as executor_mod

    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "preflight_required_compose_env.py"
    )
    monkeypatch.setattr(
        executor_mod, "preflight_required_compose_env_script", lambda: str(script)
    )


@pytest.fixture(autouse=True)
def _derive_gateway_deploy_budget_from_this_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18200: derive the gateway deploy ceiling from THIS checkout.

    ``_deploy_gateway_lane`` now reads the gateway compose file, the
    Dockerfiles it names, and the gateway systemd unit's own
    ``ExecReload --wait-timeout`` to derive its ceiling
    (``deploy_agent.gateway_budget``) and refuses fail-closed when it cannot --
    a ceiling that silently reverted to a floor because the model was
    unreadable would be the same undetectable wrongness as the flat
    floor-plus-300 this replaced. ``REPO_DIR`` is a deploy-HOST path absent
    from the unit-test sandbox, so both files are repointed at the repository
    under test -- the same visible path repoint the compose-up and
    image-build budget fixtures above take.

    This is a repoint, not a stub: the real derivation runs against the real
    ``docker/docker-compose.gateway.yml`` and
    ``docker/gateway/onex-gateway-forwarder.service``, so a compose or unit
    change moves what these tests observe. The derivation's own behaviour is
    asserted directly in ``test_gateway_budget_omn18200.py``.
    """
    from deploy_agent import executor as executor_mod
    from deploy_agent import gateway_budget

    docker_dir = Path(__file__).resolve().parents[4] / "docker"
    compose_file = str(docker_dir / "docker-compose.gateway.yml")
    service_unit = str(docker_dir / "gateway" / "onex-gateway-forwarder.service")

    def _budget() -> gateway_budget.ModelGatewayDeployBudget:
        return gateway_budget.derive_gateway_deploy_budget(
            (compose_file,),
            executor_mod.GATEWAY_BUILD_PROFILE,
            service_unit,
            per_step_seconds=executor_mod.RUNTIME_IMAGE_BUILD_PER_STEP_SECONDS,
            per_image_seconds=executor_mod.RUNTIME_IMAGE_BUILD_PER_IMAGE_SECONDS,
            build_floor_seconds=executor_mod.RUNTIME_IMAGE_BUILD_FLOOR_SECONDS,
            reload_margin_seconds=executor_mod.GATEWAY_RECREATE_MARGIN_SECONDS,
            reload_floor_seconds=executor_mod.GATEWAY_RECREATE_FLOOR_SECONDS,
            # OMN-18615 second pass: like the image-build repoint above, this
            # one re-implements the production call and so must carry EVERY
            # argument it passes. Omitting the host term silently reverts the
            # gateway ceiling to the machine-blind OMN-18072 derivation for
            # every test, while still reporting green. That is the SECOND time
            # this exact harness shape hid the defect -- the first was the
            # image-build repoint -- which is why both are now pinned by
            # live-path tests rather than by reading the fixture.
            host=executor_mod.probe_host_conditions(),
        )

    monkeypatch.setattr(executor_mod, "gateway_deploy_budget", _budget)


@pytest.fixture(autouse=True)
def _resolve_gateway_lane_from_this_checkout(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """OMN-18134: make the gateway lane inert for tests that are not testing it.

    A DEV runtime rebuild now also deploys the gateway compose project. That
    step SHELLS OUT TO A MUTATING SCRIPT (``scripts/deploy-gateway.sh
    --execute`` builds images, rewrites root-owned host files and reloads a
    systemd unit), and several existing tests drive ``rebuild_scope`` while
    stubbing only ``_compose_build`` / ``_compose_up`` / ``_pull_pinned_image``
    by hand. Leaving the new step live for those would let a unit test run a
    real deploy on whatever machine the suite happens to be on.

    So ``_deploy_gateway_lane`` is stubbed by default here, and the two seams
    are repointed at resolvable paths for the tests that DO exercise it. This
    is the same visible arrangement ``_stub_promotion_guard`` already uses
    above: tests that verify the behaviour opt out with the ``gateway_lane``
    marker and control the seams themselves.

    The script seam is a path repoint, not a stub -- it points at the real
    ``scripts/deploy-gateway.sh`` in the repository under test, so deleting or
    renaming it moves what the opted-out tests observe. The env file is a
    genuine fixture, because the real one is operator-supplied and is not in git
    by design; it declares both required maps against files that exist. Both
    refusals' own behaviour is asserted directly in
    ``test_gateway_lane_scope_omn18134.py``, which repoints them the other way.
    """
    from deploy_agent import executor as executor_mod

    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "scripts" / "deploy-gateway.sh"
    monkeypatch.setattr(executor_mod, "deploy_gateway_script", lambda: str(script))

    lane_dir = tmp_path_factory.mktemp("gateway-lane")
    lines: list[str] = []
    for name in sorted(executor_mod.GATEWAY_REQUIRED_MAP_VARS):
        target = lane_dir / f"{name.lower()}.yaml"
        target.write_text("{}\n", encoding="utf-8")
        lines.append(f"{name}={target}")
    env_file = lane_dir / "gateway.env"
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(executor_mod, "gateway_env_file", lambda: str(env_file))

    if request.node.get_closest_marker("gateway_lane") is not None:
        return
    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "_deploy_gateway_lane",
        lambda self, *args, **kwargs: None,
    )


@pytest.fixture(autouse=True)
def _keep_the_deps_convergence_observation_off_the_docker_daemon(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-18640: no-op the deps-convergence observation unless a test opts in.

    The deps leg now reads, before it acts, what compose would render for each
    core service and what the running containers carry -- `compose config
    --hash`, `compose config --format json`, and one `docker inspect` per
    service. Every one of those reaches the real machine, and dozens of
    existing tests drive ``Scope.CORE`` while stubbing only
    ``executor_mod._run``. Left live they do not fail; they SUCCEED and prepend
    three commands to the list those tests index from, which is how a change
    that observes can break tests that act.

    So the observation is a no-op by default, and the tests that are about it
    opt in with the ``deps_convergence`` marker and drive the real method.
    This is the same visible arrangement
    ``_keep_the_deps_recreate_off_the_docker_daemon`` below already uses, for
    the same reason and against the same hazard.

    It is not a hole: the observation's own behaviour -- that it names a
    changed dependency, that it never gates, that it records no environment
    value, and that it runs before the compose up -- is asserted directly in
    ``test_visible_deps_convergence_omn18640.py``.
    """
    if request.node.get_closest_marker("deps_convergence") is not None:
        return

    from deploy_agent import executor as executor_mod

    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "observe_deps_convergence",
        lambda self, lane: [],
    )


@pytest.fixture(autouse=True)
def _keep_the_deps_recreate_off_the_docker_daemon(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-18692: run the deps recreate through ``_run`` unless a test opts in.

    The DEPS compose-up is no longer a ``subprocess.run`` with a flat timeout.
    It is SUPERVISED: it spawns a live child through ``_spawn_compose``, polls
    ``docker compose ps`` while it runs, and reads ``/proc/loadavg`` plus
    ``docker builder du`` to derive its ceiling. Every one of those reaches the
    real machine, and dozens of existing tests drive ``Scope.CORE`` or
    ``Scope.FULL`` while stubbing only ``executor_mod._run``. Left live, those
    tests would run a real ``docker compose up --force-recreate`` on whatever
    machine the suite happens to be on -- which on the lab host is the dev lane.

    So the supervised method is replaced by default with a shim that calls the
    same ``_run`` seam those tests already stub, carrying the same compose argv.
    They therefore observe exactly what they observed before this change, and
    they observe it through the stub they installed rather than through a
    bypass they cannot see.

    This is the same visible arrangement ``_stub_promotion_guard`` and
    ``_resolve_gateway_lane_from_this_checkout`` already use above, and it is
    not a hole in the coverage: tests that exercise the supervisor opt out with
    the ``deps_recreate`` marker and drive the real method, and the
    supervisor's own behaviour -- the anchored ceiling, the refusal to cancel a
    live recreate, the deferral and the deps convergence -- is asserted
    directly in ``test_deps_recreate_supervisor_omn18692.py`` and
    ``test_executor_deps_recreate_omn18692.py``.
    """
    if request.node.get_closest_marker("deps_recreate") is not None:
        return

    from deploy_agent import executor as executor_mod

    def _run_deps_through_the_stubbed_seam(
        self: object,
        cmd: list[str],
        *,
        phase: object,
        expected: list[str],
        lane: object,
        extra_env: object = None,
    ) -> str:
        result = executor_mod._run(
            cmd,
            timeout=executor_mod.PHASE_TIMEOUTS[executor_mod.Phase.CORE],
            env=executor_mod._compose_env(extra_env),
        )
        return result.stderr.strip() if result.returncode != 0 else ""

    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "_supervised_deps_recreate",
        _run_deps_through_the_stubbed_seam,
    )


@pytest.fixture(autouse=True)
def _reset_loaded_code_identity() -> object:
    """OMN-18200: the loaded-code identity is per PROCESS, so clear it per test.

    ``deploy_agent.loaded_code`` holds one module-level sha, recorded once at
    agent startup, because that is exactly what it models: the commit whose tree
    THIS process imported. A pytest session is one process running many tests,
    so without this reset the first test to record one would silently supply it
    to every later test -- and a self-update test that passes on a neighbour's
    value is proving nothing.

    Reset on both sides of the yield so a test that records one does not leak it
    forwards, and a test that expects the unrecorded refusal is not defeated by
    a value left behind.
    """
    from deploy_agent import loaded_code

    loaded_code.reset_loaded_code_sha()
    yield
    loaded_code.reset_loaded_code_sha()


@pytest.fixture
def declare_loaded_code_sha(monkeypatch: pytest.MonkeyPatch):
    """Declare the sha of the code the process under test loaded (OMN-18200).

    ``self_update`` now compares the LOADED code to the clone rather than the
    clone to the remote, so every test that reaches that comparison has to say
    which code it is pretending to run. Tests driving a real clone can call
    ``record_loaded_code_sha`` directly; tests that stub ``_run`` have no clone
    to read, so the resolver is repointed and the real recording path still runs
    -- a repoint, not a bypass, and ``loaded_code``'s own behaviour (including
    that an unrecorded read raises) is asserted in
    ``test_self_update_loaded_code_omn18200.py``.
    """
    from deploy_agent import loaded_code

    def _declare(sha: str) -> str:
        monkeypatch.setattr(loaded_code, "resolve_clone_sha", lambda _agent_dir: sha)
        return loaded_code.record_loaded_code_sha("<declared by test>")

    return _declare


class _InlineSettlePool(Executor):
    """Runs each settle on the calling thread, at submission (OMN-19501).

    The settle worker is a real thread in production. Most tests here are about
    WHAT a job and its settle do, and assert on it right after ``_run_deploy``
    returns; running the settle inline keeps them deterministic without each one
    learning to drain a pool. The code path is the production one -- only the
    thread differs. Tests about the concurrency itself opt out with the
    ``real_settle_pool`` marker and drive the real worker.
    """

    def submit(  # type: ignore[override]
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - carried on the future
            future.set_exception(exc)
        return future


@pytest.fixture(autouse=True)
def _inline_settle_pool(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the settle inline unless the test drives the real worker (OMN-19501)."""
    if request.node.get_closest_marker("real_settle_pool") is not None:
        return
    monkeypatch.setattr(agent_mod, "_new_settle_pool", _InlineSettlePool)
