# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static wiring coverage for the OMN-15718 bounded-timeout fix in
deploy-runtime.sh.

Live-execution coverage for the shared helpers themselves lives in
test_compose_wait_timeout.py. This file guards that every `docker compose ...
up ...` call site in deploy-runtime.sh's deploy/restart/rollback path actually
routes through compose_up_bounded() (so the bounded deadline cannot silently
regress out of one call site while staying wired everywhere else), and that
cleanup_on_exit() reconciles container start state, not just image tags.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
LIB_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "compose_wait_timeout.sh"


def _text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{(?P<body>.*?)\n\}}",
        _text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"{name}() function not found in deploy-runtime.sh"
    return match.group("body")


@pytest.mark.unit
def test_shared_lib_exists_and_is_sourced() -> None:
    assert LIB_SCRIPT.is_file(), (
        "scripts/runtime_build/compose_wait_timeout.sh is missing; deploy-runtime.sh "
        "and refresh_stability_lane.sh both depend on it for OMN-15718."
    )
    assert (
        'source "${SCRIPT_DIR_FOR_ENV}/runtime_build/compose_wait_timeout.sh"'
        in _text()
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "function_name",
    [
        "ensure_core_infra_ready",
        "warm_broker_topic_provisioning",
        "run_runtime_migration_preflight",
        "bringup_full_stack",
        "restart_services",
    ],
)
def test_every_compose_up_call_site_is_bounded(function_name: str) -> None:
    """Every function that issues a `docker compose ... up ...` call must
    route it through compose_up_bounded() -- OMN-15718's whole point is that
    none of these can hang indefinitely on an unresolvable dependency."""
    body = _function_body(function_name)
    assert "compose_up_bounded" in body, (
        f"{function_name}() issues a compose 'up' call that is not wrapped in "
        "compose_up_bounded() -- it can hang indefinitely again (OMN-15718)."
    )


@pytest.mark.unit
def test_warm_broker_topic_provisioning_bounds_both_up_calls() -> None:
    """This function has TWO up calls (broker readiness + partition cap) --
    both must be bounded, not just the first."""
    body = _function_body("warm_broker_topic_provisioning")
    assert body.count("compose_up_bounded") >= 2, (
        "warm_broker_topic_provisioning() must bound both its compose up calls "
        f"(broker readiness + partition cap); found {body.count('compose_up_bounded')}."
    )


@pytest.mark.unit
def test_broker_up_guard_still_tolerates_failure_not_fatal() -> None:
    """Regression guard for OMN-13364: bounding the broker up call must not
    reintroduce the fatal-on-collision bug that guard originally fixed."""
    body = _function_body("warm_broker_topic_provisioning")
    assert (
        'if ! compose_up_bounded "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS}" "${broker_up_cmd[@]}"; then'
        in body
    ), (
        "The broker compose up must stay guarded (non-fatal on failure) even "
        "after being wrapped in compose_up_bounded() -- OMN-13364's "
        "name-collision tolerance must survive the OMN-15718 bounding."
    )


@pytest.mark.unit
def test_docker_wait_calls_are_also_bounded() -> None:
    """`docker wait` blocks until the container exits with no deadline of its
    own -- both one-shot wait sites must be bounded the same way as `up`."""
    cap_body = _function_body("warm_broker_topic_provisioning")
    assert "timeout --kill-after=15" in cap_body and "docker wait" in cap_body

    preflight_body = _function_body("run_runtime_migration_preflight")
    assert (
        "timeout --kill-after=15" in preflight_body and "docker wait" in preflight_body
    )


@pytest.mark.unit
def test_cleanup_on_exit_reconciles_container_start_state() -> None:
    """AC1 (OMN-15718): retagging :latest is not enough -- cleanup_on_exit()
    must also reconcile actual container running state on a failed deploy."""
    body = _function_body("cleanup_on_exit")
    restore_idx = body.find("restore_latest_image_tags")
    reconcile_idx = body.find("reconcile_runtime_container_start_state")
    assert restore_idx != -1, (
        "cleanup_on_exit must still call restore_latest_image_tags"
    )
    assert reconcile_idx != -1, (
        "cleanup_on_exit must call reconcile_runtime_container_start_state() so a "
        "failed restart_services()/bringup_full_stack() call cannot leave "
        "RUNTIME_BUILD_SERVICES containers stranded in 'Created' (OMN-15718)."
    )
    assert restore_idx < reconcile_idx, (
        "Image tags must be restored before container state is reconciled "
        "(reconciliation may recreate/start containers against the restored images)."
    )


@pytest.mark.unit
def test_reconcile_runtime_container_start_state_uses_shared_helper() -> None:
    body = _function_body("reconcile_runtime_container_start_state")
    assert "reconcile_container_running_state" in body, (
        "reconcile_runtime_container_start_state() must delegate the actual "
        "start/teardown decision to the shared reconcile_container_running_state() "
        "helper (single source of truth, shared with refresh_stability_lane.sh)."
    )
    assert "|| true" in body, (
        "The reconcile call must be guarded (|| true) -- this runs inside the "
        "EXIT trap under set -e; a teardown (return 1) must not abort the rest "
        "of cleanup_on_exit (e.g. lock release)."
    )
