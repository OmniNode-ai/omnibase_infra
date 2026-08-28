# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""`--cold` lane-scope guard ratchet for deploy-runtime.sh (OMN-16803).

Two defects this pins.

D1 — the prod refusal was reachable only through ``PROD_LANE``, which is set by
``--prod`` or ``ONEX_DEPLOY_LANE=prod`` and NOT by the compose project. That
check lives in ``parse_args``, which runs *before* ``resolve_compose_project``,
so ``--cold`` with ``OMNIBASE_INFRA_COMPOSE_PROJECT=omnibase-infra-prod`` and no
``--prod`` flag went straight past it. ``guard_cold_bringup_lane_scope`` runs
after the lane is resolved and therefore sees the real target however it was
selected.

D2 — the judge lane had no ``--cold`` refusal at all, despite the lane map
declaring it "NOT authorized for mutation — read-only".

And the correction that motivated both: ``stability-test`` must remain ALLOWED.
The cold-lane runbook previously scoped ``--cold`` to dev only, lumping stability
in with prod under a prod-specific rationale (a workspace image the
prod-promotion gate refuses). That rationale does not transfer — stability-test
is built in workspace mode by design by its own sanctioned refresh script — and
the scoping left a partially-cold governed lane with no sanctioned recovery path
at all. The stability lane sat at 6 of its declared services for a month behind
exactly that dead end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DEPLOY_SCRIPT = _REPO / "scripts" / "deploy-runtime.sh"


def _script() -> str:
    return _DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _guard_body() -> str:
    """The text of guard_cold_bringup_lane_scope, up to the next function."""
    raw = _script()
    start = raw.index("guard_cold_bringup_lane_scope() {")
    rest = raw[start:]
    end = rest.index("\nguard_hotpatch_ledger() {")
    return rest[:end]


def test_cold_lane_scope_guard_exists() -> None:
    """The guard must exist as a named function, not inline in main()."""
    assert "guard_cold_bringup_lane_scope() {" in _script(), (
        "deploy-runtime.sh lost guard_cold_bringup_lane_scope — the --cold lane "
        "scope check (OMN-16803) must stay a named, greppable guard"
    )


def test_cold_lane_scope_guard_runs_after_compose_project_resolution() -> None:
    """Ordering is the whole point: the guard must see the RESOLVED lane.

    If it were called before ``compose_project="$(resolve_compose_project)"`` it
    would be back to guessing the lane from flags, which is precisely the hole
    that let ``--cold`` reach the prod lane via the compose-project env var.
    """
    raw = _script()
    resolve_at = raw.index('compose_project="$(resolve_compose_project)"')
    call_at = raw.index('guard_cold_bringup_lane_scope "${compose_project}"')
    assert resolve_at < call_at, (
        "guard_cold_bringup_lane_scope is invoked before the compose project is "
        "resolved; it cannot see the real target lane there"
    )


def test_cold_lane_scope_guard_runs_before_any_build_or_bringup() -> None:
    """A refusal must land before anything is built, recreated, or started."""
    raw = _script()
    call_at = raw.index('guard_cold_bringup_lane_scope "${compose_project}"')
    for later in (
        'build_images "${deploy_target}"',
        'bringup_full_stack "${deploy_target}"',
        'restart_services "${deploy_target}"',
    ):
        assert call_at < raw.index(later), (
            f"guard_cold_bringup_lane_scope runs after {later!r} — a refused "
            f"lane would already have been mutated"
        )


@pytest.mark.parametrize("lane", ["prod", "judge"])
def test_refused_lanes_exit_nonzero(lane: str) -> None:
    """prod and judge must each have an explicit refusing case arm."""
    body = _guard_body()
    arm = re.search(rf"^\s*{lane}\)\s*$(.*?)^\s*;;\s*$", body, re.MULTILINE | re.DOTALL)
    assert arm is not None, f"no `{lane})` case arm in guard_cold_bringup_lane_scope"
    assert "log_error" in arm.group(1), f"{lane} arm does not log an error"
    assert re.search(r"^\s*exit [1-9]", arm.group(1), re.MULTILINE), (
        f"{lane} arm does not exit non-zero — the guard would fall through and "
        f"the cold bring-up would proceed against a forbidden lane"
    )


def test_stability_test_is_allowed() -> None:
    """stability-test must NOT be refused — that is the OMN-16803 correction.

    A partially-cold governed lane needs a sanctioned repair path; refusing
    stability here would restore the dead end this ticket exists to close.
    """
    body = _guard_body()
    arm = re.search(
        r"^\s*stability-test\)\s*$(.*?)^\s*;;\s*$", body, re.MULTILINE | re.DOTALL
    )
    assert arm is not None, "no `stability-test)` case arm — lane scope is unstated"
    assert not re.search(r"^\s*exit [1-9]", arm.group(1), re.MULTILINE), (
        "the stability-test arm exits non-zero; --cold must remain available as "
        "the sanctioned partial-cold repair path for this lane (OMN-16803)"
    )


def test_guard_is_a_noop_when_cold_not_requested() -> None:
    """A warm --restart deploy must be completely unaffected by this guard."""
    body = _guard_body()
    assert re.search(
        r'if \[\[ "\$\{COLD_FULL_BRINGUP\}" != true \]\]; then\s*\n\s*return 0',
        body,
    ), (
        "guard_cold_bringup_lane_scope does not short-circuit when --cold was "
        "not requested; warm deploys must be byte-for-byte unchanged"
    )
