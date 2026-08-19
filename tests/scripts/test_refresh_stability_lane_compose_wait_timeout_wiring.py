# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static wiring coverage for the OMN-15718 fix in refresh_stability_lane.sh.

Forensic finding this closes (2026-08-05, .201, lab-runtimes lane): the
script's failure-path rollback retagged core-service images back to the prior
known-good digest correctly, but the subsequent targeted `docker compose up`
recreate left runtime-effects/runtime-worker stranded in `Created` state
(compose honors `depends_on: migration-gate: condition: service_healthy` even
under `--no-deps`, and migration-gate could never become healthy once
forward-migration had already failed). That `up` call had no bounded
deadline, so it hung indefinitely instead of failing fast, and the script
never reached the health-gate/receipt stage -- recovered manually with
`docker start`.

Live-execution coverage for the shared helpers themselves lives in
test_compose_wait_timeout.py. This file guards the wiring: the rollback path
must be bounded, must reconcile every core-service container back to running
(or explicitly tear it down), must prove the post-rollback census matches the
pre-attempt census, and must surface a typed, distinguishable result when it
cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
LIB_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "compose_wait_timeout.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


@pytest.mark.unit
def test_shared_lib_is_sourced() -> None:
    assert LIB_SCRIPT.is_file()
    assert 'source "${SCRIPT_DIR}/compose_wait_timeout.sh"' in _text()


@pytest.mark.unit
def test_pre_attempt_container_status_census_is_captured() -> None:
    """Step 1 (pre-state capture) must record each core service's running
    STATE, not just its image id -- otherwise there is nothing to compare the
    post-rollback census against."""
    text = _text()
    assert "declare -A PRE_CONTAINER_STATUS" in text
    assert 'PRE_CONTAINER_STATUS["${svc}"]="$(container_status "${container}")"' in text
    # Must be captured in the SAME pre-state loop as PRE_IMAGE_IDS, before any
    # mutation (image tagging) happens.
    pre_state_idx = text.find("=== Capture pre-state ===")
    tag_idx = text.find("=== Tag preflight rollback anchor")
    census_idx = text.find('PRE_CONTAINER_STATUS["${svc}"]')
    assert pre_state_idx != -1 and tag_idx != -1 and census_idx != -1
    assert pre_state_idx < census_idx < tag_idx, (
        "PRE_CONTAINER_STATUS must be captured during step 1 (pre-state), "
        "strictly before step 2 (preflight rollback tagging) touches anything."
    )


@pytest.mark.unit
def test_rollback_recreate_is_bounded() -> None:
    """The targeted recreate that hung indefinitely in the forensic log must
    be routed through compose_up_bounded()."""
    text = _text()
    match = re.search(
        r'compose_up_bounded "\$\{RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS\}" \\\s*'
        r"\n\s*docker compose -p \"\$\{COMPOSE_PROJECT\}\"",
        text,
    )
    assert match is not None, (
        "The rollback's targeted `docker compose ... up --force-recreate` call "
        "must be wrapped in compose_up_bounded() (OMN-15718) -- an unbounded "
        "call here is the exact defect that hung indefinitely on 2026-08-05."
    )
    assert '"${CORE_SERVICES[@]}" || ROLLBACK_RECREATE_EXIT=$?' in text


@pytest.mark.unit
def test_rollback_recreate_timeout_is_detected_and_typed() -> None:
    text = _text()
    assert 'ROLLBACK_RECREATE_EXIT}" -eq 124' in text, (
        "The rollback path must distinguish exit 124 (timeout) from an "
        "ordinary compose failure so the log/receipt can name it specifically."
    )
    assert "ROLLBACK_RECREATE_TIMEOUT" in text
    assert "ROLLBACK_RECREATE_TIMED_OUT=true" in text


@pytest.mark.unit
def test_reconciliation_loop_covers_every_core_service() -> None:
    """AC1: after the rollback recreate, every CORE_SERVICES container must
    be reconciled to running or explicitly torn down."""
    text = _text()
    match = re.search(
        r"for svc in \"\$\{CORE_SERVICES\[@\]\}\"; do\s*\n"
        r'\s*container="\$\{CORE_CONTAINERS\[\$\{svc\}\]\}"\s*\n'
        r"\s*if ! reconcile_container_running_state \"\$\{container\}\" \"\$\{svc\}\"; then\s*\n"
        r"\s*STRANDED_SERVICES\+=\(\"\$\{svc\}\"\)",
        text,
    )
    assert match is not None, (
        "Expected a reconciliation loop over CORE_SERVICES calling "
        "reconcile_container_running_state() and collecting failures into "
        "STRANDED_SERVICES."
    )


@pytest.mark.unit
def test_reconciliation_runs_before_post_rollback_health_gate() -> None:
    """Reconciliation must happen BEFORE the health-gate re-verify -- a
    container merely 'Created' is not a lane the health-gate can fairly
    judge, and that is exactly the state a timed-out/partial recreate leaves
    behind."""
    text = _text()
    reconcile_idx = text.find("Reconciling core-service container state post-rollback")
    gate2_idx = text.find("Re-verifying health after rollback")
    assert reconcile_idx != -1 and gate2_idx != -1
    assert reconcile_idx < gate2_idx


@pytest.mark.unit
def test_post_rollback_census_is_compared_against_pre_attempt() -> None:
    """AC2 (FIX B text): verify the post-rollback census matches the
    pre-attempt census before returning."""
    text = _text()
    assert "declare -A POST_CONTAINER_STATUS" in text
    assert (
        'POST_CONTAINER_STATUS["${svc}"]="$(container_status "${container}")"' in text
    )
    assert (
        'if [[ "${POST_CONTAINER_STATUS[${svc}]}" == "${PRE_CONTAINER_STATUS[${svc}]}" ]]; then'
        in text
    ), (
        "Post-rollback status must be compared per-service against the pre-attempt census."
    )


@pytest.mark.unit
def test_stranded_containers_produce_a_typed_distinguishable_result() -> None:
    """AC3: a caller must be able to act on this failure mode without manual
    docker start/docker ps forensics -- a distinct RESULT value, not a
    generic FAILED/FAILED_ROLLED_BACK reused for unrelated failures."""
    text = _text()
    assert "FAILED_ROLLBACK_STRANDED_CONTAINERS" in text
    match = re.search(
        r'if \[\[ "\$\{#STRANDED_SERVICES\[@\]\}" -gt 0 \]\]; then\s*\n'
        r'\s*RESULT="FAILED_ROLLBACK_STRANDED_CONTAINERS"',
        text,
    )
    assert match is not None, (
        "RESULT must be overridden to FAILED_ROLLBACK_STRANDED_CONTAINERS "
        "whenever any core service had to be torn down, regardless of what "
        "the health-gate concluded."
    )


@pytest.mark.unit
def test_typed_result_is_documented_in_the_exit_code_header() -> None:
    text = _text()
    header = text[: text.find("set -euo pipefail")]
    assert "FAILED_ROLLBACK_STRANDED_CONTAINERS" in header, (
        "The new typed result must be documented in the script's own exit-code "
        "header comment, not left implicit."
    )


@pytest.mark.unit
def test_receipt_includes_rollback_reconciliation_detail() -> None:
    """The receipt JSON (the durable evidence surface) must carry the
    reconciliation detail, not just the bare triggered/gate fields it had
    before -- a caller reading only the receipt must be able to tell whether
    anything was stranded/torn down without re-deriving it from logs."""
    text = _text()
    for expected in (
        "rollback_recreate_timed_out",
        "rollback_stranded_services",
        "rollback_census_pre",
        "rollback_census_post",
    ):
        assert expected in text, f"receipt must carry --argjson {expected} ..."
    assert "recreate_timed_out: $rollback_recreate_timed_out" in text
    assert "stranded_services: $rollback_stranded_services" in text
    assert "census: {pre: $rollback_census_pre, post: $rollback_census_post}" in text


@pytest.mark.unit
def test_receipt_fields_are_defined_on_every_path_not_just_rollback() -> None:
    """ROLLBACK_RECREATE_TIMED_OUT / STRANDED_SERVICES / POST_CONTAINER_STATUS
    must be initialized before the rollback branch so `set -u` cannot crash
    receipt assembly on the health-gate-PASS (no rollback) path."""
    text = _text()
    rollback_flag_idx = text.find("ROLLBACK_TRIGGERED=false")
    stranded_init_idx = text.find("STRANDED_SERVICES=()")
    receipt_idx = text.find("# 7. Emit receipt")
    assert rollback_flag_idx != -1 and stranded_init_idx != -1 and receipt_idx != -1
    assert rollback_flag_idx < stranded_init_idx < receipt_idx
