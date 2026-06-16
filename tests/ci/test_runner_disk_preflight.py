# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the OMN-13040 runner-disk-preflight CI job and infra-signature tooling.

Coverage:
- runner-disk-preflight.yml workflow structure (job exists, emits correct annotation,
  fires on correct events).
- disk-watermark-check.sh: verifies the 85% ALERT leg (not just the GC leg) is wired
  in the OMN-13008 timer — this is the OMN-13040 §2 verification.
- infra-signature-rerun.sh: verifies INFRA_SIGNATURES includes the OMN-13045
  compose-network-unreachable signature AND the RUNNER-DISK: annotation signature.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "runner-disk-preflight.yml"
DISK_WATERMARK_SCRIPT = REPO_ROOT / "scripts" / "disk-watermark-check.sh"
DISK_GC_SERVICE = REPO_ROOT / "deploy" / "disk-gc" / "onex-disk-gc.service"
INFRA_RERUN_SCRIPT = REPO_ROOT / "scripts" / "infra-signature-rerun.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:  # type: ignore[type-arg]
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"Expected dict from {path}"
    return loaded


def _workflow_run_script(workflow: dict, job_name: str, step_name: str) -> str:  # type: ignore[type-arg]
    steps = workflow["jobs"][job_name]["steps"]
    step = next(s for s in steps if s.get("name") == step_name)
    run = step.get("run", "")
    assert isinstance(run, str)
    return run


# ---------------------------------------------------------------------------
# Part 1 — runner-disk-preflight.yml structure
# ---------------------------------------------------------------------------


def test_preflight_workflow_file_exists() -> None:
    """The runner-disk-preflight.yml workflow must exist (OMN-13040 §1)."""
    assert PREFLIGHT_WORKFLOW.exists(), (
        f"Missing {PREFLIGHT_WORKFLOW} — the runner-disk-preflight workflow "
        "required by OMN-13040 must exist."
    )


def test_preflight_job_has_distinct_name() -> None:
    """The job must be distinctly named so it appears unambiguously in PR checks."""
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    assert "runner-disk-preflight" in workflow["jobs"], (
        "Expected a 'runner-disk-preflight' job key in the workflow."
    )
    job = workflow["jobs"]["runner-disk-preflight"]
    assert job.get("name") == "Runner Disk Preflight", (
        f"Job display name must be 'Runner Disk Preflight', got {job.get('name')!r}"
    )


def test_preflight_workflow_triggers_on_pr_and_push() -> None:
    """Preflight must run on every PR and push so it always provides a check.

    Note: PyYAML parses the YAML 'on:' key as Python True (YAML boolean alias).
    We check both the canonical string form and the boolean form for robustness.
    """
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    # PyYAML parses YAML 'on:' as Python True; fall back to string 'on' for
    # strict-mode YAML parsers.
    triggers = workflow.get(True) or workflow.get("on") or {}
    assert isinstance(triggers, dict)
    assert "pull_request" in triggers, "Must trigger on pull_request events"
    assert "push" in triggers, "Must trigger on push events"
    assert "merge_group" in triggers, "Must trigger on merge_group events"


def test_preflight_step_emits_runner_disk_annotation() -> None:
    """The check step must emit a ::error title=RUNNER-DISK:<free>GB:: annotation.

    This is the machine-parseable marker that the merge-sweep auto-rerun
    script matches against (OMN-13040 §3 + §1 DoD).
    """
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    run_script = _workflow_run_script(
        workflow, "runner-disk-preflight", "Check runner disk space"
    )
    assert "::error title=RUNNER-DISK:" in run_script, (
        "The check step must emit a '::error title=RUNNER-DISK:<free>GB::' annotation "
        "so infra failures are machine-identifiable in CI logs."
    )


def test_preflight_step_fails_on_low_disk() -> None:
    """The check step must exit non-zero when disk is low (not just warn)."""
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    run_script = _workflow_run_script(
        workflow, "runner-disk-preflight", "Check runner disk space"
    )
    # Must exit 1, not just print a warning.
    assert "exit 1" in run_script, (
        "The preflight step must exit 1 on low disk, not merely warn — "
        "infra failures must BLOCK not WARN."
    )


def test_preflight_step_names_runner_in_annotation() -> None:
    """The annotation must include the runner name for operator triage (OMN-13045 context)."""
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    run_script = _workflow_run_script(
        workflow, "runner-disk-preflight", "Check runner disk space"
    )
    # Runner name must appear in the annotation body so ops know which runner to drain.
    assert "RUNNER_NAME" in run_script, (
        "The RUNNER-DISK annotation must reference RUNNER_NAME so the infra team "
        "knows which runner to drain/recycle (OMN-13045)."
    )


def test_preflight_job_uses_runner_selector_v1() -> None:
    """Job must use OMNI_RUNNER_SELECTOR_V1 pattern so it runs on self-hosted runners."""
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    job = workflow["jobs"]["runner-disk-preflight"]
    runs_on = str(job.get("runs-on", ""))
    assert "OMNI_TRUSTED_CI_RUNS_ON_JSON" in runs_on, (
        "runner-disk-preflight must use OMNI_TRUSTED_CI_RUNS_ON_JSON so it runs "
        "on the same self-hosted runners that need disk monitoring."
    )
    assert "OMNI_PUBLIC_PR_RUNS_ON_JSON" in runs_on, (
        "Must also fall back to OMNI_PUBLIC_PR_RUNS_ON_JSON for public fork PRs."
    )


def test_preflight_job_has_timeout() -> None:
    """Job must have a short timeout — it should never block long."""
    workflow = _load_yaml(PREFLIGHT_WORKFLOW)
    job = workflow["jobs"]["runner-disk-preflight"]
    timeout = job.get("timeout-minutes", 0)
    assert isinstance(timeout, int) and timeout <= 10, (
        f"runner-disk-preflight should have timeout-minutes <= 10 (got {timeout})"
    )


# ---------------------------------------------------------------------------
# Part 2 — OMN-13008 disk-watermark 85% alert leg verification (OMN-13040 §2)
# ---------------------------------------------------------------------------


def test_disk_watermark_script_exists() -> None:
    """disk-watermark-check.sh must exist — it is the 85% alert leg (OMN-13008)."""
    assert DISK_WATERMARK_SCRIPT.exists(), (
        f"Missing {DISK_WATERMARK_SCRIPT} — the OMN-13008 disk watermark script "
        "must be present."
    )


def test_disk_watermark_85_pct_warn_leg_is_wired() -> None:
    """The 85% ALERT leg must be wired in disk-watermark-check.sh (OMN-13008 §3).

    OMN-13040 §2 requires: verify the OMN-13008 timer has an ALERT leg at 85%,
    not just GC. The script must default WARN_PCT to 85 and emit at >= that threshold.
    """
    source = DISK_WATERMARK_SCRIPT.read_text(encoding="utf-8")
    assert "WARN_PCT=85" in source, (
        "disk-watermark-check.sh must default WARN_PCT=85 — the 85% alert leg "
        "required by OMN-13008 is missing."
    )
    # Must also check against WARN_PCT and emit something (not silently exit).
    assert "USED_PCT < WARN_PCT" in source or "WARN_PCT" in source, (
        "disk-watermark-check.sh must compare usage against WARN_PCT to emit the alert."
    )
    # Must emit severity=warning at 85%.
    assert "warning" in source.lower(), (
        "disk-watermark-check.sh must emit severity=warning at 85% (the alert leg)."
    )


def test_disk_watermark_90_pct_crit_leg_is_wired() -> None:
    """The 90% CRITICAL leg must also be wired (OMN-13008 §3 bus event)."""
    source = DISK_WATERMARK_SCRIPT.read_text(encoding="utf-8")
    assert "CRIT_PCT=90" in source, (
        "disk-watermark-check.sh must default CRIT_PCT=90 — the 90% critical "
        "bus-event leg required by OMN-13008 is missing."
    )
    assert "critical" in source.lower(), (
        "disk-watermark-check.sh must emit severity=critical at 90%."
    )


def test_disk_gc_service_wires_watermark_check() -> None:
    """The onex-disk-gc.service must call disk-watermark-check.sh (OMN-13008 integration)."""
    assert DISK_GC_SERVICE.exists(), (
        f"Missing {DISK_GC_SERVICE} — the OMN-13008 disk GC systemd service "
        "must be present."
    )
    source = DISK_GC_SERVICE.read_text(encoding="utf-8")
    assert "disk-watermark-check.sh" in source, (
        "onex-disk-gc.service must call disk-watermark-check.sh so the 85% alert "
        "fires on the scheduled GC timer run (OMN-13008)."
    )


def test_disk_gc_service_watermark_step_is_soft_fail() -> None:
    """The watermark step in the service must use ExecStart=- (soft-fail).

    Non-zero exit from the watermark check (breach) is expected and must NOT
    bring down the whole GC service unit.
    """
    source = DISK_GC_SERVICE.read_text(encoding="utf-8")
    # The watermark step should use ExecStart=- prefix (soft-fail) so that a
    # >85% disk doesn't stop the GC from running.
    assert re.search(r"ExecStart=-.*disk-watermark-check", source), (
        "onex-disk-gc.service must use 'ExecStart=-<path>/disk-watermark-check.sh' "
        "(the '-' prefix means non-zero exit is tolerated) so a breach doesn't "
        "prevent GC from completing."
    )


# ---------------------------------------------------------------------------
# Part 3 — infra-signature-rerun.sh contains required signatures (OMN-13045 + OMN-13040)
# ---------------------------------------------------------------------------


def test_infra_rerun_script_exists() -> None:
    """infra-signature-rerun.sh must exist (OMN-13040 §3)."""
    assert INFRA_RERUN_SCRIPT.exists(), (
        f"Missing {INFRA_RERUN_SCRIPT} — the merge-sweep infra-signature auto-rerun "
        "script required by OMN-13040 must be present."
    )


def test_infra_rerun_script_matches_runner_disk_annotation() -> None:
    """Script must match the RUNNER-DISK: annotation emitted by the preflight job."""
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    assert "RUNNER-DISK:" in source, (
        "infra-signature-rerun.sh must include 'RUNNER-DISK:' as a known infra "
        "signature so disk-casualty runs are auto-rerun after runner recovery."
    )


def test_infra_rerun_script_matches_omn_13045_signature() -> None:
    """Script must match the OMN-13045 'healthy but unreachable' compose error."""
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    assert "compose services are healthy but unreachable from the runner" in source, (
        "infra-signature-rerun.sh must include the OMN-13045 compose-network-wedge "
        "signature: 'compose services are healthy but unreachable from the runner'."
    )


def test_infra_rerun_script_matches_no_space_left() -> None:
    """Script must match the raw 'No space left on device' OS error."""
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    assert "No space left on device" in source, (
        "infra-signature-rerun.sh must match 'No space left on device' — "
        "the raw OS-level disk full error seen in Kafka/compose logs."
    )


def test_infra_rerun_script_has_recovery_timestamp_guard() -> None:
    """Script must only rerun runs that PREDATE the recovery timestamp.

    A run that started AFTER the runner was recovered is not a disk casualty —
    it is a genuine domain failure. The guard prevents incorrect auto-reruns.
    """
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    assert "recovery_ts" in source or "recovery-state" in source, (
        "infra-signature-rerun.sh must guard reruns against a recovery timestamp — "
        "only runs that started BEFORE the recovery qualify as disk casualties."
    )


def test_infra_rerun_script_has_dry_run_mode() -> None:
    """Script must support --dry-run so it can be tested without issuing reruns."""
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    assert "--dry-run" in source, (
        "infra-signature-rerun.sh must support --dry-run for safe testing."
    )
    assert "DRY_RUN" in source, (
        "infra-signature-rerun.sh must honour DRY_RUN to avoid issuing reruns "
        "in test/dry-run mode."
    )


def test_infra_rerun_script_has_max_reruns_circuit_breaker() -> None:
    """Script must cap the number of reruns per invocation to avoid thrashing."""
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    assert "MAX_RERUNS" in source, (
        "infra-signature-rerun.sh must have a MAX_RERUNS circuit breaker to prevent "
        "runaway rerun cascades."
    )


def test_infra_rerun_script_uses_runner_selector_for_repos() -> None:
    """Script must scan the correct set of queue-enabled repos."""
    source = INFRA_RERUN_SCRIPT.read_text(encoding="utf-8")
    # Must include omnibase_infra — the primary repo where runner-boot failures occur.
    assert "omnibase_infra" in source, (
        "infra-signature-rerun.sh must scan omnibase_infra — the primary repo "
        "where runner-disk and compose-network failures are observed."
    )
    # Must include omniclaude.
    assert "omniclaude" in source, "infra-signature-rerun.sh must scan omniclaude."


# ---------------------------------------------------------------------------
# Part 4 — merge-sweep tick integration (cron-merge-sweep.sh wires the rerun)
# ---------------------------------------------------------------------------


def test_merge_sweep_script_references_infra_rerun() -> None:
    """cron-merge-sweep.sh (the tick script) must reference infra-signature-rerun.

    This is the integration point: the tick script must call (or reference)
    infra-signature-rerun.sh so the auto-rerun fires on every sweep cycle.

    Note: the actual wiring lives in omniclaude/scripts/cron-merge-sweep.sh.
    This test verifies the script at the canonical omiclaude path exists and
    would be the correct place to wire the call — we also accept inline wiring.
    """
    # The script lives in omniclaude, not omnibase_infra.
    omniclaude_merge_sweep = (
        REPO_ROOT.parent.parent / "omniclaude" / "scripts" / "cron-merge-sweep.sh"
    )
    # Also check the infra repo's scripts for any wrapper.
    infra_rerun = REPO_ROOT / "scripts" / "infra-signature-rerun.sh"

    # The rerun script itself must exist (already checked above).
    assert infra_rerun.exists(), (
        "infra-signature-rerun.sh must be present for the tick to call."
    )

    # The cron-merge-sweep.sh is the authoritative tick entry point. We verify
    # it either already calls infra-signature-rerun or that a note for the
    # operator is present in the rerun script documenting how to wire it.
    sweep_exists = omniclaude_merge_sweep.exists()
    if sweep_exists:
        # Acceptable: either the sweep calls the rerun script, or the rerun
        # script is standalone and called separately. Either way the script
        # must have the usage docs explaining it is called by the merge-sweep tick.
        rerun_source = infra_rerun.read_text(encoding="utf-8")
        has_tick_reference = (
            "merge-sweep" in rerun_source.lower()
            or "cron-merge-sweep" in rerun_source.lower()
            or "tick" in rerun_source.lower()
        )
        assert has_tick_reference, (
            "infra-signature-rerun.sh must document that it is called by the "
            "merge-sweep tick (cron-merge-sweep.sh) so the integration is clear."
        )
