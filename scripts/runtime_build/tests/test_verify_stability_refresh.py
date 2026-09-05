# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/runtime_build/verify_stability_refresh.py [OMN-14873].

Mocked subprocess/HTTP -- no Docker daemon or live lane required. Covers the
pre-specified acceptance checks from the design: PASS/FAIL boundary at exactly
min_contracts, digest-unchanged -> FAIL, ancestry true/false branches, and the
rollback re-verification path (a second health-gate run against a
post-rollback snapshot).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1]
# The verifier imports its sibling ``health_payload`` module the way it does
# when executed as a script (``sys.path[0]`` is the script's own directory).
# ``spec_from_file_location`` does not set that up, so the test must.
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "verify_stability_refresh.py"
_spec = importlib.util.spec_from_file_location("verify_stability_refresh", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["verify_stability_refresh"] = _mod
_spec.loader.exec_module(_mod)

check_manifest_count = _mod.check_manifest_count
check_health = _mod.check_health
check_cluster_health = _mod.check_cluster_health
check_partition_headroom = _mod.check_partition_headroom
check_consumer_group = _mod.check_consumer_group
check_consumer_group_with_retry = _mod.check_consumer_group_with_retry
check_service_digest = _mod.check_service_digest
run_health_gate = _mod.run_health_gate
build_receipt = _mod.build_receipt
HealthGateReport = _mod.HealthGateReport
PartitionHeadroomCheck = _mod.PartitionHeadroomCheck
CORE_SERVICES = _mod.CORE_SERVICES
DEFAULT_MIN_CONTRACTS = _mod.DEFAULT_MIN_CONTRACTS
DEFAULT_PARTITION_WARN_THRESHOLD = _mod.DEFAULT_PARTITION_WARN_THRESHOLD


def _topic_list_output(partitions: list[int]) -> str:
    """Fixed-width `rpk topic list` output -- NAME / PARTITIONS / REPLICAS."""
    header = "NAME                  PARTITIONS   REPLICAS"
    rows = [
        f"topic-{i:04d}          {p}            1" for i, p in enumerate(partitions)
    ]
    return "\n".join([header, *rows]) + "\n"


# ─── helpers ─────────────────────────────────────────────────────────────────


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(body: dict | list, status: int = 200):
    payload = json.dumps(body).encode()

    def _open(url, timeout=10):
        return _FakeHTTPResponse(payload, status=status)

    return _open


# ─── manifest count: PASS/FAIL boundary exactly at min_contracts ───────────


def test_manifest_count_exactly_at_floor_passes():
    opener = _opener({"contracts": list(range(DEFAULT_MIN_CONTRACTS))})
    count, err = check_manifest_count(
        "http://x/manifest", DEFAULT_MIN_CONTRACTS, opener=opener
    )
    assert err is None
    assert count == DEFAULT_MIN_CONTRACTS


def test_manifest_count_one_below_floor_reported_as_not_ok():
    opener = _opener({"contracts": list(range(DEFAULT_MIN_CONTRACTS - 1))})
    count, err = check_manifest_count(
        "http://x/manifest", DEFAULT_MIN_CONTRACTS, opener=opener
    )
    assert err is None
    assert count == DEFAULT_MIN_CONTRACTS - 1
    # run_health_gate is what actually flips manifest_ok -- assert the boundary there.
    report = HealthGateReport(
        lane="stability-test", manifest_floor=DEFAULT_MIN_CONTRACTS
    )
    report.manifest_count = count
    report.manifest_ok = count is not None and count >= DEFAULT_MIN_CONTRACTS
    assert report.manifest_ok is False


def test_manifest_count_list_shape_supported():
    opener = _opener([{"name": "a"}, {"name": "b"}])
    count, err = check_manifest_count("http://x/manifest", 1, opener=opener)
    assert err is None
    assert count == 2


# ─── digest-unchanged -> FAIL ───────────────────────────────────────────────


def test_digest_unchanged_fails_digest_changed_check():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="sha256:same"),  # image id (unchanged)
            _completed(stdout="deadbeef1234"),  # revision label
        ]
    )
    result = check_service_digest(
        "omninode-runtime",
        "omninode-stability-test-runtime",
        pre_image_id="sha256:same",
        expected_revision="deadbeef1234",
        runner=runner,
    )
    assert result.digest_changed is False
    assert result.revision_match is True  # revision can match while digest is stale


def test_digest_changed_and_revision_match_passes():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="sha256:new"),
            _completed(stdout="deadbeef1234"),
        ]
    )
    result = check_service_digest(
        "omninode-runtime",
        "omninode-stability-test-runtime",
        pre_image_id="sha256:old",
        expected_revision="deadbeef1234",
        runner=runner,
    )
    assert result.digest_changed is True
    assert result.revision_match is True


def test_revision_mismatch_is_exists_but_wrong_not_silent_pass():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="sha256:new"),
            _completed(stdout="stalerevision00"),
        ]
    )
    result = check_service_digest(
        "omninode-runtime",
        "omninode-stability-test-runtime",
        pre_image_id="sha256:old",
        expected_revision="deadbeef1234",
        runner=runner,
    )
    assert result.digest_changed is True
    assert result.revision_match is False


# ─── health / cluster / consumer-group checks ──────────────────────────────


def test_health_ok_status_healthy():
    opener = _opener({"status": "healthy", "details": {"healthy": True}})
    verdict = check_health("http://x/health", opener=opener)
    assert verdict.ok is True
    assert verdict.status == "healthy"
    assert "healthy" in verdict.detail


def test_health_not_ok_status_degraded():
    opener = _opener({"status": "degraded", "details": {"healthy": False}})
    verdict = check_health("http://x/health", opener=opener)
    assert verdict.ok is False


# The full strict verdict table -- including the payloads that passed before
# OMN-17563 -- lives in test_health_payload_omn17563.py.


def test_cluster_health_ok():
    runner = MagicMock(return_value=_completed(stdout="Healthy:             true\n"))
    ok, _detail = check_cluster_health("redpanda-container", runner=runner)
    assert ok is True


def test_cluster_health_not_ok_nonzero_exit():
    runner = MagicMock(return_value=_completed(returncode=1, stderr="boom"))
    ok, _detail = check_cluster_health("redpanda-container", runner=runner)
    assert ok is False


# ─── partition headroom [OMN-14013] ─────────────────────────────────────────
#
# `rpk cluster health` never surfaces partition-allocation headroom -- these
# checks prove the NEW dedicated probe does, and that it distinguishes
# "healthy headroom" / "crossed the visibility warn threshold" / "at or over
# cap (the literal live incident: 7046/7000, `rpk cluster health` green
# throughout)" as three genuinely different states.


def test_partition_headroom_ok_well_below_threshold():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="15000\n"),  # cluster config get
            _completed(stdout=_topic_list_output([1] * 1529)),  # topic list
        ]
    )
    result = check_partition_headroom("redpanda-container", runner=runner)
    assert result.cap == 15000
    assert result.total_partitions == 1529
    assert result.at_or_over_cap is False
    assert result.crossed_warn_threshold is False
    assert result.error is None


def test_partition_headroom_crosses_warn_threshold_but_not_at_cap():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="8000\n"),
            _completed(stdout=_topic_list_output([1] * 7047)),
        ]
    )
    result = check_partition_headroom(
        "redpanda-container", runner=runner, warn_threshold=0.8
    )
    assert result.usage_ratio is not None
    assert result.usage_ratio > 0.8
    assert result.crossed_warn_threshold is True
    assert result.at_or_over_cap is False


def test_partition_headroom_at_or_over_cap_is_a_real_failure():
    """The literal OMN-14013 live incident: 7046/7000, `rpk cluster health`
    reported Healthy: true throughout. This check must call it a failure."""
    runner = MagicMock(
        side_effect=[
            _completed(stdout="7000\n"),
            _completed(stdout=_topic_list_output([1] * 7046)),
        ]
    )
    result = check_partition_headroom("redpanda-container", runner=runner)
    assert result.at_or_over_cap is True
    assert "AT OR OVER CAP" in result.detail


def test_partition_headroom_exactly_at_cap_is_at_or_over():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="7000\n"),
            _completed(stdout=_topic_list_output([1] * 7000)),
        ]
    )
    result = check_partition_headroom("redpanda-container", runner=runner)
    assert result.at_or_over_cap is True


def test_partition_headroom_cap_probe_failure_is_not_silently_ok():
    runner = MagicMock(
        side_effect=[
            _completed(returncode=1, stderr="no such config"),
            _completed(stdout=_topic_list_output([1] * 100)),
        ]
    )
    result = check_partition_headroom("redpanda-container", runner=runner)
    assert result.cap is None
    assert result.at_or_over_cap is False  # unknown, not silently healthy
    assert result.error is not None


def test_partition_headroom_topic_list_probe_failure_is_not_silently_ok():
    runner = MagicMock(
        side_effect=[
            _completed(stdout="15000\n"),
            _completed(returncode=1, stderr="connection refused"),
        ]
    )
    result = check_partition_headroom("redpanda-container", runner=runner)
    assert result.total_partitions is None
    assert result.error is not None


def test_partition_headroom_parses_partitions_column_by_header_not_position():
    """The PARTITIONS column must be located by its header label, not a
    hardcoded index -- a reordered rpk table (NAME / REPLICAS / PARTITIONS)
    must still sum the right column."""
    reordered = "NAME       REPLICAS   PARTITIONS\ntopic-a    1          3\ntopic-b    1          4\n"
    runner = MagicMock(
        side_effect=[
            _completed(stdout="100\n"),
            _completed(stdout=reordered),
        ]
    )
    result = check_partition_headroom("redpanda-container", runner=runner)
    assert result.total_partitions == 7


def _group_describe_output(state: str) -> str:
    """rpk group describe has NO -f json mode -- fixed-width plain text."""
    return (
        "GROUP        some.group\n"
        "COORDINATOR  0\n"
        f"STATE        {state}\n"
        "BALANCER     \n"
        "MEMBERS      1\n"
        "TOTAL-LAG    0\n"
    )


def test_consumer_group_stable():
    runner = MagicMock(return_value=_completed(stdout=_group_describe_output("Stable")))
    result = check_consumer_group("redpanda-container", "some.group", runner=runner)
    assert result.stable is True
    assert result.state == "Stable"


def test_consumer_group_empty_is_healthy_demand_driven_idle():
    """Empty = registered with the broker, zero current members -- the normal
    idle state for a demand-driven consumer, not a wiring failure."""
    runner = MagicMock(return_value=_completed(stdout=_group_describe_output("Empty")))
    result = check_consumer_group("redpanda-container", "some.group", runner=runner)
    assert result.stable is True
    assert result.state == "Empty"


def test_consumer_group_dead_is_not_stable():
    """Dead is the real 'silent wiring death' signal -- group coordinator does
    not know this group at all."""
    runner = MagicMock(return_value=_completed(stdout=_group_describe_output("Dead")))
    result = check_consumer_group("redpanda-container", "some.group", runner=runner)
    assert result.stable is False
    assert result.state == "Dead"


def test_consumer_group_describe_error_is_not_a_silent_stable():
    runner = MagicMock(return_value=_completed(returncode=1, stderr="no such group"))
    result = check_consumer_group("redpanda-container", "some.group", runner=runner)
    assert result.stable is False
    assert result.error is not None


def test_consumer_group_no_state_line_is_not_a_silent_stable():
    runner = MagicMock(return_value=_completed(stdout="GROUP  some.group\n"))
    result = check_consumer_group("redpanda-container", "some.group", runner=runner)
    assert result.stable is False
    assert result.error is not None


def test_consumer_group_retry_recovers_after_transient_empty_then_dead():
    """Right after a force-recreate the group can show a transient bad state
    before the consumer rejoins; the bounded retry should recover on a later
    attempt without sleeping in real time (sleep_fn stubbed)."""
    runner = MagicMock(
        side_effect=[
            _completed(stdout=_group_describe_output("Dead")),
            _completed(stdout=_group_describe_output("Dead")),
            _completed(stdout=_group_describe_output("Stable")),
        ]
    )
    sleeps: list[float] = []
    result = check_consumer_group_with_retry(
        "redpanda-container",
        "some.group",
        runner=runner,
        attempts=5,
        interval_seconds=0.01,
        sleep_fn=sleeps.append,
    )
    assert result.stable is True
    assert result.state == "Stable"
    assert len(sleeps) == 2  # slept between attempts 1->2 and 2->3, not after success


def test_consumer_group_retry_exhausts_and_reports_last_state():
    runner = MagicMock(return_value=_completed(stdout=_group_describe_output("Dead")))
    result = check_consumer_group_with_retry(
        "redpanda-container",
        "some.group",
        runner=runner,
        attempts=3,
        interval_seconds=0.01,
        sleep_fn=lambda _s: None,
    )
    assert result.stable is False
    assert result.state == "Dead"


# ─── full health-gate orchestration: PASS / FAIL end to end ────────────────


def _full_pass_runner():
    """A subprocess runner producing all-passing docker/rpk output, keyed by
    the 4 core services (2 docker inspect calls each) + 1 cluster health call
    + N consumer-group describe calls."""
    calls: list[list[str]] = []

    def _run(cmd, capture_output=True, text=True, timeout=30, check=False):
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            fmt = cmd[-1]
            if "Image" in fmt:
                return _completed(stdout="sha256:new-image")
            return _completed(stdout="newrevision1234")
        if "config" in cmd and "get" in cmd:
            return _completed(stdout="15000\n")
        if "topic" in cmd and "list" in cmd:
            return _completed(stdout=_topic_list_output([1] * 1529))
        if cmd[:3] == ["docker", "exec", "redpanda-container"] and "cluster" in cmd:
            return _completed(stdout="Healthy:                          true\n")
        if "group" in cmd and "describe" in cmd:
            return _completed(stdout=_group_describe_output("Stable"))
        raise AssertionError(f"unexpected command: {cmd}")

    return _run


def test_health_gate_overall_pass():
    pre_image_ids = dict.fromkeys(CORE_SERVICES, "sha256:old-image")
    opener = _opener({"contracts": list(range(DEFAULT_MIN_CONTRACTS))})
    # OMN-17624: the gate requires a monitor verdict, so a provably-healthy
    # lane must carry one. Without it the gate correctly refuses.
    health_opener = _opener(
        {
            "status": "healthy",
            "details": {"runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}},
        }
    )

    # run_health_gate calls check_health then check_manifest_count with the
    # SAME opener param; use one opener that serves both shapes based on url.
    def combo_opener(url, timeout=10):
        if "manifest" in url:
            return opener(url, timeout=timeout)
        return health_opener(url, timeout=timeout)

    report = run_health_gate(
        lane="stability-test",
        pre_image_ids=pre_image_ids,
        expected_revision="newrevision1234",
        manifest_url="http://x/manifest",
        health_url="http://x/health",
        broker_container="redpanda-container",
        min_contracts=DEFAULT_MIN_CONTRACTS,
        consumer_groups=["group.a", "group.b"],
        runner=_full_pass_runner(),
        opener=combo_opener,
        sleep_fn=lambda _s: None,
    )
    assert report.overall == "PASS"


def test_health_gate_overall_fail_when_a_group_is_dead():
    pre_image_ids = dict.fromkeys(CORE_SERVICES, "sha256:old-image")

    def runner(cmd, capture_output=True, text=True, timeout=30, check=False):
        if cmd[:2] == ["docker", "inspect"]:
            fmt = cmd[-1]
            if "Image" in fmt:
                return _completed(stdout="sha256:new-image")
            return _completed(stdout="newrevision1234")
        if "config" in cmd and "get" in cmd:
            return _completed(stdout="15000\n")
        if "topic" in cmd and "list" in cmd:
            return _completed(stdout=_topic_list_output([1] * 1529))
        if "cluster" in cmd:
            return _completed(stdout="Healthy:                          true\n")
        if "group" in cmd and "describe" in cmd:
            # Dead, not Empty -- Empty is now a healthy demand-driven-idle
            # state; Dead is the genuine "group unknown to broker" failure.
            return _completed(stdout=_group_describe_output("Dead"))
        raise AssertionError(f"unexpected command: {cmd}")

    def opener(url, timeout=10):
        if "manifest" in url:
            return _FakeHTTPResponse(
                json.dumps({"contracts": list(range(DEFAULT_MIN_CONTRACTS))}).encode()
            )
        # OMN-17624: an "otherwise-healthy refresh" must carry the monitor
        # verdict the gate now requires; the point of this test is partition
        # headroom, not the verdict policy.
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "status": "healthy",
                    "details": {
                        "runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}
                    },
                }
            ).encode()
        )

    report = run_health_gate(
        lane="stability-test",
        pre_image_ids=pre_image_ids,
        expected_revision="newrevision1234",
        manifest_url="http://x/manifest",
        health_url="http://x/health",
        broker_container="redpanda-container",
        min_contracts=DEFAULT_MIN_CONTRACTS,
        consumer_groups=["group.a"],
        runner=runner,
        opener=opener,
        sleep_fn=lambda _s: None,
    )
    assert report.overall == "FAIL"
    assert report.groups_stable is False


def test_health_gate_overall_fail_when_partition_cap_reached():
    """OMN-14013: everything else passes, but the broker is AT its partition
    cap -- overall must be FAIL (a real, checked defect `rpk cluster health`
    alone would have missed)."""
    pre_image_ids = dict.fromkeys(CORE_SERVICES, "sha256:old-image")

    def runner(cmd, capture_output=True, text=True, timeout=30, check=False):
        if cmd[:2] == ["docker", "inspect"]:
            fmt = cmd[-1]
            if "Image" in fmt:
                return _completed(stdout="sha256:new-image")
            return _completed(stdout="newrevision1234")
        if "config" in cmd and "get" in cmd:
            return _completed(stdout="7000\n")
        if "topic" in cmd and "list" in cmd:
            return _completed(stdout=_topic_list_output([1] * 7046))
        if "cluster" in cmd:
            return _completed(stdout="Healthy:                          true\n")
        if "group" in cmd and "describe" in cmd:
            return _completed(stdout=_group_describe_output("Stable"))
        raise AssertionError(f"unexpected command: {cmd}")

    def opener(url, timeout=10):
        if "manifest" in url:
            return _FakeHTTPResponse(
                json.dumps({"contracts": list(range(DEFAULT_MIN_CONTRACTS))}).encode()
            )
        # OMN-17624: an "otherwise-healthy refresh" must carry the monitor
        # verdict the gate now requires; the point of this test is partition
        # headroom, not the verdict policy.
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "status": "healthy",
                    "details": {
                        "runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}
                    },
                }
            ).encode()
        )

    report = run_health_gate(
        lane="stability-test",
        pre_image_ids=pre_image_ids,
        expected_revision="newrevision1234",
        manifest_url="http://x/manifest",
        health_url="http://x/health",
        broker_container="redpanda-container",
        min_contracts=DEFAULT_MIN_CONTRACTS,
        consumer_groups=["group.a"],
        runner=runner,
        opener=opener,
        sleep_fn=lambda _s: None,
    )
    assert report.overall == "FAIL"
    assert report.partition_headroom is not None
    assert report.partition_headroom.at_or_over_cap is True
    assert report.partition_headroom_ok is False


def test_health_gate_overall_pass_when_partition_headroom_only_crosses_warn():
    """OMN-14013: crossing the warn threshold (but not the cap) is visibility
    only -- it must NOT retroactively fail an otherwise-healthy refresh."""
    pre_image_ids = dict.fromkeys(CORE_SERVICES, "sha256:old-image")

    def runner(cmd, capture_output=True, text=True, timeout=30, check=False):
        if cmd[:2] == ["docker", "inspect"]:
            fmt = cmd[-1]
            if "Image" in fmt:
                return _completed(stdout="sha256:new-image")
            return _completed(stdout="newrevision1234")
        if "config" in cmd and "get" in cmd:
            return _completed(stdout="8000\n")
        if "topic" in cmd and "list" in cmd:
            return _completed(stdout=_topic_list_output([1] * 7047))  # ~88%
        if "cluster" in cmd:
            return _completed(stdout="Healthy:                          true\n")
        if "group" in cmd and "describe" in cmd:
            return _completed(stdout=_group_describe_output("Stable"))
        raise AssertionError(f"unexpected command: {cmd}")

    def opener(url, timeout=10):
        if "manifest" in url:
            return _FakeHTTPResponse(
                json.dumps({"contracts": list(range(DEFAULT_MIN_CONTRACTS))}).encode()
            )
        # OMN-17624: an "otherwise-healthy refresh" must carry the monitor
        # verdict the gate now requires; the point of this test is partition
        # headroom, not the verdict policy.
        return _FakeHTTPResponse(
            json.dumps(
                {
                    "status": "healthy",
                    "details": {
                        "runtime_health": {"status": "HEALTHY", "age_seconds": 1.0}
                    },
                }
            ).encode()
        )

    report = run_health_gate(
        lane="stability-test",
        pre_image_ids=pre_image_ids,
        expected_revision="newrevision1234",
        manifest_url="http://x/manifest",
        health_url="http://x/health",
        broker_container="redpanda-container",
        min_contracts=DEFAULT_MIN_CONTRACTS,
        consumer_groups=["group.a"],
        runner=runner,
        opener=opener,
        sleep_fn=lambda _s: None,
    )
    assert report.overall == "PASS"
    assert report.partition_headroom is not None
    assert report.partition_headroom.crossed_warn_threshold is True
    assert report.partition_headroom.at_or_over_cap is False


# ─── receipt: ancestry true/false + rollback re-verification ───────────────


def test_receipt_success_when_gate_passes():
    gate = HealthGateReport(lane="stability-test")
    gate.manifest_ok = True
    gate.health_ok = True
    gate.cluster_healthy = True
    gate.services = []
    gate.consumer_groups = []
    # Force overall PASS by monkeypatching the properties via a minimal report
    # that actually satisfies overall == PASS requires non-empty services/groups
    # with all-true; build one directly for this assertion instead.
    passing = HealthGateReport(lane="stability-test", manifest_floor=1)
    passing.manifest_count = 1
    passing.manifest_ok = True
    passing.health_ok = True
    passing.cluster_healthy = True
    passing.services = [
        _mod.ServiceDigestCheck(
            service="omninode-runtime",
            container="c",
            pre_image_id="a",
            post_image_id="b",
            digest_changed=True,
            revision_label="r",
            expected_revision="r",
            revision_match=True,
        )
    ]
    passing.consumer_groups = [
        _mod.ConsumerGroupCheck(group="g", state="Stable", stable=True)
    ]
    assert passing.overall == "PASS"

    receipt = build_receipt(
        lane="stability-test",
        prior_refs={"omnibase_infra": "aaa"},
        new_refs={"omnibase_infra": "bbb"},
        ancestry_ok=True,
        ancestry_commands=["git merge-base --is-ancestor aaa bbb"],
        build_scope=["omninode-runtime"],
        gate=passing,
        rollback_triggered=False,
        rollback_gate=None,
    )
    assert receipt["result"] == "SUCCESS"
    assert receipt["ancestry_proof"]["merge_base_is_ancestor"] is True


def test_receipt_ancestry_false_branch_recorded_but_does_not_crash():
    failing = HealthGateReport(lane="stability-test")
    receipt = build_receipt(
        lane="stability-test",
        prior_refs={"omnibase_infra": "bbb"},
        new_refs={"omnibase_infra": "aaa"},
        ancestry_ok=False,
        ancestry_commands=["git merge-base --is-ancestor bbb aaa"],
        build_scope=["omninode-runtime"],
        gate=failing,
        rollback_triggered=False,
        rollback_gate=None,
    )
    assert receipt["ancestry_proof"]["merge_base_is_ancestor"] is False
    assert receipt["result"] == "FAILED"


def test_receipt_rollback_reverified_success():
    """Rollback path: gate FAILs, rollback triggers, re-verify PASSes."""
    failing_gate = HealthGateReport(lane="stability-test")
    passing_rollback_gate = HealthGateReport(lane="stability-test", manifest_floor=1)
    passing_rollback_gate.manifest_count = 1
    passing_rollback_gate.manifest_ok = True
    passing_rollback_gate.health_ok = True
    passing_rollback_gate.cluster_healthy = True
    passing_rollback_gate.services = [
        _mod.ServiceDigestCheck(
            service="omninode-runtime",
            container="c",
            pre_image_id="a",
            post_image_id="a",  # rolled back to the SAME image as before refresh
            digest_changed=True,  # relative to the FAILED new image, not pre-refresh
            revision_label="old-rev",
            expected_revision="old-rev",
            revision_match=True,
        )
    ]
    passing_rollback_gate.consumer_groups = [
        _mod.ConsumerGroupCheck(group="g", state="Stable", stable=True)
    ]
    assert passing_rollback_gate.overall == "PASS"

    receipt = build_receipt(
        lane="stability-test",
        prior_refs={"omnibase_infra": "aaa"},
        new_refs={"omnibase_infra": "bbb"},
        ancestry_ok=True,
        ancestry_commands=["git merge-base --is-ancestor aaa bbb"],
        build_scope=["omninode-runtime"],
        gate=failing_gate,
        rollback_triggered=True,
        rollback_gate=passing_rollback_gate,
    )
    assert receipt["result"] == "FAILED_ROLLED_BACK"
    assert receipt["rollback"]["triggered"] is True
    assert receipt["rollback"]["gate"]["overall"] == "PASS"


def test_receipt_rollback_still_unhealthy_is_failed_not_masked():
    """If rollback re-verification ALSO fails, the receipt must say FAILED,
    never claim success -- this is the STOP-and-report condition."""
    failing_gate = HealthGateReport(lane="stability-test")
    still_failing_rollback_gate = HealthGateReport(lane="stability-test")

    receipt = build_receipt(
        lane="stability-test",
        prior_refs={"omnibase_infra": "aaa"},
        new_refs={"omnibase_infra": "bbb"},
        ancestry_ok=True,
        ancestry_commands=["git merge-base --is-ancestor aaa bbb"],
        build_scope=["omninode-runtime"],
        gate=failing_gate,
        rollback_triggered=True,
        rollback_gate=still_failing_rollback_gate,
    )
    assert receipt["result"] == "FAILED"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
