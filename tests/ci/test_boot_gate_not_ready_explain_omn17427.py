# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427 — the boot wait's timeout says WHY a Deployment is not ready.

Delivery run 39668407696 waited its full 1800s and ended with
``omninode-runtime 0/1 NOT READY`` plus two projection writers 0/1, and its step
log said nothing more. The pod status, events and logs were only in the
uploaded artifact, so every red run was re-diagnosed by hand. These tests pin
the explanation ``boot_gate.py wait`` now prints on a timeout, against a fake
kubectl; no cluster is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import boot_gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DELIVER = REPO_ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"
NS = "lane-a"

RUNTIME_DEPLOYMENT = {
    "metadata": {"name": "omninode-runtime"},
    "spec": {"selector": {"matchLabels": {"app": "omninode-runtime"}}},
    "status": {
        "conditions": [
            {
                "type": "Available",
                "status": "False",
                "reason": "MinimumReplicasUnavailable",
                "message": "Deployment does not have minimum availability.",
            }
        ]
    },
}

CRASHING_POD = {
    "metadata": {
        "name": "omninode-runtime-7d9-abcde",
        "ownerReferences": [{"kind": "ReplicaSet", "name": "omninode-runtime-7d9"}],
    },
    "status": {
        "phase": "Running",
        "conditions": [
            {"type": "Ready", "status": "False", "reason": "ContainersNotReady"}
        ],
        "containerStatuses": [
            {
                "name": "runtime",
                "ready": False,
                "restartCount": 14,
                "state": {
                    "waiting": {
                        "reason": "CrashLoopBackOff",
                        "message": "back-off 5m0s restarting failed container",
                    }
                },
                "lastState": {
                    "terminated": {
                        "reason": "Error",
                        "exitCode": 1,
                        "finishedAt": "2026-10-21T21:01:00Z",
                    }
                },
            }
        ],
    },
}

EVENTS = {
    "items": [
        {
            "involvedObject": {"kind": "Pod", "name": "omninode-runtime-7d9-abcde"},
            "type": "Warning",
            "reason": "BackOff",
            "message": "Back-off restarting failed container runtime",
            "count": 40,
            "lastTimestamp": "2026-10-21T21:01:30Z",
        },
        {
            # A sibling Deployment whose name has omninode-runtime as a PREFIX.
            # Matching by prefix would report its events as the runtime's.
            "involvedObject": {
                "kind": "Pod",
                "name": "omninode-runtime-effects-5f6-zzzzz",
            },
            "type": "Normal",
            "reason": "Started",
            "message": "sibling-only-event",
            "lastTimestamp": "2026-10-21T20:40:00Z",
        },
        {
            "involvedObject": {"kind": "ReplicaSet", "name": "omninode-runtime-7d9"},
            "type": "Normal",
            "reason": "SuccessfulCreate",
            "message": "Created pod: omninode-runtime-7d9-abcde",
            "lastTimestamp": "2026-10-21T20:33:00Z",
        },
    ]
}

CURRENT_LOG = "\n".join(f"current line {i}" for i in range(300))
PREVIOUS_LOG = "\n".join(
    [*(f"WARNING discovery {i}" for i in range(299)), "FATAL: the real cause"]
)


def _fake_kubectl(
    responses: dict[tuple[str, ...], tuple[int, str, str]],
) -> boot_gate.KubectlRunner:
    def run(args: list[str]) -> tuple[int, str, str]:
        return responses.get(tuple(args), (1, "", f"unexpected: {args}"))

    return run


def _runtime_responses() -> dict[tuple[str, ...], tuple[int, str, str]]:
    pods = {"items": [CRASHING_POD]}
    replicasets = {
        "items": [
            {
                "metadata": {
                    "name": "omninode-runtime-7d9",
                    "ownerReferences": [
                        {"kind": "Deployment", "name": "omninode-runtime"}
                    ],
                }
            }
        ]
    }
    selector = "app=omninode-runtime"
    pod = "omninode-runtime-7d9-abcde"
    logs = ("logs", "-n", NS, pod, "--all-containers", "--timestamps")
    return {
        ("get", "events", "-n", NS, "-o", "json"): (0, json.dumps(EVENTS), ""),
        ("get", "deployment", "omninode-runtime", "-n", NS, "-o", "json"): (
            0,
            json.dumps(RUNTIME_DEPLOYMENT),
            "",
        ),
        ("get", "pods", "-n", NS, "-l", selector, "-o", "json"): (
            0,
            json.dumps(pods),
            "",
        ),
        ("get", "replicasets", "-n", NS, "-l", selector, "-o", "json"): (
            0,
            json.dumps(replicasets),
            "",
        ),
        logs: (0, CURRENT_LOG, ""),
        (*logs, "--previous"): (0, PREVIOUS_LOG, ""),
    }


def test_explains_a_crashlooping_runtime_in_the_step_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = boot_gate.explain_not_ready(
        namespace=NS,
        names=["omninode-runtime"],
        log_lines=50,
        diagnostics_dir=tmp_path,
        run=_fake_kubectl(_runtime_responses()),
    )
    assert text in capsys.readouterr().err
    # Pod status and LAST state: the exit code is the first thing a reader needs.
    assert "CrashLoopBackOff" in text
    assert "restarts=14" in text
    assert "lastState=terminated (Error, exit 1" in text
    assert "MinimumReplicasUnavailable" in text
    # Events for the pod and its ReplicaSet, never the prefix-named sibling's.
    assert "Back-off restarting failed container" in text
    assert "SuccessfulCreate" in text
    assert "sibling-only-event" not in text
    # Previous logs carry the death; the tail keeps it and drops the noise.
    assert "FATAL: the real cause" in text
    assert "WARNING discovery 0\n" not in text
    assert "current line 299" in text
    assert "current line 0\n" not in text


def test_the_artifact_copy_keeps_the_logs_untruncated(tmp_path: Path) -> None:
    boot_gate.explain_not_ready(
        namespace=NS,
        names=["omninode-runtime"],
        log_lines=10,
        diagnostics_dir=tmp_path,
        run=_fake_kubectl(_runtime_responses()),
    )
    written = (tmp_path / "not-ready-omninode-runtime.txt").read_text()
    assert "current line 0\n" in written
    assert "WARNING discovery 0\n" in written
    assert "FATAL: the real cause" in written
    assert "CrashLoopBackOff" in written


def test_a_deployment_with_no_pod_reports_its_replicaset_events() -> None:
    """FailedCreate (admission, quota) lives on the ReplicaSet when no pod exists."""
    deployment = {
        "metadata": {"name": "omnimarket-projection-savings-writer"},
        "spec": {"selector": {"matchLabels": {"app": "savings"}}},
        "status": {},
    }
    events = {
        "items": [
            {
                "involvedObject": {
                    "kind": "ReplicaSet",
                    "name": "omnimarket-projection-savings-writer-6c8",
                },
                "type": "Warning",
                "reason": "FailedCreate",
                "message": 'pods "x" is forbidden: violates PodSecurity',
                "count": 12,
                "lastTimestamp": "2026-10-21T20:40:00Z",
            }
        ]
    }
    replicasets = {
        "items": [
            {
                "metadata": {
                    "name": "omnimarket-projection-savings-writer-6c8",
                    "ownerReferences": [
                        {
                            "kind": "Deployment",
                            "name": "omnimarket-projection-savings-writer",
                        }
                    ],
                }
            }
        ]
    }
    name = "omnimarket-projection-savings-writer"
    responses = {
        ("get", "events", "-n", NS, "-o", "json"): (0, json.dumps(events), ""),
        ("get", "deployment", name, "-n", NS, "-o", "json"): (
            0,
            json.dumps(deployment),
            "",
        ),
        ("get", "pods", "-n", NS, "-l", "app=savings", "-o", "json"): (
            0,
            json.dumps({"items": []}),
            "",
        ),
        ("get", "replicasets", "-n", NS, "-l", "app=savings", "-o", "json"): (
            0,
            json.dumps(replicasets),
            "",
        ),
    }
    text = boot_gate.explain_not_ready(
        namespace=NS, names=[name], run=_fake_kubectl(responses)
    )
    assert "no pod matches selector app=savings" in text
    assert "FailedCreate" in text
    assert "violates PodSecurity" in text


def test_an_unreadable_cluster_is_reported_not_raised() -> None:
    """This runs on the failure path; it must never turn a FAIL into a traceback."""

    def broken(args: list[str]) -> tuple[int, str, str]:
        return 127, "", "kubectl could not be run: not found"

    text = boot_gate.explain_not_ready(
        namespace=NS, names=["omninode-runtime"], run=broken
    )
    assert "deployment unreadable" in text


def test_wait_explains_exactly_the_not_ready_deployments_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [
        ("omnimarket-projection-savings-writer", 0, 1, ""),
        ("omninode-runtime", 0, 1, "MinimumReplicasUnavailable: x"),
        ("omninode-runtime-effects", 1, 1, ""),
        ("lane-a-redpanda", 1, 1, ""),
    ]
    seen: dict[str, Any] = {}

    def record(**kwargs: Any) -> str:
        seen.update(kwargs)
        return ""

    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)
    monkeypatch.setattr(boot_gate, "explain_not_ready", record)

    result = boot_gate.wait_for_boot(
        namespace=NS,
        timeout_seconds=0,
        poll_seconds=1,
        lane_prefix="lane-a-",
        broker_deployment="lane-a-redpanda",
        require_topic=True,
        log_lines=7,
        diagnostics_dir=tmp_path,
    )
    assert result == 1
    assert seen["names"] == ["omnimarket-projection-savings-writer", "omninode-runtime"]
    assert seen["log_lines"] == 7
    assert seen["diagnostics_dir"] == tmp_path


def test_the_workflow_writes_the_explanation_into_the_uploaded_artifact() -> None:
    workflow = yaml.safe_load(DELIVER.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["candidate-boot-gate"]["steps"]
    wait = next(step for step in steps if step.get("id") == "boot-wait")
    upload = next(step for step in steps if step.get("name") == "Upload diagnostics")
    upload_path = upload["with"]["path"]
    assert "boot_gate.py wait" in wait["run"]
    assert f"--diagnostics-dir {upload_path}/" in wait["run"], (
        "the not-ready explanation must land under the uploaded artifact path"
    )
