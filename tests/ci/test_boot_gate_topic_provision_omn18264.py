# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18264 — the lab lane provisions topics from the committed Job.

WHAT BROKE, and why only now. This lane's Redpanda runs `--mode=dev-container`,
which leaves auto-create ON, so a missing topic was invisible here and
OMN-18264's own note says the lab cannot reproduce the failure. That was true
when it was written. `SnapshotCache` used `subscribe()`, and a metadata request
that NAMES topics is the shape that auto-creates on a permissive broker, so the
consumer created the topics it then read.

omnimarket#3024 swapped `subscribe()` for manual `assign()`, and #3034 bounded
the resulting wait with `await consumer.topics()` — which asks for the whole
universe and names nothing, so it never auto-creates. Nothing else fills the
gap: `onex.snapshot.projection.consumer-flow.v1` has no writer Deployment on
this lane, so no producer creates it either.

Measured on delivery run 36365404894: every other runtime Deployment Ready,
`omnimarket-projection-api` 0/1 in CrashLoopBackOff at `Exit Code: 137` with its
log stopping at "Waiting for application startup.", and the topic ABSENT.

These tests pin that the renderer moves ONLY the three fields it must, reuses
omninode_infra's committed manifest rather than a second implementation, and
fails closed on every shape it cannot prove it understood — an un-rewritten Job
would run the wrong image against the lane and still report success.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "boot_gate.py"
_spec = importlib.util.spec_from_file_location("boot_gate_omn18264", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
boot_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boot_gate)

# Synthetic registry host: the renderer never parses the image string, it only
# substitutes it, so the real account id would be an exposed identifier for no
# test value (OMN-17320).
IMAGE = "registry.invalid/omninode-runtime@sha256:" + "ab" * 32


def _job(containers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if containers is None:
        containers = [
            {
                "name": "onex-topic-provision",
                "image": "…/omninode-runtime:placeholder",
                "envFrom": [{"configMapRef": {"name": "onex-runtime-config"}}],
            }
        ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "onex-topic-provision", "namespace": "onex-dev"},
        "spec": {
            "backoffLimit": 3,
            "activeDeadlineSeconds": 600,
            "template": {"spec": {"restartPolicy": "Never", "containers": containers}},
        },
    }


def _write(tmp_path: Path, docs: list[dict[str, Any]] | dict[str, Any]) -> Path:
    p = tmp_path / "job.yaml"
    if isinstance(docs, dict):
        p.write_text(yaml.safe_dump(docs, sort_keys=False))
    else:
        p.write_text(yaml.safe_dump_all(docs, sort_keys=False))
    return p


def _render(
    tmp_path: Path, manifest: Path, name: str = "onex-topic-provision-99"
) -> tuple[int, Path]:
    out = tmp_path / "rendered.yaml"
    rc = boot_gate.render_topic_provision_job(
        manifest=manifest, name=name, namespace="onex-dev", image=IMAGE, out=out
    )
    return rc, out


def test_it_rewrites_the_name_namespace_and_image(tmp_path: Path) -> None:
    rc, out = _render(tmp_path, _write(tmp_path, _job()))
    assert rc == 0
    d = yaml.safe_load(out.read_text())
    assert d["metadata"]["name"] == "onex-topic-provision-99"
    assert d["metadata"]["namespace"] == "onex-dev"
    assert d["spec"]["template"]["spec"]["containers"][0]["image"] == IMAGE


def test_the_run_unique_name_is_what_makes_a_second_apply_possible(
    tmp_path: Path,
) -> None:
    """A Job's pod template is immutable, so the committed name cannot be reused."""
    manifest = _write(tmp_path, _job())
    first = yaml.safe_load(_render(tmp_path, manifest, "prov-1")[1].read_text())
    second = yaml.safe_load(_render(tmp_path, manifest, "prov-2")[1].read_text())
    assert first["metadata"]["name"] != second["metadata"]["name"]


def test_the_broker_is_not_restated_only_inherited(tmp_path: Path) -> None:
    """envFrom must survive untouched: the overlay owns the broker address."""
    rc, out = _render(tmp_path, _write(tmp_path, _job()))
    assert rc == 0
    c = yaml.safe_load(out.read_text())["spec"]["template"]["spec"]["containers"][0]
    assert c["envFrom"] == [{"configMapRef": {"name": "onex-runtime-config"}}]
    assert "KAFKA_BOOTSTRAP_SERVERS" not in yaml.safe_dump(c)


def test_everything_other_than_those_three_fields_is_preserved(tmp_path: Path) -> None:
    rc, out = _render(tmp_path, _write(tmp_path, _job()))
    assert rc == 0
    d = yaml.safe_load(out.read_text())
    assert d["spec"]["backoffLimit"] == 3
    assert d["spec"]["activeDeadlineSeconds"] == 600
    assert d["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_a_manifest_holding_more_than_the_job_is_refused(tmp_path: Path) -> None:
    """Refusing to guess which document provisions topics."""
    extra = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "x"}}
    rc, _ = _render(tmp_path, _write(tmp_path, [_job(), extra]))
    assert rc == 1


def test_a_manifest_with_no_job_is_refused(tmp_path: Path) -> None:
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "x"}}
    rc, _ = _render(tmp_path, _write(tmp_path, cm))
    assert rc == 1


@pytest.mark.parametrize(
    "containers", [[], [{"name": "a", "image": "i"}, {"name": "b", "image": "i"}]]
)
def test_an_ambiguous_container_count_is_refused(
    tmp_path: Path, containers: list[dict[str, Any]]
) -> None:
    rc, _ = _render(tmp_path, _write(tmp_path, _job(containers)))
    assert rc == 1


def test_a_container_with_no_image_is_refused(tmp_path: Path) -> None:
    rc, _ = _render(tmp_path, _write(tmp_path, _job([{"name": "only"}])))
    assert rc == 1


def test_unparseable_yaml_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "job.yaml"
    p.write_text("{{ not yaml")
    rc, _ = _render(tmp_path, p)
    assert rc == 1


def test_the_gate_runs_provisioning_before_it_waits_for_the_topic() -> None:
    """Ordering is the whole point: provision, then wait."""
    wf = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "deliver-dev-candidate-to-staging.yml"
    )
    steps = [
        s.get("name", "")
        for s in yaml.safe_load(wf.read_text())["jobs"]["candidate-boot-gate"]["steps"]
    ]
    provision = next(
        i for i, n in enumerate(steps) if "Provision the contract-declared topics" in n
    )
    wait = next(
        i for i, n in enumerate(steps) if "projection topic" in n and "Wait" in n
    )
    assert provision < wait, steps[provision : wait + 1]


def test_the_step_uses_omninode_infras_committed_manifest_not_a_copy() -> None:
    """One provisioner for both lanes, or they drift."""
    wf = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "deliver-dev-candidate-to-staging.yml"
    )
    body = yaml.safe_load(wf.read_text())["jobs"]["candidate-boot-gate"]["steps"]
    step = next(
        s for s in body if "Provision the contract-declared topics" in s.get("name", "")
    )
    assert (
        "omninode_infra/k8s/onex-dev/runtime/job-onex-topic-provision.yaml"
        in step["run"]
    )
