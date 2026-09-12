# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 AC5/AC6 -- the onex-lab overlay applier.

Each test here pins a behaviour that was ABSENT on 2026-09-11, when the lab lane
ran ``omnimarket==0.4.30`` against a dev at 0.4.61 under a green trigger. The
applier's whole contract is that a failure is DATA -- a record with a failing
check -- and never an exception into the deploy path, so the negative cases are
the load-bearing ones.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.lab_overlay import (
    API_IMAGE_NAME,
    CLOUD_MIGRATE_IMAGE_NAME,
    INFRA_MIGRATE_IMAGE_NAME,
    LAB_LANE_VALUE,
    LANE_MANAGED_SELECTOR,
    OMNIMARKET_VERSION_PROGRAM,
    RUNTIME_IMAGE_NAME,
    RUNTIME_POD_SELECTOR,
    LabOverlayApplier,
    LabOverlayRefusalError,
    ModelLabOverlayCheck,
    load_record,
    normalise_image_ref,
    record_path,
    write_store_binding,
)

pytestmark = pytest.mark.unit

SHA = "b" * 40
STAMP = "20260912T010203Z"
MANIFEST_SHA = "c" * 40
# The k3s CONTENT digest of the promoted runtime image, which is what a pod's
# imageID carries for an imported image -- measured live on 2026-09-12, and NOT
# the Docker image id.
RUNTIME_DIGEST = "sha256:" + "d" * 64
POD_IMAGE_ID = f"docker.io/library/import-2026-09-12@{RUNTIME_DIGEST}"

STORE_ENV = {
    "INFISICAL_ADDR": "http://192.168.0.2:8880",
    "INFISICAL_CLIENT_ID": "sentinel-client-id-zzz",
    "INFISICAL_CLIENT_SECRET": "sentinel-client-secret-zzz",
    "INFISICAL_PROJECT_ID": "sentinel-project-id-zzz",
}


class FakePopen:
    """Stand-in for ``docker save``, whose stdout the import pipes rather than
    captures. Records the refs it was asked to save."""

    def __init__(self) -> None:
        self.refs: list[str] = []
        self.returncode = 0

    def __call__(self, argv: Any, **kwargs: Any) -> FakePopen:
        self.refs.append(list(argv)[-1])
        self.stdout = None
        return self

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class FakeRunner:
    """Records every argv and answers from a caller-supplied script.

    The applier funnels every external command through one callable precisely so
    the failure modes that matter -- a garbage-collected pin, an apply that exits
    non-zero, a lane whose pods keep the previous image -- are reachable without a
    lab host, a Docker daemon or a cluster.
    """

    def __init__(self, answers: dict[str, tuple[int, str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        argv = list(argv)
        self.calls.append(argv)
        joined = " ".join(argv)
        for needle, (code, out) in self.answers.items():
            if needle in joined:
                return subprocess.CompletedProcess(argv, code, out, "")
        return subprocess.CompletedProcess(argv, 0, self._default(joined), "")

    @staticmethod
    def _default(joined: str) -> str:
        if "rev-parse origin/dev" in joined:
            return MANIFEST_SHA
        if "images ls name==" in joined:
            # The DIGEST column, in the real `ctr images ls` shape: a header row
            # then one row per ref. The applier reads field 3.
            ref = joined.split("name==", 1)[1].split()[0]
            return (
                "REF TYPE DIGEST SIZE PLATFORMS LABELS\n"
                f"{ref} application/vnd.oci.image.index.v1+json {RUNTIME_DIGEST} "
                "760.0 MiB linux/amd64 io.cri-containerd.image=managed"
            )
        if "images ls -q" in joined:
            return "\n".join(
                [
                    f"docker.io/{RUNTIME_IMAGE_NAME}:{STAMP}-{SHA[:8]}",
                    f"docker.io/{INFRA_MIGRATE_IMAGE_NAME}:{STAMP}-{SHA[:8]}",
                    f"docker.io/{CLOUD_MIGRATE_IMAGE_NAME}:{MANIFEST_SHA[:8]}-{STAMP}",
                    f"docker.io/{API_IMAGE_NAME}:{MANIFEST_SHA[:8]}-{STAMP}",
                ]
            )
        if "get deployments" in joined:
            # Two lane-managed Deployments live, one of which the render no longer
            # declares: the 2026-09-12 state, where the standalone delegation
            # writer omninode_infra#1355 retired kept running after the apply.
            return json.dumps(
                {
                    "items": [
                        {"metadata": {"name": "omninode-runtime"}},
                        {
                            "metadata": {
                                "name": "omnimarket-projection-delegation-writer"
                            }
                        },
                    ]
                }
            )
        if "get pods" in joined:
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": "omninode-runtime-1"},
                            "status": {
                                "containerStatuses": [
                                    {
                                        "name": "omninode-runtime",
                                        "imageID": POD_IMAGE_ID,
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        if "importlib.metadata" in joined:
            return "0.4.61"
        return ""

    def argv_containing(self, needle: str) -> list[list[str]]:
        return [call for call in self.calls if needle in " ".join(call)]


@pytest.fixture
def overlay_source(tmp_path: Path) -> Path:
    """A stand-in omninode_infra clone whose archive carries the apply script."""
    source = tmp_path / "omninode_infra"
    (source / "k8s" / "onex-lab").mkdir(parents=True)
    (source / "k8s" / "onex-lab" / "apply_lab_lane.sh").write_text("#!/bin/sh\n")
    return source


def _applier(
    tmp_path: Path,
    overlay_source: Path,
    runner: FakeRunner,
    *,
    env: dict[str, str] | None = None,
) -> LabOverlayApplier:
    applier = LabOverlayApplier(
        state_dir=tmp_path / "state",
        repo_dir=tmp_path / "omnibase_infra",
        overlay_source_dir=overlay_source,
        env={**STORE_ENV, **(env or {})},
        runner=runner,
        popen=FakePopen(),
    )

    # `archive_overlay` shells out to git and tar, neither of which exists as a
    # real repository here. Stub the SOURCE materialisation only: everything the
    # tests are about happens after it.
    def _archive(destination: Path) -> str:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "k8s" / "onex-lab"
        target.mkdir(parents=True, exist_ok=True)
        (target / "apply_lab_lane.sh").write_text("#!/bin/sh\n")
        (destination / "docker").mkdir(exist_ok=True)
        return MANIFEST_SHA

    # apply_lab_lane.sh writes its render to $LAB_WORK_DIR; the real script is a
    # stub here, so the fixture writes the render the prune reads.
    render = tmp_path / "state" / "lab-overlay" / "work" / "apply"
    render.mkdir(parents=True, exist_ok=True)
    (render / "onex-lab-render.yaml").write_text(
        "kind: Deployment\nmetadata:\n  name: omninode-runtime\n"
    )

    applier.archive_overlay = _archive  # type: ignore[method-assign]
    applier.materialise_kubeconfig = lambda destination: destination  # type: ignore[method-assign]
    return applier


# --------------------------------------------------------------------------- #
# the record exists on BOTH outcomes                                          #
# --------------------------------------------------------------------------- #
def test_apply_writes_a_passing_record(tmp_path: Path, overlay_source: Path) -> None:
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id="cid-1")

    record = json.loads(path.read_text())
    assert record["sha"] == SHA
    assert record["lane"] == LAB_LANE_VALUE
    assert record["agent_command_id"] == "cid-1"
    assert all(check["ok"] for check in record["checks"]), record["checks"]
    names = [check["name"] for check in record["checks"]]
    assert names == [
        "overlay_source_resolved",
        "images_pinned",
        "lab_overlay_applied",
        "deployed_image",
        "runtime_omnimarket_version",
        "retired_workloads_pruned",
    ]


def test_a_failing_apply_still_writes_a_record(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The regressed case. A skipped emitter and a failed one look identical to
    the gate unless the failure is recorded, which is what OMN-18200 is about."""
    runner = FakeRunner({"apply_lab_lane.sh": (1, "FATAL: the lane is not strict")})
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id="cid-2")

    record = json.loads(path.read_text())
    applied = next(c for c in record["checks"] if c["name"] == "lab_overlay_applied")
    assert applied["ok"] is False
    assert "exited 1" in applied["evidence"]
    # Every check carries evidence even on a failure, so the receipt this feeds
    # cannot contain a verdict with nothing behind it.
    assert all(check["evidence"] for check in record["checks"])


def test_a_missing_pin_is_refused_before_the_apply_runs(
    tmp_path: Path, overlay_source: Path
) -> None:
    """A pin that is not in the k3s content store must be a named refusal, not an
    ImagePullBackOff discovered ten minutes into a rollout."""
    runner = FakeRunner({"images ls -q": (0, "docker.io/library/busybox:1.36")})
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    record = json.loads(path.read_text())
    pinned = next(c for c in record["checks"] if c["name"] == "images_pinned")
    assert pinned["ok"] is False
    assert "k3s content store" in pinned["evidence"]
    # The positive control is IN the evidence: a store that lists one image is an
    # absence, while a store that lists none would be an unreadable socket.
    assert "positive control" in pinned["evidence"]
    assert not applier._runner.argv_containing("apply_lab_lane.sh")


def test_an_unreadable_content_store_does_not_read_as_every_pin_missing(
    tmp_path: Path, overlay_source: Path
) -> None:
    """Rule 16. A failed read must raise, not return an empty set."""
    runner = FakeRunner({"images ls -q": (1, "permission denied")})
    applier = _applier(tmp_path, overlay_source, runner)

    with pytest.raises(LabOverlayRefusalError):
        applier.resident_images()


# --------------------------------------------------------------------------- #
# AC6 -- a stale lane cannot pass                                             #
# --------------------------------------------------------------------------- #
def test_a_lane_still_running_the_previous_image_fails_deployed_image(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The 2026-09-11 state itself: pods on a three-day-old image. Compared by
    image ID rather than by tag, because two builds can share a tag."""
    stale = json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "omninode-runtime-1"},
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "omninode-runtime",
                                "imageID": "docker.io/x@sha256:" + "e" * 64,
                            }
                        ]
                    },
                }
            ]
        }
    )
    runner = FakeRunner({"get pods": (0, stale)})
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    record = json.loads(path.read_text())
    deployed = next(c for c in record["checks"] if c["name"] == "deployed_image")
    assert deployed["ok"] is False
    assert "still on another image" in deployed["evidence"]


def test_a_lane_at_a_different_omnimarket_version_than_compose_fails(
    tmp_path: Path, overlay_source: Path
) -> None:
    """compose at 0.4.61, lab at 0.4.30, trigger green -- must be red."""
    calls: list[str] = []

    class VersionRunner(FakeRunner):
        def __call__(self, argv: Any, **kwargs: Any) -> subprocess.CompletedProcess:
            joined = " ".join(argv)
            if "importlib.metadata" in joined:
                calls.append(joined)
                version = "0.4.30" if "kubectl" in joined else "0.4.61"
                self.calls.append(list(argv))
                return subprocess.CompletedProcess(list(argv), 0, version, "")
            return super().__call__(argv, **kwargs)

    applier = _applier(tmp_path, overlay_source, VersionRunner())
    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    record = json.loads(path.read_text())
    version = next(
        c for c in record["checks"] if c["name"] == "runtime_omnimarket_version"
    )
    assert version["ok"] is False
    assert "0.4.30" in version["evidence"] and "0.4.61" in version["evidence"]
    assert len(calls) == 2, "both lanes must be read, not one"


def test_no_running_pod_fails_rather_than_passing_vacuously(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = FakeRunner({"get pods": (0, json.dumps({"items": []}))})
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    record = json.loads(path.read_text())
    for name in ("deployed_image", "runtime_omnimarket_version"):
        check = next(c for c in record["checks"] if c["name"] == name)
        assert check["ok"] is False, name


# --------------------------------------------------------------------------- #
# the apply invocation                                                        #
# --------------------------------------------------------------------------- #
def test_the_apply_carries_all_four_pins_and_both_store_flags(
    tmp_path: Path, overlay_source: Path
) -> None:
    """apply_lab_lane.sh refuses a partial pin set and refuses one store flag
    without the other, so the caller must pass all six or the lane is unpinned or
    half-configured."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    apply_calls = runner.argv_containing("apply_lab_lane.sh")
    assert len(apply_calls) == 1
    argv = apply_calls[0]
    for flag in (
        "--runtime-image",
        "--api-image",
        "--infra-migrate-image",
        "--cloud-migrate-image",
        "--lab-secret-store-env",
        "--lab-secret-store-environment",
    ):
        assert flag in argv, flag
    assert argv[argv.index("--runtime-image") + 1] == (
        f"{RUNTIME_IMAGE_NAME}:{STAMP}-{SHA[:8]}"
    )
    assert argv[argv.index("--api-image") + 1] == (
        f"{API_IMAGE_NAME}:{MANIFEST_SHA[:8]}-{STAMP}"
    )
    assert argv[argv.index("--lab-secret-store-environment") + 1] == "dev"


def test_all_four_pins_are_built_or_promoted_and_none_is_carried_forward(
    tmp_path: Path, overlay_source: Path
) -> None:
    """Measured 2026-09-12: k3s garbage-collects unreferenced images, and
    apply_lab_lane.sh deletes both migrate Jobs, so between applies nothing
    references either migrate image and both become collectable. An apply that
    pinned what it found died at the migrate barrier and left three runtime pods
    crash-looping on a missing db_metadata relation. So THREE images are built
    every run and the fourth is promoted -- never carried forward from the
    ledger, from docker, or from the content store."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    builds = runner.argv_containing("docker build")
    assert len(builds) == 3, [" ".join(b) for b in builds]
    built = {b[b.index("-t") + 1].split(":")[0] for b in builds}
    assert built == {
        INFRA_MIGRATE_IMAGE_NAME,
        CLOUD_MIGRATE_IMAGE_NAME,
        API_IMAGE_NAME,
    }
    # The runtime image is TAGGED, never rebuilt: a second build of the same
    # source is a different digest for no reason.
    assert len(runner.argv_containing("docker tag")) == 1
    assert not [c for c in builds if "Dockerfile.runtime" in " ".join(c)]
    # Every one of the four is imported into containerd, because residency
    # between applies is the thing that is not guaranteed.
    assert len(applier._popen.refs) == 4, applier._popen.refs


def test_no_pin_is_read_off_a_live_deployment(
    tmp_path: Path, overlay_source: Path
) -> None:
    """A resident image is not a pinnable one. The caller must never ask the lane
    what it is running and reuse that as a pin -- that is what kept the lane
    stale while passing."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    for call in runner.argv_containing("kubectl"):
        joined = " ".join(call)
        assert "deployment/onex-api" not in joined
        assert "spec.template.spec.containers[0].image" not in joined
    assert not hasattr(applier, "carried_forward_api_pin")


def test_the_api_image_is_built_the_way_ci_builds_it(
    tmp_path: Path, overlay_source: Path
) -> None:
    """Same Dockerfile and same CONTEXT as build-and-push-onex-api.yml. The
    context is docker/onex-api, not the repo root, so a lab image built from a
    wider tree would be a lookalike rather than the image CI ships."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    api = next(
        b
        for b in runner.argv_containing("docker build")
        if API_IMAGE_NAME in " ".join(b)
    )
    assert api[api.index("-f") + 1] == "docker/onex-api/Dockerfile"
    assert api[-1] == "docker/onex-api"


def test_a_retired_deployment_is_pruned_and_recorded(
    tmp_path: Path, overlay_source: Path
) -> None:
    """Measured 2026-09-12: after an apply that pinned a fresh image, 16 of 17
    Deployments carried it and the seventeenth was the standalone delegation
    writer omninode_infra#1355 had already retired, still four days stale.
    apply_lab_lane.sh applies a document set and never prunes, so left alone the
    lane runs retired workloads and the receipt says PASS."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    path = applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    record = json.loads(path.read_text())
    pruned = next(
        c for c in record["checks"] if c["name"] == "retired_workloads_pruned"
    )
    assert pruned["ok"] is True
    assert "omnimarket-projection-delegation-writer" in pruned["evidence"]
    deletes = runner.argv_containing("delete deployment/")
    assert len(deletes) == 1
    assert "omnimarket-projection-delegation-writer" in " ".join(deletes[0])
    # The one the render declares is never touched.
    assert "deployment/omninode-runtime" not in " ".join(" ".join(c) for c in deletes)


def test_the_prune_only_reaches_lane_managed_objects(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The scope is a label selector, not a filter on good intentions: anything a
    person or another system put in this namespace is out of reach."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    listing = next(c for c in runner.argv_containing("get deployments") if "-l" in c)
    assert listing[listing.index("-l") + 1] == LANE_MANAGED_SELECTOR


def test_an_empty_render_is_refused_rather_than_retiring_everything(
    tmp_path: Path,
) -> None:
    """Rule 16. A render declaring no Deployment is a failed read, not a
    statement that every workload on the lane is retired."""
    runner = FakeRunner()
    applier = LabOverlayApplier(
        state_dir=tmp_path / "state",
        repo_dir=tmp_path,
        overlay_source_dir=tmp_path,
        env=STORE_ENV,
        runner=runner,
        popen=FakePopen(),
    )
    empty = tmp_path / "render.yaml"
    empty.write_text("kind: ConfigMap\nmetadata:\n  name: x\n")

    with pytest.raises(LabOverlayRefusalError, match="declares no Deployment"):
        applier.declared_deployments(empty)
    assert not runner.argv_containing("delete deployment/")


def test_no_in_place_mutation_verb_ever_reaches_the_lane(
    tmp_path: Path, overlay_source: Path
) -> None:
    """The overlay's own apply path is the only thing that WRITES objects; this
    module never edits a live one. The verbs that edit in place -- apply, patch,
    set, edit, replace, scale, annotate, label -- are the drift k8s/onex-lab
    exists to refuse, and none of them appears.

    `delete` is the one exception and it is a narrow one: pruning a Deployment
    the render no longer declares removes a retired workload, it does not edit a
    declared one. It is asserted separately to be label-scoped and
    render-authorised."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    forbidden = {
        "apply",
        "patch",
        "set",
        "edit",
        "replace",
        "scale",
        "annotate",
        "label",
        "rollout",
    }
    for call in runner.argv_containing("kubectl"):
        assert call[0] == "kubectl"
        assert call[3] in {"get", "exec", "delete"}, call
        assert not forbidden & set(call), call
        if call[3] == "delete":
            assert call[4].startswith("deployment/"), call


# --------------------------------------------------------------------------- #
# the store binding                                                           #
# --------------------------------------------------------------------------- #
def test_the_store_binding_is_0600_and_never_reaches_a_command_line(
    tmp_path: Path, overlay_source: Path
) -> None:
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)
    applier.apply(sha=SHA, stamp=STAMP, correlation_id=None)

    for call in runner.calls:
        joined = " ".join(call)
        assert STORE_ENV["INFISICAL_CLIENT_SECRET"] not in joined
        assert STORE_ENV["INFISICAL_CLIENT_ID"] not in joined
    # The apply is handed a PATH, and the file is gone afterwards.
    argv = runner.argv_containing("apply_lab_lane.sh")[0]
    binding = Path(argv[argv.index("--lab-secret-store-env") + 1])
    assert binding.name == "store-binding.env"
    assert not binding.exists()


def test_write_store_binding_permissions_and_content(tmp_path: Path) -> None:
    target = tmp_path / "binding.env"
    write_store_binding(target, STORE_ENV)

    assert oct(target.stat().st_mode)[-3:] == "600"
    body = target.read_text()
    for key, value in STORE_ENV.items():
        assert f"{key}={value}\n" in body


def test_write_store_binding_names_every_gap_at_once(tmp_path: Path) -> None:
    """A blank value is a missing value, and all gaps are named in one pass --
    the same posture lab_secret_store.py takes for the same five keys."""
    partial = {**STORE_ENV, "INFISICAL_CLIENT_ID": "", "INFISICAL_PROJECT_ID": "   "}

    with pytest.raises(LabOverlayRefusalError) as excinfo:
        write_store_binding(tmp_path / "binding.env", partial)

    message = str(excinfo.value)
    assert "INFISICAL_CLIENT_ID" in message
    assert "INFISICAL_PROJECT_ID" in message
    assert "never minted" in message


# --------------------------------------------------------------------------- #
# small units                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("onex-lab/omninode-runtime:t", "docker.io/onex-lab/omninode-runtime:t"),
        ("docker.io/onex-lab/x:t", "docker.io/onex-lab/x:t"),
        ("busybox:1.36", "docker.io/library/busybox:1.36"),
        ("192.168.0.2:5000/x:t", "192.168.0.2:5000/x:t"),
    ],
)
def test_normalise_image_ref(ref: str, expected: str) -> None:
    """The k3s content store lists an unqualified name under docker.io. Comparing
    the two spellings directly is how a resident image reads as absent."""
    assert normalise_image_ref(ref) == expected


def test_load_record_distinguishes_absent_from_malformed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    assert load_record(state, SHA) is None

    path = record_path(state, SHA)
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_record(state, SHA)


def test_a_check_without_evidence_is_refused() -> None:
    """An ok:true with no evidence is indistinguishable from a check that was
    never run, which is the shape the receipt contract exists to refuse."""
    with pytest.raises(ValueError, match="evidence is required"):
        ModelLabOverlayCheck(name="x", ok=True, evidence="")


# --------------------------------------------------------------------------- #
# three facts measured on the live lane, pinned so they cannot regress         #
# --------------------------------------------------------------------------- #
def test_the_pod_selector_uses_the_label_the_overlay_actually_stamps() -> None:
    """Measured 2026-09-12 on the lab lane: the runtime pods carry
    ``app.kubernetes.io/name=omninode-runtime`` and no ``app`` label at all. A
    selector of ``app=...`` matched ZERO pods, which would have made the readback
    report "no Running pod" -- a permanent false FAIL -- on every single run."""
    assert RUNTIME_POD_SELECTOR == "app.kubernetes.io/name=omninode-runtime"


def test_the_version_probe_reads_the_installed_distribution() -> None:
    """Measured 2026-09-12 on BOTH lanes: ``omnimarket.__version__`` is the
    literal 0.1.0 in every image, while ``importlib.metadata`` reports 0.4.61 on
    the compose lane and 0.4.30 on the lab lane -- the two numbers this ticket is
    about. A comparison built on ``__version__`` would have compared 0.1.0 against
    0.1.0 and PASSED on the exact discrepancy it exists to catch."""
    assert "importlib.metadata" in OMNIMARKET_VERSION_PROGRAM
    assert "__version__" not in OMNIMARKET_VERSION_PROGRAM


def test_the_promoted_digest_is_the_content_digest_not_the_docker_image_id(
    tmp_path: Path, overlay_source: Path
) -> None:
    """Measured 2026-09-12: a pod's ``imageID`` for an imported image is the OCI
    index digest containerd holds (``sha256:5e65c13a…`` for the lane's then-current
    pin, byte-identical to the DIGEST column of ``ctr images ls``), never the
    Docker config digest. Comparing against ``docker inspect {{.Id}}`` can match
    nothing, so a correctly applied lane would read as stale forever."""
    runner = FakeRunner()
    applier = _applier(tmp_path, overlay_source, runner)

    digest = applier.promote_and_import(source="local:latest", target="onex-lab/x:t")

    assert digest == RUNTIME_DIGEST
    assert not runner.argv_containing("docker inspect"), (
        "the docker image id is not the value a pod imageID carries"
    )


def test_an_absent_content_store_row_is_a_refusal_not_an_empty_digest(
    tmp_path: Path, overlay_source: Path
) -> None:
    """An empty digest would be `in` every pod's imageID and turn the readback
    into a vacuous PASS."""
    runner = FakeRunner({"images ls name==": (0, "REF TYPE DIGEST SIZE\n")})
    applier = _applier(tmp_path, overlay_source, runner)

    with pytest.raises(LabOverlayRefusalError, match="no digest row"):
        applier.containerd_digest("onex-lab/x:t")
