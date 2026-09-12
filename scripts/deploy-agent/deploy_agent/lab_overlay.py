# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 AC5/AC6/AC7 -- the automated caller of the onex-lab apply path.

WHY THIS FILE EXISTS
--------------------
``omni_home`` ``CLAUDE.md`` rule 24(a) states that every runtime-affecting merge
to ``dev`` re-applies the k3s ``onex-lab`` overlay from the merged head and emits
a lab-pass receipt for it. Measured 2026-09-11, that half of the rule had **no
implementation at all**: ``omninode_infra``'s ``k8s/onex-lab/apply_lab_lane.sh``
had zero callers anywhere in the org, so the persistent lab lane was advanced
only when a person ran the script by hand. It sat on an image stamped
2026-09-08 running ``omnimarket==0.4.30`` while ``dev`` was at ``0.4.61``, and
the rebuild trigger reported green the whole time. This module is that caller.

WHY IT LIVES IN THE DEPLOY AGENT AND NOT IN A WORKFLOW JOB
----------------------------------------------------------
Three capabilities are needed at once, and exactly one process on the lab host
has all three.

*Host root.* The k3s admin kubeconfig is ``0600 root`` and the k3s containerd
socket is root-owned, so pinning an image into ``onex-lab/`` means importing it
into that containerd. The runner fleet has the Docker socket and a **read-only**
``lab-ci-reader`` ServiceAccount kubeconfig (OMN-18188) -- enough to read the
lane, not to apply to it. A runner could reach host root only by launching a
privileged container through the Docker socket, which is a worse blast radius
than using the process that already owns host root by design.

*The image.* The runtime image for the merged sha exists on the lab host's
Docker daemon because this agent just built it. Nothing has to be pulled, and no
registry credential is involved.

*The completion signal.* OMN-18200 AC3's asynchronous gap is real: the agent's
build begins minutes after the command is published, so nothing that runs
synchronously with the publish can know the lane converged. This module runs
**after** this agent's own verify step, so the signal is not a wall clock.

WHAT IS SPLIT OUT, AND WHY
--------------------------
A GitHub Actions artifact can only be created by a job inside a run, so the
receipt itself cannot be written here. This module writes a **record** -- the
same ``{name, ok, evidence}`` check list ``scripts/ci/lab_pass_receipt.py``
consumes through ``--checks-json`` -- into the agent's state directory, keyed by
sha, and the agent's existing HTTP surface serves it. The new
``verify-lab-overlay-converged`` job in ``runtime-rebuild-trigger.yml`` reads it
and emits the receipt. One writer of facts, one publisher of receipts.

THE RECORD IS WRITTEN ON BOTH OUTCOMES, and that is the whole point (AC2's
shape, applied to this lane). A record that exists only when everything worked
cannot tell "it failed" apart from "nobody ran it", and telling those apart is
the receipt's entire job. Every failure below therefore becomes a FAILING CHECK
rather than an exception: the apply's failure is data, not an error.

FAILURE HERE NEVER FAILS THE COMPOSE DEPLOY. The compose dev lane converged on
its own merits before this module is called; turning a lab-overlay failure into
a failed rebuild would report a lane that IS running the merged sha as broken.
The lab verdict travels in its own receipt, on its own lane value.

THE FOUR IMAGE PINS, ALL BUILT OR PROMOTED EVERY RUN
----------------------------------------------------
``apply_lab_lane.sh`` commits no digest by design (OMN-17533 AC-5) and requires
the caller to pin all four images. This module builds or promotes **all four**
from the merged sha on **every** run and pins what it just imported. Nothing is
ever carried forward from the ledger, from the host's Docker daemon, or from the
content store.

That is a measured requirement. **k3s garbage-collects unreferenced images**, and
``apply_lab_lane.sh`` deletes both migrate Jobs on completion, so from the moment
an apply finishes nothing references either migrate image. Measured on the lab
lane 2026-09-12: both were resident at a 15:54Z apply and gone by that night. The
apply that then pinned what it found passed the preflight, wrote every object, and
died at the migrate barrier because the Job could not pull an image that was no
longer there -- leaving three runtime pods in CrashLoopBackOff on a missing
``public.db_metadata`` relation, migrations never having run. A resident image is
therefore not a pinnable one.

The worse half of a scavenging caller is not that it breaks; it is that it
*keeps the lane stale and passes*. Nothing rebuilds these images on their own,
which is exactly why the lane sat three days behind, so a caller that faithfully
re-pins what it finds reproduces the defect it was written to remove. Building
every pin also gets the AC6 readback for free: the version in the pod has to
match the sha by construction.

``--runtime-image``
    ``docker tag`` of the image this agent just built for the compose lane from
    the merged sha, promoted into the ``onex-lab/`` name space and imported. The
    one pin that is promoted rather than rebuilt, because a second build of the
    same source is a different digest for no reason. Imported every run like the
    rest.
``--infra-migrate-image``
    Built from ``docker/Dockerfile.migrate`` in this agent's own clone, already
    at the merged sha. ``alpine`` plus two ``COPY`` layers, so seconds. It is the
    OMN-17702 barrier the runtime image's entrypoint dies without.
``--cloud-migrate-image``
    Built the same way from the archived overlay tree, so its lineage is the
    overlay's lineage.
``--api-image``
    Built from the overlay tree's ``docker/onex-api/Dockerfile`` with context
    ``docker/onex-api`` -- the same file and the same context
    ``build-and-push-onex-api.yml`` uses, so the lab image is the image CI builds
    and not a lookalike assembled from a wider tree. A two-stage
    ``python:3.12-slim`` build, minutes cold and fast warm.

Each pin is then verified resident in the content store immediately before the
apply, so a pin that is somehow still absent after its own build is a named
refusal rather than an ``ImagePullBackOff`` ten minutes into a rollout.

THE APPLY DOES NOT PRUNE, SO THIS MODULE DOES
---------------------------------------------
``apply_lab_lane.sh`` applies a document set. A workload removed from the
manifests therefore keeps running on the lane forever, on whatever image it was
last given. Measured 2026-09-12: after an apply that pinned a fresh image, 16 of
17 Deployments carried it and the seventeenth was the standalone
``omnimarket-projection-delegation-writer`` that ``omninode_infra#1355`` had
already retired, still on a stamp four days old. Left alone, the lane runs retired
workloads and the receipt says PASS -- the same false green in a new place. The
prune considers only Deployments carrying both of the overlay's own management
labels, treats the render as the sole authority on what survives, refuses a render
that declares no Deployment at all, and reports every deletion in the record.

THE OVERLAY LINEAGE IS NOT THE MERGED SHA, and the receipt says so. The merged
sha is an ``omnibase_infra`` commit; the overlay is ``omninode_infra`` code. The
apply therefore uses ``omninode_infra`` at ``origin/dev``, recorded in the
record's evidence as ``manifest_sha``. Conflating the two would put a sha in the
evidence that names no tree the apply ever read.

NOTHING MUTATES A SHARED CLONE'S WORKING TREE. The overlay source is obtained
with ``git fetch`` plus ``git archive`` into this agent's own state directory:
``fetch`` writes only refs, and ``archive`` writes only into the destination, so
a peer lane working in that canonical clone is untouched.

THE SECRET-STORE BINDING IS WRITTEN BY REDIRECT AND SHREDDED. The lane needs the
OMN-18168 managed-store binding or provider-key registration through the product
surface answers 503, which is what makes the C7 chains meaningless on this lane.
The identity is the one the lab host ALREADY HOLDS in the agent's own env store
-- reused, never minted, never rotated (rule 22) -- written to a ``0600`` file
with ``os.open`` and no value on a command line, in an environment, or in a log,
then removed in a ``finally``. ``k8s/onex-lab/lab_secret_store.py`` refuses any
address that is not a lab address, so the blast radius of this flag is bounded
by code this module does not own.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

#: The receipt lane value this module's record is emitted under.
#:
#: NOT ``onex-lab``. That value is ALREADY the ephemeral ``kind`` cluster the
#: ``candidate-boot-gate`` job creates and destroys in
#: ``deliver-dev-candidate-to-staging.yml``, which emits
#: ``lab-pass-receipt-onex-lab-<sha>`` for the same sha on the same repository.
#: ``lab_pass_receipt.evaluate_gate`` takes the NEWEST artifact for a name, on
#: the documented premise that a name has one emitter and a later artifact is a
#: re-run of it. A second, unrelated emitter on that name would therefore
#: silently supersede the boot gate's verdict -- a green kind boot masked by a
#: red lab apply, or the reverse -- with nothing anywhere saying so.
#:
#: They are also not the same claim. The kind cluster proves the candidate boots
#: against the real manifests on a throwaway node with one side-loaded image.
#: This lane proves the PERSISTENT lab cluster is running the merged sha, with
#: the host's own secret store bound and the lane's tenant minted, which is the
#: surface ``provider_key_chain.sh`` and the m3 chain runners actually grade
#: against.
LAB_LANE_VALUE = "onex-lab-k3s"

#: The lab namespace. Matches ``apply_lab_lane.sh``'s own default.
LAB_NAMESPACE = "onex-dev"

#: The Deployment whose image is the carried-forward ``--api-image`` pin.
API_DEPLOYMENT = "onex-api"

#: The Deployment the runtime-pin comparison is read back from. Its image is
#: rewritten by the overlay on every runtime-family container, so any one of
#: them witnesses the pin; this is the one whose name is stable across the
#: overlay's history.
RUNTIME_DEPLOYMENT = "omninode-runtime"

#: The label the overlay actually stamps, read off the live lane on
#: 2026-09-12 rather than assumed. NOT ``app``: the k8s recommended key is what
#: these manifests use, and a selector of ``app=omninode-runtime`` matches ZERO
#: pods on this lane -- which would have made the readback report "no Running
#: pod" on every single run, a permanent false FAIL. Found by a positive
#: control, not by review.
RUNTIME_POD_SELECTOR = f"app.kubernetes.io/name={RUNTIME_DEPLOYMENT}"

#: The two labels the overlay stamps on everything it owns, read off the live
#: lane rather than assumed. The prune considers ONLY objects carrying both, so a
#: workload a person or another system put in this namespace is out of reach of
#: it by construction.
LANE_MANAGED_SELECTOR = "omninode/managed=true,omninode/lane=onex-lab"

#: The ``onex-lab/`` names the three built-or-promoted pins are published under.
#: ``docker.io/`` is prepended when the reference is compared against the k3s
#: content store, which normalises every unqualified name to that registry.
RUNTIME_IMAGE_NAME = "onex-lab/omninode-runtime"
INFRA_MIGRATE_IMAGE_NAME = "onex-lab/omnibase-infra-migrate"
CLOUD_MIGRATE_IMAGE_NAME = "onex-lab/omninode-cloud-migrate"
API_IMAGE_NAME = "onex-lab/omnicloud-core"

#: Build recipes for the three images this caller builds from source, keyed by
#: the repository tree they are built in. Context and Dockerfile are BOTH named
#: because they differ: the two migrate bundles build from their repo root while
#: onex-api builds from ``docker/onex-api`` -- the same context
#: ``build-and-push-onex-api.yml`` uses, so the lab image is the image CI builds
#: and not a lookalike assembled from a wider tree.
MIGRATE_DOCKERFILE = "docker/Dockerfile.migrate"
MIGRATE_CONTEXT = "."
API_DOCKERFILE = "docker/onex-api/Dockerfile"
API_CONTEXT = "docker/onex-api"

#: The five bootstrap keys ``k8s/onex-lab/lab_secret_store.py`` demands, minus
#: the environment slug, which ``apply_lab_lane.sh`` passes as its own flag.
#: Verbatim from that module's ``REQUIRED_KEYS``; the two halves of the chain
#: must demand the same bootstrap or the lane can be configured well enough to
#: accept a customer's key and not well enough to hand it back.
STORE_BINDING_KEYS: tuple[str, ...] = (
    "INFISICAL_ADDR",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
)

#: The store environment customer keys are written into. Never defaulted inside
#: ``apply_lab_lane.sh`` (OMN-17349 AC5: an unset slug must refuse the write,
#: not pick an environment); declared here because THIS caller is the thing that
#: knows which lane it is applying.
STORE_BINDING_ENVIRONMENT = "dev"

#: The installed-distribution version, NOT ``omnimarket.__version__``.
#:
#: Measured on both lanes 2026-09-12: ``__version__`` is the literal ``0.1.0``
#: in every image on both the compose lane and the k3s lab lane, while
#: ``importlib.metadata`` reports 0.4.61 and 0.4.30 respectively -- which are the
#: two numbers the OMN-18200 diagnosis is about. A cross-lane comparison built on
#: ``__version__`` would have compared 0.1.0 against 0.1.0 and PASSED on exactly
#: the discrepancy it exists to catch.
OMNIMARKET_VERSION_PROGRAM = (
    "from importlib.metadata import version; "
    "import sys; sys.stdout.write(version('omnimarket'))"
)

#: Named rather than silently absent, so a reader can see what an
#: ``onex-lab-k3s`` receipt does NOT cover. Each needs a surface this module
#: does not reach today. Widening the set means adding the probe AND the check
#: name in one change -- never the name alone.
PROBES_NOT_YET_WIRED: tuple[str, ...] = (
    "consumer_group_lag",
    "delegation_golden_chain",
    "provider_key_chain",
)

#: Whole-apply ceiling. ``apply_lab_lane.sh`` carries its own internal waits
#: (300s for Postgres, 900s per migrate bundle, 900s per Deployment rollout,
#: 600s for the tenant mint) and this bound is the outer one: it is what keeps a
#: wedged lab apply from holding this agent's single-flight lock indefinitely
#: and serialising the next merge's rebuild behind it. A budget that expires is
#: recorded as a failing check, never as a silent truncation.
DEFAULT_APPLY_BUDGET_SECONDS = 1500

#: Bounds for the short host commands. Generous for a loaded 32-core host, and
#: short enough that a hung probe surfaces as a failing check inside the outer
#: budget rather than consuming it.
SHORT_TIMEOUT_SECONDS = 120
#: ``docker save | ctr images import`` of a multi-gigabyte runtime image.
IMPORT_TIMEOUT_SECONDS = 900
#: The two migrate bundles are ``alpine`` plus ``COPY`` layers.
MIGRATE_BUILD_TIMEOUT_SECONDS = 600
#: ``onex-api`` is a two-stage ``python:3.12-slim`` build with a pip install, so
#: it is minutes rather than seconds on a cold layer cache and fast on a warm
#: one. It is still built EVERY run -- see ``_derive_pins`` for why a resident
#: image is not a pinnable one.
API_BUILD_TIMEOUT_SECONDS = 1200


class LabOverlayRefusalError(Exception):
    """A precondition this module will not proceed without.

    Raised only inside the record-producing boundary, where it is converted into
    a failing check. It never escapes into the deploy path.
    """


@dataclass(frozen=True)
class ModelLabOverlayCheck:
    """One named check and the evidence for its verdict.

    Field-for-field the shape ``scripts/ci/lab_pass_receipt.py``'s
    ``ModelLabPassCheck`` validates, so the record this module writes is
    consumable by ``emit --checks-json`` with no translation layer. Evidence is
    required on BOTH verdicts: an ``ok: true`` with no evidence is
    indistinguishable from a check that was never run.
    """

    name: str
    ok: bool
    evidence: str

    def __post_init__(self) -> None:
        if not self.name:
            msg = "check name is required"
            raise ValueError(msg)
        if not isinstance(self.ok, bool):
            msg = f"check {self.name!r}: ok must be a bool"
            raise ValueError(msg)
        if not self.evidence:
            msg = (
                f"check {self.name!r}: evidence is required. A check with no "
                "evidence is indistinguishable from one that was never run."
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "evidence": self.evidence}


@dataclass(frozen=True)
class ModelRunningImage:
    """What one Running pod reports about the image it is actually running.

    ``reference`` is the tag the kubelet resolved; ``image_id`` is the digest
    containerd reports for it. They answer different questions and only the
    first is comparable against a pin -- see ``_readback_checks``.
    """

    reference: str
    image_id: str


@dataclass(frozen=True)
class ModelLabOverlayPins:
    """The four image references one apply pins, and the lineage of each."""

    runtime_image: str
    api_image: str
    infra_migrate_image: str
    cloud_migrate_image: str
    #: The ``omninode_infra`` commit the overlay itself was rendered from. NOT
    #: the merged ``omnibase_infra`` sha -- see this module's header.
    manifest_sha: str

    def apply_arguments(self) -> list[str]:
        return [
            "--runtime-image",
            self.runtime_image,
            "--api-image",
            self.api_image,
            "--infra-migrate-image",
            self.infra_migrate_image,
            "--cloud-migrate-image",
            self.cloud_migrate_image,
        ]

    def as_evidence(self) -> str:
        return (
            f"runtime={self.runtime_image} api={self.api_image} "
            f"infra-migrate={self.infra_migrate_image} "
            f"cloud-migrate={self.cloud_migrate_image} "
            f"overlay=omninode_infra@{self.manifest_sha}"
        )


@dataclass(frozen=True)
class ModelLabOverlayRecord:
    """What the CI emitter reads back, keyed by the merged sha.

    Deliberately NOT a receipt. A receipt carries a derived verdict and is
    published as an artifact; this is the check list and the window, written by
    the process that did the work. The verdict is derived once, by
    ``lab_pass_receipt.py``, from these checks -- so there is no path where a
    writer states a verdict its own checks contradict.
    """

    sha: str
    lane: str
    started_at: str
    finished_at: str
    agent_command_id: str | None
    checks: tuple[ModelLabOverlayCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha": self.sha,
            "lane": self.lane,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "agent_command_id": self.agent_command_id,
            "checks": [check.to_dict() for check in self.checks],
        }


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(text: str, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def record_path(state_dir: Path, sha: str) -> Path:
    """Where one sha's record lives.

    Under the agent's own state directory so it inherits the durability the job
    store already has, and named by sha so the HTTP reader cannot be handed a
    record for a different commit.
    """
    return Path(state_dir) / "lab-overlay" / f"{sha}.json"


def load_record(state_dir: Path, sha: str) -> dict[str, Any] | None:
    """Read one record back, or ``None`` when none exists.

    A malformed record raises rather than reading as absent: "the writer wrote
    something unparseable" and "the writer never ran" are different facts and
    the reader must not collapse them.
    """
    path = record_path(state_dir, sha)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path}: a record must be a JSON object"
        raise ValueError(msg)
    return payload


def write_store_binding(path: Path, env: Mapping[str, str]) -> None:
    """Write the OMN-18168 store binding file, by redirect only.

    ``os.open`` with ``0o600`` rather than ``Path.write_text`` plus ``chmod``:
    the second form creates the file at the process umask first, so the values
    exist world-readable for as long as it takes to narrow them. Nothing is
    logged, echoed, or placed on a command line -- ``apply_lab_lane.sh`` is
    handed the PATH, never the values.

    Every missing or blank key is named at once rather than one per run, which
    is the posture ``lab_secret_store.py`` records for the same five keys.
    """
    missing = [key for key in STORE_BINDING_KEYS if not (env.get(key) or "").strip()]
    if missing:
        msg = (
            "the lab store binding cannot be assembled: "
            f"{', '.join(missing)} missing or blank in the agent env store. "
            "The identity is the one the lab host already holds; it is reused, "
            "never minted (rule 22)."
        )
        raise LabOverlayRefusalError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key}={env[key]}\n" for key in STORE_BINDING_KEYS)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)


def shred(path: Path) -> None:
    """Overwrite then unlink. Never raises -- a cleanup that can fail the job it
    is cleaning up after would turn a successful apply into a failed one."""
    try:
        if path.exists():
            size = path.stat().st_size
            fd = os.open(str(path), os.O_WRONLY)
            try:
                os.write(fd, b"\0" * size)
                os.fsync(fd)
            finally:
                os.close(fd)
            path.unlink()
    except OSError as exc:
        logger.warning("lab_overlay: could not shred %s: %s", path, exc)


def normalise_image_ref(ref: str) -> str:
    """The form the k3s content store lists an unqualified name under.

    ``k3s ctr images ls`` reports ``docker.io/onex-lab/x:tag`` for an image
    imported as ``onex-lab/x:tag``. Comparing the two spellings directly is how
    a resident image reads as absent -- and, worse, how the same image under two
    names reaches a fixed-name Job as ``field is immutable``, which
    ``apply_lab_lane.sh``'s own header records taking the whole lane down.
    """
    if "/" not in ref:
        return f"docker.io/library/{ref}"
    head = ref.split("/", 1)[0]
    if "." in head or ":" in head or head == "localhost":
        return ref
    return f"docker.io/{ref}"


class LabOverlayApplier:
    """Applies the onex-lab overlay for one merged sha and records the result.

    Every external command goes through ``self._run`` so the whole class is
    testable without a lab host, a Docker daemon, or a cluster. That is not
    decoration: the failure modes this module must get right -- a
    garbage-collected pin, an apply that exits non-zero, a lane whose pods keep
    the previous image -- are all reachable only by controlling what the
    commands return.
    """

    def __init__(
        self,
        *,
        state_dir: Path,
        repo_dir: Path,
        overlay_source_dir: Path,
        env: Mapping[str, str],
        budget_seconds: int = DEFAULT_APPLY_BUDGET_SECONDS,
        runner: Any = None,
        popen: Any = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.repo_dir = Path(repo_dir)
        self.overlay_source_dir = Path(overlay_source_dir)
        self.env = dict(env)
        self.budget_seconds = budget_seconds
        self._runner = runner or subprocess.run
        #: ``docker save`` is the one command whose stdout is PIPED rather than
        #: captured, so it needs Popen rather than run. Injected for the same
        #: reason every other command is: the import path must be testable
        #: without a multi-gigabyte image and a root-owned socket.
        self._popen = popen or subprocess.Popen
        #: The k3s CONTENT digest of the runtime image actually promoted this
        #: run -- not the Docker image id, which a pod's ``imageID`` never
        #: equals for an imported image (see ``promote_and_import``). Set by
        #: ``_derive_pins`` and read by the readback. Empty until then, and the
        #: readback refuses an empty value rather than passing vacuously.
        self._runtime_digest: str = ""

    # -- command plumbing ---------------------------------------------------
    def _run(
        self,
        argv: Sequence[str],
        *,
        timeout: int,
        cwd: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        child_env = dict(self.env)
        if extra_env:
            child_env.update(extra_env)
        completed = self._runner(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=child_env,
            check=False,
        )
        if check and completed.returncode != 0:
            msg = (
                f"{' '.join(argv[:4])} exited {completed.returncode}: "
                f"{_truncate(str(completed.stderr or completed.stdout))}"
            )
            raise LabOverlayRefusalError(msg)
        return completed

    def _stdout(self, argv: Sequence[str], *, timeout: int, **kwargs: Any) -> str:
        return str(self._run(argv, timeout=timeout, **kwargs).stdout or "").strip()

    # -- kubeconfig ---------------------------------------------------------
    def materialise_kubeconfig(self, destination: Path) -> Path:
        """Copy the root-owned k3s admin kubeconfig to a ``0600`` file this
        process can read, so nothing downstream needs ``sudo``.

        ``apply_lab_lane.sh`` calls ``kubectl`` directly, dozens of times, and
        rewriting it to thread a ``sudo`` prefix through every call site is
        exactly the retyped-sequence drift its own header exists to refuse.
        Pointing ``KUBECONFIG`` at a readable copy leaves that script untouched.

        This grants nothing: the account this agent runs as already holds
        passwordless ``sudo`` on the lab host, so the copy materialises access
        it already has. It is removed in the caller's ``finally``.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        uid, gid = os.getuid(), os.getgid()
        self._run(
            [
                "sudo",
                "-n",
                "install",
                "-m",
                "0600",
                "-o",
                str(uid),
                "-g",
                str(gid),
                "/etc/rancher/k3s/k3s.yaml",
                str(destination),
            ],
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        return destination

    # -- overlay source -----------------------------------------------------
    def archive_overlay(self, destination: Path) -> str:
        """Materialise ``omninode_infra@origin/dev`` into ``destination``.

        ``fetch`` then ``archive``, never ``checkout`` or ``pull``: the source is
        a canonical clone shared with every other lane on this host, and
        ``fetch`` writes only refs while ``archive`` writes only into the
        destination. Returns the resolved commit, which the record carries as
        the overlay's lineage.
        """
        self._run(
            ["git", "fetch", "--quiet", "origin", "dev"],
            timeout=SHORT_TIMEOUT_SECONDS,
            cwd=self.overlay_source_dir,
        )
        manifest_sha = self._stdout(
            ["git", "rev-parse", "origin/dev"],
            timeout=SHORT_TIMEOUT_SECONDS,
            cwd=self.overlay_source_dir,
        )
        if len(manifest_sha) != 40:
            msg = f"git rev-parse origin/dev returned {manifest_sha!r}, not a sha"
            raise LabOverlayRefusalError(msg)

        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        # `git archive` into a tar, then extract. NOT a worktree: `git worktree
        # add` registers itself in the shared clone's administrative files, which
        # is a mutation of state every peer lane on this host reads.
        #
        # Written to a file descriptor rather than captured: a tar is bytes, and
        # this class's `_run` decodes stdout as text, which would corrupt it.
        tar_path = destination.parent / f"{destination.name}.tar"
        with open(tar_path, "wb") as handle:
            completed = self._runner(
                ["git", "archive", "--format=tar", manifest_sha],
                stdout=handle,
                stderr=subprocess.PIPE,
                timeout=SHORT_TIMEOUT_SECONDS,
                cwd=str(self.overlay_source_dir),
                env=dict(self.env),
                check=False,
            )
        if completed.returncode != 0:
            msg = f"git archive {manifest_sha[:12]} exited {completed.returncode}"
            raise LabOverlayRefusalError(msg)
        self._run(
            ["tar", "-xf", str(tar_path), "-C", str(destination)],
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        tar_path.unlink(missing_ok=True)
        return manifest_sha

    # -- images -------------------------------------------------------------
    def resident_images(self) -> frozenset[str]:
        """Every image name the k3s content store holds.

        Read with ``sudo -n`` because the socket is root-owned. A read that
        FAILS raises rather than returning an empty set: an empty set would make
        every pin read as garbage-collected, which is rule 16's false zero.
        """
        out = self._stdout(
            ["sudo", "-n", "k3s", "ctr", "-n", "k8s.io", "images", "ls", "-q"],
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        return frozenset(line.strip() for line in out.splitlines() if line.strip())

    def promote_and_import(self, *, source: str, target: str) -> str:
        """Tag ``source`` as ``target``, import it, and return its CONTENT digest.

        The returned value is the k3s content store's digest for ``target``, and
        that choice is measured rather than assumed. A pod's
        ``status.containerStatuses[].imageID`` for an imported image is the OCI
        index digest that containerd holds -- read live on 2026-09-12,
        ``sha256:5e65c13a…`` for the lane's then-current pin, byte-identical to
        the DIGEST column of ``k3s ctr images ls`` for the same ref. It is NOT the
        Docker image id (``docker inspect {{.Id}}``, a config digest), so a
        readback comparing those two values can never match and would report a
        correctly-applied lane as stale on every run.

        Comparing content and not the tag is the point of AC6: a tag can be moved,
        and the 2026-09-11 state -- fifteen Deployments on a three-day-old stamp
        under a green trigger -- was visible only because somebody read a pod.
        """
        self._run(["docker", "tag", source, target], timeout=SHORT_TIMEOUT_SECONDS)
        self._import_into_containerd(target)
        return self.containerd_digest(target)

    def containerd_digest(self, ref: str) -> str:
        """The k3s content store's digest for one ref, or a refusal.

        An absent row is a refusal and never an empty string: an empty digest
        compared against a pod's ``imageID`` would match nothing, which reads as
        a stale lane rather than as a failed read (rule 16).
        """
        out = self._stdout(
            [
                "sudo",
                "-n",
                "k3s",
                "ctr",
                "-n",
                "k8s.io",
                "images",
                "ls",
                f"name=={normalise_image_ref(ref)}",
            ],
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        for line in out.splitlines():
            fields = line.split()
            if (
                len(fields) >= 3
                and fields[0] != "REF"
                and fields[2].startswith("sha256:")
            ):
                return fields[2]
        msg = (
            f"the k3s content store has no digest row for {ref}; "
            f"`images ls` returned {len(out.splitlines())} line(s)"
        )
        raise LabOverlayRefusalError(msg)

    def _import_into_containerd(self, ref: str) -> None:
        """``docker save <ref> | k3s ctr images import -``.

        A pipe rather than a temporary file: the runtime image is multiple
        gigabytes and the lab host's disk pressure has already evicted this
        lane's Postgres once.
        """
        save = self._popen(
            ["docker", "save", ref],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(self.env),
        )
        try:
            completed = self._runner(
                ["sudo", "-n", "k3s", "ctr", "-n", "k8s.io", "images", "import", "-"],
                stdin=save.stdout,
                capture_output=True,
                text=True,
                timeout=IMPORT_TIMEOUT_SECONDS,
                env=dict(self.env),
                check=False,
            )
        finally:
            if save.stdout is not None:
                save.stdout.close()
            save.wait(timeout=SHORT_TIMEOUT_SECONDS)
        if save.returncode != 0:
            msg = f"docker save {ref} exited {save.returncode}"
            raise LabOverlayRefusalError(msg)
        if completed.returncode != 0:
            msg = (
                f"k3s ctr images import of {ref} exited {completed.returncode}: "
                f"{_truncate(str(completed.stderr or ''))}"
            )
            raise LabOverlayRefusalError(msg)

    def build_and_import(
        self,
        *,
        tree: Path,
        dockerfile: str,
        context: str,
        target: str,
        timeout: int,
    ) -> None:
        """Build one image from source and import it into the k3s content store.

        Every image this caller pins goes through here, and every one is rebuilt
        on every run. That is not thrift-blindness; it is the only shape in which
        a pin is guaranteed pullable. See ``_derive_pins``.
        """
        self._run(
            ["docker", "build", "-f", dockerfile, "-t", target, context],
            timeout=timeout,
            cwd=tree,
        )
        self._import_into_containerd(target)

    # -- prune -------------------------------------------------------------
    def declared_deployments(self, render_path: Path) -> frozenset[str]:
        """Every Deployment name the rendered overlay declares.

        Read from the render ``apply_lab_lane.sh`` itself produced, not from a
        list in this file: a hand-kept list is how a prune starts deleting a
        workload somebody legitimately added.
        """
        names: set[str] = set()
        with render_path.open(encoding="utf-8") as handle:
            for doc in yaml.safe_load_all(handle):
                if isinstance(doc, dict) and doc.get("kind") == "Deployment":
                    names.add(str(doc["metadata"]["name"]))
        if not names:
            msg = (
                f"{render_path} declares no Deployment at all; refusing to treat "
                "that as 'everything on the lane is retired'"
            )
            raise LabOverlayRefusalError(msg)
        return frozenset(names)

    def prune_retired_deployments(
        self, kubeconfig: Path, render_path: Path
    ) -> list[str]:
        """Delete lane-managed Deployments the render no longer declares.

        WHY THIS IS NEEDED. ``apply_lab_lane.sh`` applies a document set; it does
        not prune. A workload removed from the manifests therefore keeps running
        on the lane forever, on whatever image it was last given. Measured on the
        lab lane 2026-09-12: after an apply that pinned a fresh image, 16 of 17
        Deployments carried it and the seventeenth was
        ``omnimarket-projection-delegation-writer`` -- the standalone writer
        ``omninode_infra#1355`` had already retired from the manifests -- still on
        a stamp four days old. Without this step the lane runs retired workloads
        and the receipt says PASS, which is the same false green in a new place.

        THE SCOPE IS NARROW AND IT IS A SCOPE, NOT A FILTER ON GOOD INTENTIONS.
        Only Deployments carrying BOTH of the overlay's own management labels are
        considered, so nothing a person or another system put in this namespace is
        in reach, and the render is the sole authority on what survives. A render
        that declares no Deployment is refused rather than read as "retire
        everything" (rule 16: an empty result is not evidence of absence).

        Every deletion is returned so it lands in the record's evidence. A prune
        nobody can read afterwards is a silent mutation.
        """
        declared = self.declared_deployments(render_path)
        raw = self._kubectl(
            [
                "get",
                "deployments",
                "-l",
                LANE_MANAGED_SELECTOR,
                "-o",
                "json",
            ],
            kubeconfig,
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        payload = json.loads(raw) if raw else {"items": []}
        live = {
            str(item["metadata"]["name"])
            for item in payload.get("items", [])
            if isinstance(item, dict)
        }
        retired = sorted(live - declared)
        for name in retired:
            logger.info("lab_overlay: pruning retired Deployment %s", name)
            self._kubectl(
                ["delete", f"deployment/{name}", "--ignore-not-found", "--wait=true"],
                kubeconfig,
                timeout=SHORT_TIMEOUT_SECONDS,
            )
        return retired

    # -- live lane reads ----------------------------------------------------
    def _kubectl(self, args: Sequence[str], kubeconfig: Path, *, timeout: int) -> str:
        return self._stdout(
            ["kubectl", "-n", LAB_NAMESPACE, *args],
            timeout=timeout,
            extra_env={"KUBECONFIG": str(kubeconfig)},
        )

    def running_runtime_images(self, kubeconfig: Path) -> dict[str, ModelRunningImage]:
        """``pod name -> the image reference and imageID`` for Running pods.

        BOTH are read, and only the REFERENCE is compared. See
        ``_readback_checks`` for why the digest cannot be.
        """
        raw = self._kubectl(
            [
                "get",
                "pods",
                "-l",
                RUNTIME_POD_SELECTOR,
                "--field-selector=status.phase=Running",
                "-o",
                "json",
            ],
            kubeconfig,
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        payload = json.loads(raw) if raw else {"items": []}
        running: dict[str, ModelRunningImage] = {}
        for item in payload.get("items", []):
            name = item.get("metadata", {}).get("name", "")
            for status in item.get("status", {}).get("containerStatuses", []) or []:
                if status.get("name") == RUNTIME_DEPLOYMENT:
                    running[name] = ModelRunningImage(
                        reference=str(status.get("image", "")),
                        image_id=str(status.get("imageID", "")),
                    )
        return running

    def pod_omnimarket_version(self, kubeconfig: Path, pod: str) -> str:
        return self._kubectl(
            [
                "exec",
                pod,
                "-c",
                RUNTIME_DEPLOYMENT,
                "--",
                "python",
                "-c",
                OMNIMARKET_VERSION_PROGRAM,
            ],
            kubeconfig,
            timeout=SHORT_TIMEOUT_SECONDS,
        )

    def compose_lane_omnimarket_version(self, container: str) -> str:
        return self._stdout(
            [
                "docker",
                "exec",
                container,
                "python",
                "-c",
                OMNIMARKET_VERSION_PROGRAM,
            ],
            timeout=SHORT_TIMEOUT_SECONDS,
        )

    # -- the apply ----------------------------------------------------------
    def apply(self, *, sha: str, stamp: str, correlation_id: str | None) -> Path:
        """Apply the overlay for one merged sha and write the record.

        Returns the record's path. Raises only on a failure to WRITE the record,
        because an unwritten record is the one outcome worse than a failing one:
        the CI emitter cannot then tell "the lab apply failed" from "the agent
        never got there", which is the distinction the whole surface exists for.

        The phases are ordered so that a refusal costs as little as possible and
        names the thing that is missing: source and pins first (cheap, and the
        two that go stale), then the apply, then the readback that proves the
        lane is running what was pinned.
        """
        started_at = _utc_now_iso()
        checks: list[ModelLabOverlayCheck] = []
        work = self.state_dir / "lab-overlay" / "work"
        kubeconfig = work / "kubeconfig"
        binding = work / "store-binding.env"
        overlay_tree = work / "omninode_infra"
        pins: ModelLabOverlayPins | None = None

        try:
            self.materialise_kubeconfig(kubeconfig)
            manifest_sha = self.archive_overlay(overlay_tree)
            checks.append(
                ModelLabOverlayCheck(
                    name="overlay_source_resolved",
                    ok=True,
                    evidence=(
                        f"git archive omninode_infra@{manifest_sha} from "
                        f"{self.overlay_source_dir} (origin/dev; the overlay's own "
                        "lineage, not the merged omnibase_infra sha)"
                    ),
                )
            )
        except (LabOverlayRefusalError, OSError, subprocess.SubprocessError) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="overlay_source_resolved",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )
            return self._write(
                sha=sha,
                started_at=started_at,
                correlation_id=correlation_id,
                checks=checks,
            )

        try:
            pins = self._derive_pins(
                sha=sha,
                stamp=stamp,
                overlay_tree=overlay_tree,
                manifest_sha=manifest_sha,
            )
            checks.append(
                ModelLabOverlayCheck(
                    name="images_pinned",
                    ok=True,
                    evidence=pins.as_evidence(),
                )
            )
        except (LabOverlayRefusalError, OSError, subprocess.SubprocessError) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="images_pinned",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )
            return self._write(
                sha=sha,
                started_at=started_at,
                correlation_id=correlation_id,
                checks=checks,
            )

        runtime_digest = self._runtime_digest
        try:
            write_store_binding(binding, self.env)
            script = overlay_tree / "k8s" / "onex-lab" / "apply_lab_lane.sh"
            if not script.exists():
                msg = f"{script} does not exist in the archived overlay tree"
                raise LabOverlayRefusalError(msg)
            completed = self._run(
                [
                    "bash",
                    str(script),
                    *pins.apply_arguments(),
                    "--lab-secret-store-env",
                    str(binding),
                    "--lab-secret-store-environment",
                    STORE_BINDING_ENVIRONMENT,
                ],
                timeout=self.budget_seconds,
                cwd=overlay_tree,
                extra_env={
                    "KUBECONFIG": str(kubeconfig),
                    "LAB_NAMESPACE": LAB_NAMESPACE,
                    "LAB_WORK_DIR": str(work / "apply"),
                },
                check=False,
            )
            checks.append(
                ModelLabOverlayCheck(
                    name="lab_overlay_applied",
                    ok=completed.returncode == 0,
                    evidence=(
                        f"k8s/onex-lab/apply_lab_lane.sh exited "
                        f"{completed.returncode} with the four pins and the "
                        f"OMN-18168 store binding ({STORE_BINDING_ENVIRONMENT}); "
                        f"tail: {_truncate(str(completed.stdout or completed.stderr))}"
                    ),
                )
            )
        except (LabOverlayRefusalError, OSError, subprocess.SubprocessError) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="lab_overlay_applied",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )
        finally:
            shred(binding)

        checks.extend(
            self._readback_checks(
                kubeconfig=kubeconfig,
                pins=pins,
                runtime_digest=runtime_digest,
                render_path=work / "apply" / "onex-lab-render.yaml",
            )
        )
        shred(kubeconfig)
        return self._write(
            sha=sha,
            started_at=started_at,
            correlation_id=correlation_id,
            checks=checks,
        )

    def _derive_pins(
        self,
        *,
        sha: str,
        stamp: str,
        overlay_tree: Path,
        manifest_sha: str,
    ) -> ModelLabOverlayPins:
        """Build, import and pin all four images, then verify every one is resident.

        ALL FOUR ARE BUILT OR PROMOTED FROM THE MERGED SHA ON EVERY RUN, AND NONE
        IS EVER CARRIED FORWARD. That is a measured requirement, not a
        preference. **k3s garbage-collects unreferenced images**, and
        ``apply_lab_lane.sh`` deletes the two migrate Jobs on completion, so from
        the moment an apply finishes nothing in the cluster references either
        migrate image and both become collectable. Measured on the lab lane
        2026-09-12: both were resident at a 15:54Z apply and gone by that night,
        leaving only partially-collected unnamed ``import-*`` refs. The apply that
        then pinned what it found passed the preflight, wrote every object, and
        died at the migrate barrier -- the Job could not pull an image that was no
        longer there -- leaving three runtime pods in CrashLoopBackOff on a
        missing ``public.db_metadata`` relation because migrations had never run.

        So a resident image is NOT a pinnable one: residency between applies is
        the thing that is not guaranteed. Scavenging a pin from the ledger, from
        the host's Docker daemon, or from the content store is the same defect
        three ways, and the worst of it is that it *keeps the lane stale and
        passes* -- which is the condition this whole ticket exists to remove.

        A pin that is somehow still not resident after its own build is refused
        HERE, by name. The alternative is an apply that writes every object,
        rolls every Deployment and surfaces minutes later as
        ``ImagePullBackOff`` -- the succeeds-then-fails-later shape
        ``preflight_lab_cluster.sh`` exists to remove for the node address.
        """
        tag = f"{stamp}-{sha[:8]}"
        overlay_tag = f"{manifest_sha[:8]}-{stamp}"
        runtime_image = f"{RUNTIME_IMAGE_NAME}:{tag}"
        infra_migrate_image = f"{INFRA_MIGRATE_IMAGE_NAME}:{tag}"
        cloud_migrate_image = f"{CLOUD_MIGRATE_IMAGE_NAME}:{overlay_tag}"
        api_image = f"{API_IMAGE_NAME}:{overlay_tag}"

        # The runtime image is PROMOTED rather than rebuilt -- this agent just
        # built it for the compose lane from the merged sha, and a second build
        # of the same source would be a different digest for no reason. It is
        # still imported every run, for the same GC reason as the rest.
        self._runtime_digest = self.promote_and_import(
            source=self.env.get("LAB_RUNTIME_SOURCE_IMAGE")
            or "omnibase-infra-omninode-runtime:latest",
            target=runtime_image,
        )
        # omnibase_infra's own tree, already checked out at the merged sha.
        self.build_and_import(
            tree=self.repo_dir,
            dockerfile=MIGRATE_DOCKERFILE,
            context=MIGRATE_CONTEXT,
            target=infra_migrate_image,
            timeout=MIGRATE_BUILD_TIMEOUT_SECONDS,
        )
        # The overlay's tree, so these two carry the overlay's lineage.
        self.build_and_import(
            tree=overlay_tree,
            dockerfile=MIGRATE_DOCKERFILE,
            context=MIGRATE_CONTEXT,
            target=cloud_migrate_image,
            timeout=MIGRATE_BUILD_TIMEOUT_SECONDS,
        )
        self.build_and_import(
            tree=overlay_tree,
            dockerfile=API_DOCKERFILE,
            context=API_CONTEXT,
            target=api_image,
            timeout=API_BUILD_TIMEOUT_SECONDS,
        )

        resident = self.resident_images()
        absent = [
            ref
            for ref in (
                runtime_image,
                infra_migrate_image,
                cloud_migrate_image,
                api_image,
            )
            if normalise_image_ref(ref) not in resident
        ]
        if absent:
            # The positive control leads, and that ordering is deliberate: the
            # evidence field is truncated, and a long pin list would otherwise
            # push the control out of the record -- leaving a zero nobody can
            # tell apart from an unreadable socket (rule 16).
            msg = (
                f"positive control: the k3s content store lists {len(resident)} "
                f"image(s), so this is an absence and not an unreadable socket; "
                f"{len(absent)} pin(s) are absent from it: {', '.join(absent)}"
            )
            raise LabOverlayRefusalError(msg)

        return ModelLabOverlayPins(
            runtime_image=runtime_image,
            api_image=api_image,
            infra_migrate_image=infra_migrate_image,
            cloud_migrate_image=cloud_migrate_image,
            manifest_sha=manifest_sha,
        )

    def _readback_checks(
        self,
        *,
        kubeconfig: Path,
        pins: ModelLabOverlayPins,
        runtime_digest: str,
        render_path: Path,
    ) -> list[ModelLabOverlayCheck]:
        """AC6, as two checks the lane cannot pass while it is stale.

        ``deployed_image`` compares the IMAGE REFERENCE each Running pod reports
        against the pin this run applied. ``runtime_omnimarket_version`` compares
        the lab pod's installed distribution against the compose dev lane's own
        running container, which is the exact discrepancy found by hand on
        2026-09-11 -- compose at 0.4.61, lab at 0.4.30.

        WHY THE REFERENCE AND NOT THE DIGEST, corrected 2026-09-12 by a live run
        rather than by review. The first build of this check compared the pod's
        ``imageID`` against the DIGEST column of ``k3s ctr images ls`` for the
        applied pin, on a measurement showing the two were byte-identical. They
        are identical only for an image imported from an OCI archive carrying an
        index, which is what the lane's previous hand-applied pin happened to be.
        For an image imported from ``docker save``, that column reports the
        CONFIG digest -- the same value ``docker inspect {{.Id}}`` returns --
        while the kubelet reports the manifest digest, and the two never match.
        Measured on the first real run: the pod ran the correct freshly applied
        pin and reported ``sha256:5f0b4623...`` while the content store reported
        ``sha256:b0f56ab5...`` for the same tag. The check FAILED on a correctly
        applied lane.

        That is a false red, and a false red on this check is not harmless: the
        receipt's verdict is derived from its checks, so every receipt would have
        been a FAIL and the gate would have taught its readers to ignore it.

        The reference is comparable and is not weaker here. The tag is minted
        fresh on every run from this run's own timestamp and the merged sha
        (``_derive_pins``), so it cannot be a tag some earlier build also carried
        -- which is the reuse a tag comparison is normally weak against. Both
        digests are still RECORDED in the evidence, because they are useful to a
        reader; they are simply not asserted equal, since they answer different
        questions.
        """
        checks: list[ModelLabOverlayCheck] = []
        try:
            running = self.running_runtime_images(kubeconfig)
            if not running:
                msg = (
                    f"no Running {RUNTIME_DEPLOYMENT} pod in {LAB_NAMESPACE}, so the "
                    "lane's image cannot be read back"
                )
                raise LabOverlayRefusalError(msg)
            expected = normalise_image_ref(pins.runtime_image)
            mismatched = {
                pod: image.reference
                for pod, image in running.items()
                if normalise_image_ref(image.reference) != expected
            }
            # The DIGEST portion, not the first N characters of the whole
            # reference: a pod's imageID is often `<registry>/<name>@sha256:...`,
            # so a blind prefix records the registry and truncates the one part a
            # reader wants.
            digests = ", ".join(
                f"{pod}={image.image_id.rsplit('@', 1)[-1][:19]}..."
                for pod, image in sorted(running.items())
            )
            checks.append(
                ModelLabOverlayCheck(
                    name="deployed_image",
                    ok=not mismatched,
                    evidence=(
                        f"{len(running)} Running {RUNTIME_DEPLOYMENT} pod(s); expected "
                        f"{pins.runtime_image}; "
                        + (
                            f"{len(mismatched)} on another image: {sorted(mismatched.values())}"
                            if mismatched
                            else "every pod runs it"
                        )
                        + f". Content digests recorded, not compared: applied pin carries "
                        f"{runtime_digest[:26]}... in the k3s content store; pods report "
                        f"{digests}"
                    ),
                )
            )
        except (
            LabOverlayRefusalError,
            OSError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="deployed_image",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )
            running = {}

        try:
            if not running:
                msg = "no Running pod to read a version from"
                raise LabOverlayRefusalError(msg)
            pod = sorted(running)[0]
            lab_version = self.pod_omnimarket_version(kubeconfig, pod)
            compose_version = self.compose_lane_omnimarket_version(
                self.env.get("LAB_COMPOSE_RUNTIME_CONTAINER") or "omninode-runtime"
            )
            checks.append(
                ModelLabOverlayCheck(
                    name="runtime_omnimarket_version",
                    ok=bool(lab_version) and lab_version == compose_version,
                    evidence=(
                        f"omnimarket in {pod} on the k3s lab lane = "
                        f"{lab_version or '(unreadable)'}; in the compose dev lane's "
                        f"running container = {compose_version or '(unreadable)'}"
                    ),
                )
            )
        except (
            LabOverlayRefusalError,
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="runtime_omnimarket_version",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )

        # The prune runs LAST and reports what it deleted. Ordering is not
        # arbitrary: pruning before the image readback would remove the very
        # evidence that a Deployment was running the wrong image.
        try:
            retired = self.prune_retired_deployments(kubeconfig, render_path)
            checks.append(
                ModelLabOverlayCheck(
                    name="retired_workloads_pruned",
                    ok=True,
                    evidence=(
                        f"lane-managed Deployments absent from {render_path.name} "
                        + (
                            f"and deleted: {retired}"
                            if retired
                            else "and deleted: none -- the lane declares exactly "
                            "what the render declares"
                        )
                    ),
                )
            )
        except (
            LabOverlayRefusalError,
            OSError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="retired_workloads_pruned",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )
        return checks

    def _write(
        self,
        *,
        sha: str,
        started_at: str,
        correlation_id: str | None,
        checks: Sequence[ModelLabOverlayCheck],
    ) -> Path:
        """Write the record atomically, on every outcome.

        A temporary file plus ``os.replace`` so the HTTP reader can never observe
        a half-written record and report it as malformed -- which the reader is
        required to treat as a finding rather than as an absence.
        """
        record = ModelLabOverlayRecord(
            sha=sha,
            lane=LAB_LANE_VALUE,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            agent_command_id=correlation_id,
            checks=tuple(checks),
        )
        path = record_path(self.state_dir, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.info(
            "lab_overlay: wrote %s (%d check(s), %d failing)",
            path,
            len(checks),
            sum(1 for check in checks if not check.ok),
        )
        return path
