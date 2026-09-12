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

THE FOUR IMAGE PINS
-------------------
``apply_lab_lane.sh`` commits no digest by design (OMN-17533 AC-5) and requires
the caller to pin all four images. Three of the four are derived by this
pipeline and one is read off the live lane:

``--runtime-image``
    ``docker tag`` of the image this agent just built, promoted into the
    ``onex-lab/`` name space and imported into the k3s content store. Never
    rebuilt -- a second build of the same source would be a different digest
    for no reason.
``--infra-migrate-image``
    Built here from ``docker/Dockerfile.migrate`` in this agent's own clone,
    which is already checked out at the merged sha. It is an ``alpine`` image
    whose only layers are two ``COPY`` directives, so the build is seconds. It
    is rebuilt EVERY run rather than carried forward, because it is the
    OMN-17702 barrier: a migrate bundle whose lineage lags the runtime image is
    precisely the failure that barrier exists to catch.
``--cloud-migrate-image``
    Built here the same way from ``omninode_infra``'s ``docker/Dockerfile.migrate``
    in the archived overlay tree, so its lineage is the overlay's lineage.
``--api-image``
    READ OFF THE LIVE ``onex-api`` Deployment. This is the one pin this module
    does not produce, stated rather than hidden: ``onex-api`` is a full
    application build from ``omninode_infra``, and this trigger fires on
    ``omnibase_infra`` merges, so building it here would rebuild a large image
    on every merge of an unrelated repository. Carried forward, and verified
    resident in the k3s content store, so a pin that has been garbage-collected
    is a named refusal rather than an ``ImagePullBackOff`` ten minutes later.

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

#: The ``onex-lab/`` names the three built-or-promoted pins are published under.
#: ``docker.io/`` is prepended when the reference is compared against the k3s
#: content store, which normalises every unqualified name to that registry.
RUNTIME_IMAGE_NAME = "onex-lab/omninode-runtime"
INFRA_MIGRATE_IMAGE_NAME = "onex-lab/omnibase-infra-migrate"
CLOUD_MIGRATE_IMAGE_NAME = "onex-lab/omninode-cloud-migrate"

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

    def build_migrate_bundle(self, *, context: Path, target: str) -> None:
        """Build one migrate bundle and import it.

        Both bundles are ``alpine`` plus ``COPY`` layers, so this is seconds, and
        both are rebuilt every run on purpose: the migrate image is the
        OMN-17702 barrier the runtime image's entrypoint dies without, and a
        bundle carried forward from an older lineage than the runtime it gates
        is the exact condition the barrier exists to refuse.
        """
        self._run(
            [
                "docker",
                "build",
                "-f",
                "docker/Dockerfile.migrate",
                "-t",
                target,
                ".",
            ],
            timeout=MIGRATE_BUILD_TIMEOUT_SECONDS,
            cwd=context,
        )
        self._import_into_containerd(target)

    # -- live lane reads ----------------------------------------------------
    def _kubectl(self, args: Sequence[str], kubeconfig: Path, *, timeout: int) -> str:
        return self._stdout(
            ["kubectl", "-n", LAB_NAMESPACE, *args],
            timeout=timeout,
            extra_env={"KUBECONFIG": str(kubeconfig)},
        )

    def carried_forward_api_pin(self, kubeconfig: Path) -> str:
        """The ``--api-image`` pin, read off the live ``onex-api`` Deployment.

        A blank result is a refusal, not a default. There is no fallback image:
        an unpinned lane is a lane whose verdict names no candidate, which is
        the rule ``apply_lab_lane.sh`` states for all four of its arguments.
        """
        ref = self._kubectl(
            [
                "get",
                f"deployment/{API_DEPLOYMENT}",
                "-o",
                "jsonpath={.spec.template.spec.containers[0].image}",
            ],
            kubeconfig,
            timeout=SHORT_TIMEOUT_SECONDS,
        )
        if not ref:
            msg = (
                f"the live {API_DEPLOYMENT} Deployment in {LAB_NAMESPACE} declares no "
                "image, so the api pin cannot be carried forward. Apply the lane "
                "once by hand to establish it."
            )
            raise LabOverlayRefusalError(msg)
        return ref

    def running_runtime_image_ids(self, kubeconfig: Path) -> dict[str, str]:
        """``pod name -> imageID`` for the runtime Deployment's Running pods."""
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
        ids: dict[str, str] = {}
        for item in payload.get("items", []):
            name = item.get("metadata", {}).get("name", "")
            for status in item.get("status", {}).get("containerStatuses", []) or []:
                if status.get("name") == RUNTIME_DEPLOYMENT:
                    ids[name] = str(status.get("imageID", ""))
        return ids

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
                kubeconfig=kubeconfig,
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
        kubeconfig: Path,
        overlay_tree: Path,
        manifest_sha: str,
    ) -> ModelLabOverlayPins:
        """Produce the four pins, verifying every one is resident before the apply.

        A pin that is not in the k3s content store is refused HERE, by name. The
        alternative is an apply that writes every object, rolls every Deployment
        and surfaces minutes later as ``ImagePullBackOff`` -- the same
        succeeds-then-fails-later shape ``preflight_lab_cluster.sh`` exists to
        remove for the node address.
        """
        tag = f"{stamp}-{sha[:8]}"
        runtime_image = f"{RUNTIME_IMAGE_NAME}:{tag}"
        infra_migrate_image = f"{INFRA_MIGRATE_IMAGE_NAME}:{tag}"
        cloud_migrate_image = f"{CLOUD_MIGRATE_IMAGE_NAME}:{manifest_sha[:8]}-{stamp}"

        self._runtime_digest = self.promote_and_import(
            source=self.env.get("LAB_RUNTIME_SOURCE_IMAGE")
            or "omnibase-infra-omninode-runtime:latest",
            target=runtime_image,
        )
        self.build_migrate_bundle(context=self.repo_dir, target=infra_migrate_image)
        self.build_migrate_bundle(context=overlay_tree, target=cloud_migrate_image)
        api_image = self.carried_forward_api_pin(kubeconfig)

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
    ) -> list[ModelLabOverlayCheck]:
        """AC6, as two checks the lane cannot pass while it is stale.

        ``deployed_image`` compares by image ID, not by tag: two builds can carry
        one tag, and the 2026-09-11 state was a lane pinned to a three-day-old
        stamp with a green trigger above it. ``runtime_omnimarket_version``
        compares the lab pod against the compose dev lane's own running
        container, which is the exact discrepancy that was found by hand --
        compose at 0.4.61, lab at 0.4.30.
        """
        checks: list[ModelLabOverlayCheck] = []
        try:
            running = self.running_runtime_image_ids(kubeconfig)
            if not running:
                msg = (
                    f"no Running {RUNTIME_DEPLOYMENT} pod in {LAB_NAMESPACE}, so the "
                    "lane's image cannot be read back"
                )
                raise LabOverlayRefusalError(msg)
            # An empty expected digest would make `in` true for every pod and
            # turn this check into a vacuous PASS -- the readback reached
            # without a promotion having happened. Refuse it explicitly.
            expected = runtime_digest.split(":")[-1]
            if not expected:
                msg = (
                    "no runtime digest was recorded for this run, so there is "
                    "nothing to compare the lane's pods against"
                )
                raise LabOverlayRefusalError(msg)
            mismatched = {
                pod: image_id
                for pod, image_id in running.items()
                if expected not in image_id
            }
            checks.append(
                ModelLabOverlayCheck(
                    name="deployed_image",
                    ok=not mismatched,
                    evidence=(
                        f"{len(running)} Running {RUNTIME_DEPLOYMENT} pod(s); "
                        f"expected the promoted {pins.runtime_image} "
                        f"(k3s content digest {runtime_digest[:19]}...); "
                        + (
                            f"{len(mismatched)} still on another image: "
                            f"{sorted(mismatched)}"
                            if mismatched
                            else "every pod matches"
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
