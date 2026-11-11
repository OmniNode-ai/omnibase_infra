# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's overlay-tree builds for the compose dev lane.

The cluster apply this module used to run for every passing dev roll is retired
and deleted (operator ruling 2026-11-10, OMN-23521); its cluster's control
service is dead. What remains here builds images onto the host Docker daemon,
which is what the compose dev lane reads:

* ``capture_compose_inputs`` -- under the dev lane lock, the runtime image is
  tagged by a unique name so the settle builds against THIS job's build and not
  the next one's (OMN-19501);
* ``archive_overlay`` -- ``git fetch`` then ``git archive`` of the overlay
  source's ``origin/dev``, never a checkout of the shared clone;
* ``build_repair_migrate_image`` -- the cloud-migrate repair build on a failed
  dev roll (OMN-18545), with its own sha-keyed record.

The onex-api image the pin delivery advances to is built by
``omnibase_infra.handlers.handler_onex_api_image_build`` from the archived tree
(OMN-23521); this module only supplies the tree and the capture.
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

#: OMN-18113. OCI provenance stamped on the api image, naming the omninode_infra
#: commit its source came from. The tag already carries an abbreviation of the
#: same sha; the label is what survives a retag, and it is what
#: ``scripts/runtime_build/repoint_dev_lane_onex_api.py`` cross-checks the tag
#: against before it writes the compose lane's pin.
API_REVISION_LABEL = "org.opencontainers.image.revision"
API_SOURCE_LABEL = "ai.omninode.image.source-repo"

#: OMN-18354. The build argument ``docker/onex-api/Dockerfile`` stage 0 requires
#: and gives no default to. Named here rather than spelled at the call site so
#: the string that must equal the Dockerfile's own ``ARG`` has one home.
API_RUNTIME_IMAGE_BUILD_ARG = "RUNTIME_IMAGE"

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

#: Bounds for the short host commands. Generous for a loaded 32-core host, and
#: short enough that a hung probe surfaces as a failing check inside the outer
#: budget rather than consuming it.
SHORT_TIMEOUT_SECONDS = 120
#: The two migrate bundles are ``alpine`` plus ``COPY`` layers.
MIGRATE_BUILD_TIMEOUT_SECONDS = 600
#: ``onex-api`` is a two-stage ``python:3.12-slim`` build with a pip install, so
#: it is minutes rather than seconds on a cold layer cache and fast on a warm
#: one. It is still built EVERY run, so a pin never names a stale image.
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
class ModelLabOverlayCapture:
    """The compose-lane inputs of one apply, taken while the job holds the lane.

    OMN-19501. The apply now runs AFTER the dev lane lock is released, so the
    next compose job may by then be rebuilding
    ``omnibase-infra-omninode-runtime:latest``, resetting the deploy-source
    clone and recreating ``omninode-runtime``. Everything the apply used to read
    off that lane is read once here instead, under the lock, by
    ``LabOverlayApplier.capture_compose_inputs``:

    * ``runtime_image`` -- the unique pin tag already pointing at THIS job's
      build (``docker tag`` of ``:latest`` at the verdict);
    * ``infra_migrate_image`` -- built from the deploy-source clone while it
      is still at this job's sha, not yet imported into k3s;
    * ``compose_omnimarket_version`` -- what the compose lane ran for this sha,
      the value the ``runtime_omnimarket_version`` readback compares against.

    A capture never raises. ``pin_error`` and ``compose_version_error`` carry a
    failure to the record as a failing check, so the ``onex-lab-k3s`` receipt
    reads FAIL rather than "missing".
    """

    sha: str
    stamp: str
    runtime_image: str
    infra_migrate_image: str
    pin_error: str | None
    compose_omnimarket_version: str | None
    compose_version_error: str | None


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


def repair_record_path(state_dir: Path, sha: str) -> Path:
    """Where one sha's REPAIR record lives, and why it is not beside the other.

    OMN-18545. A repair record is not a lab-overlay record: the lane was never
    advanced, only an image was built. Writing it to ``record_path`` would put it
    at the same key an apply uses, and ``_write`` replaces unconditionally -- so
    a second job resolving the same dev HEAD and failing would OVERWRITE an
    earlier job's passing apply record with one asserting the lane was not
    advanced to that sha. That is not a stale record, it is a false one, and it
    would supersede a PASS receipt with a FAIL on the next emit.

    A SUBDIRECTORY, deliberately. ``load_latest_record`` globs ``*.json``
    non-recursively directly under ``lab-overlay/``, so nothing under here can
    become "the latest applied record" -- which matters because the OMN-18399
    descendant-tolerance fallback reads that answer and labels it *applied*. A
    repair record answering that question would make the fallback assert an apply
    that never happened.
    """
    return Path(state_dir) / "lab-overlay" / "repair" / f"{sha}.json"


def load_repair_record(state_dir: Path, sha: str) -> dict[str, Any] | None:
    """Read one repair record back, or ``None`` when none exists.

    Same malformed-raises contract as ``load_record``, for the same reason.
    """
    path = repair_record_path(state_dir, sha)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path}: a record must be a JSON object"
        raise ValueError(msg)
    return payload


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


def load_latest_record(state_dir: Path) -> dict[str, Any] | None:
    """Read the most recently written lab-overlay record, across all shas.

    OMN-18399. ``load_record`` answers "what is the record for THIS sha", which
    is the fast path and stays unchanged. Under a busy ``dev`` branch this
    agent's own ``_current_git_sha`` can resolve to a LATER head than the
    command that triggered a given run (``agent.py`` pulls whatever ``dev``
    HEAD is current when it reaches the job, not the triggering command's sha),
    so an intermediate merge sha's exact record is never written at all -- the
    same coalescing ``check_dev_lane_staleness.py`` already tolerates for the
    compose lane (OMN-18388). This function gives the CI reader something to
    compare the requested sha against: "is the requested sha an ancestor of the
    most recently applied one", resolved by the caller via GitHub's compare
    API, because the CI job's own checkout is depth-1 and cannot answer that
    locally.

    A directory listing, not an index: every ``*.json`` file directly under
    ``lab-overlay/`` (the ``work/`` subdirectory used during an apply is a
    directory, not a ``.json`` file at this level, so a non-recursive glob
    never descends into it), sorted by ``finished_at`` (ISO-8601 UTC, and
    therefore string-sortable) ascending, returning the last one.

    A malformed record among them raises, same contract as ``load_record``:
    "a record exists and is unreadable" and "no record exists" are different
    facts, and silently skipping a malformed file could report an OLDER
    record as the latest one -- the opposite of what this function is for.
    """
    directory = Path(state_dir) / "lab-overlay"
    if not directory.is_dir():
        return None
    candidates: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"{path}: a record must be a JSON object"
            raise ValueError(msg)
        candidates.append(payload)
    if not candidates:
        return None
    candidates.sort(key=lambda record: str(record.get("finished_at", "")))
    return candidates[-1]


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
        runner: Any = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.repo_dir = Path(repo_dir)
        self.overlay_source_dir = Path(overlay_source_dir)
        self.env = dict(env)
        self._runner = runner or subprocess.run

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

    def capture_compose_inputs(self, *, sha: str, stamp: str) -> ModelLabOverlayCapture:
        """Take every compose-lane input of the apply, under the lane lock (OMN-19501).

        The caller holds the dev lane lock; this is the only part of the overlay
        that must, and it is short: one ``docker tag``, one ``alpine`` plus
        ``COPY`` build, one ``docker exec``. Nothing is imported into k3s here,
        nothing reads the overlay source, and nothing touches the cluster -- that
        is the minutes-long part, and it runs after the lane is released.

        The runtime image is TAGGED to this run's unique pin name rather than
        read later from ``:latest``, because after the release ``:latest`` names
        whatever the next job builds. The extended TLA+ model on OMN-19501 shows
        the cost of reading it late (``OverlayPromotesOwnBuild`` violated): the
        overlay promotes the next job's build under this job's sha.

        Never raises; failures are carried on the capture (see its docstring).
        """
        tag = f"{stamp}-{sha[:8]}"
        runtime_image = f"{RUNTIME_IMAGE_NAME}:{tag}"
        infra_migrate_image = f"{INFRA_MIGRATE_IMAGE_NAME}:{tag}"
        pin_error: str | None = None
        try:
            self._run(
                [
                    "docker",
                    "tag",
                    self.env.get("LAB_RUNTIME_SOURCE_IMAGE")
                    or "omnibase-infra-omninode-runtime:latest",
                    runtime_image,
                ],
                timeout=SHORT_TIMEOUT_SECONDS,
            )
            # omnibase_infra's own tree, still at the merged sha while the lock
            # is held.
            self.build_and_import(
                tree=self.repo_dir,
                dockerfile=MIGRATE_DOCKERFILE,
                context=MIGRATE_CONTEXT,
                target=infra_migrate_image,
                timeout=MIGRATE_BUILD_TIMEOUT_SECONDS,
            )
        except (LabOverlayRefusalError, OSError, subprocess.SubprocessError) as exc:
            pin_error = f"{type(exc).__name__}: {_truncate(str(exc))}"

        compose_version: str | None = None
        compose_version_error: str | None = None
        try:
            compose_version = self.compose_lane_omnimarket_version(
                self.env.get("LAB_COMPOSE_RUNTIME_CONTAINER") or "omninode-runtime"
            )
        except (LabOverlayRefusalError, OSError, subprocess.SubprocessError) as exc:
            compose_version_error = f"{type(exc).__name__}: {_truncate(str(exc))}"

        return ModelLabOverlayCapture(
            sha=sha,
            stamp=stamp,
            runtime_image=runtime_image,
            infra_migrate_image=infra_migrate_image,
            pin_error=pin_error,
            compose_omnimarket_version=compose_version,
            compose_version_error=compose_version_error,
        )

    def build_and_import(
        self,
        *,
        tree: Path,
        dockerfile: str,
        context: str,
        target: str,
        timeout: int,
    ) -> None:
        """Build one image from source onto the host Docker daemon.

        Every caller wants the image where the compose dev lane reads it, the
        host daemon, and nothing imports it anywhere else (OMN-23521).
        """
        self._run(
            ["docker", "build", "-f", dockerfile, "-t", target, context],
            timeout=timeout,
            cwd=tree,
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

    def build_repair_migrate_image(
        self, *, sha: str, stamp: str, correlation_id: str | None
    ) -> Path:
        """Build the cloud-migrate image ALONE, for a deploy job that failed.

        OMN-18545. The compose dev lane runs whatever ``ONEX_CLOUD_MIGRATE_IMAGE``
        names; the dev-lane migration preflight requires the cloud-migration
        one-shots to exit 0 USING THAT IMAGE; and this module is the only thing
        in the repository that builds a replacement from the overlay tree. While
        the pinned image was broken the preflight raised, the deploy job failed,
        and ``apply`` never ran -- so the agent could not produce the image that
        would let the preflight pass. This is the narrow path out of that loop.

        WHY THIS IS NOT JUST ``apply`` ON THE FAILING PATH. ``apply`` PROMOTES
        the runtime pin (``_derive_pins``: the captured tag of
        ``omnibase-infra-omninode-runtime:latest``) on the premise that "this
        agent just built it for the compose lane from the merged sha". On a
        FAILED job that premise can be false -- a job that dies before or during
        the image build leaves that tag at the PREVIOUS deploy's image -- and
        the readback checks cannot catch it, because ``deployed_image`` compares
        the lane against the tag this run just minted and
        ``runtime_omnimarket_version`` compares it against the compose lane,
        which on a failed deploy is also still the old image. Running the full
        apply on failure would therefore roll the persistent k3s lane to a tag
        NAMING the merged sha while it ran the previous commit's binary, and
        report PASS. That is the false green this whole surface exists to
        remove, so the failing path does not apply the lane at all: it builds
        one image and touches nothing else.

        It is also the cheap half. ``apply`` is four imports plus a lane apply,
        bounded at roughly two hours, and it runs under the agent's single-flight
        lock -- which on a broken migrate image would be taken on EVERY merge,
        since every merge fails. This is one ``alpine``-plus-two-``COPY`` build.

        THE RECORD LIVES UNDER ITS OWN KEY, and is deliberately a failing one.
        It carries the same ``{name, ok, evidence}`` shape ``apply`` writes, so a
        reader diagnosing a stuck lane has the build's outcome and evidence
        durably -- but it goes to ``repair_record_path``, not ``record_path``,
        because a repair is not an apply and writing it to the apply's key would
        overwrite a passing record for the same sha with a false one. That
        docstring carries the full argument. ``lab_overlay_applied`` is recorded
        ``ok=False`` unconditionally on top of that, so a repair record can never
        be read as a lab pass even by something that finds it directly.
        """
        started_at = _utc_now_iso()
        checks: list[ModelLabOverlayCheck] = []
        work = self.state_dir / "lab-overlay" / "work"
        overlay_tree = work / "omninode_infra"

        try:
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
            checks.append(self._repair_lane_not_applied_check())
            return self._write_repair(
                sha=sha,
                started_at=started_at,
                correlation_id=correlation_id,
                checks=checks,
            )

        # The same tag shape a successful apply would have minted, so the
        # operator pins one name whichever path produced the image.
        target = f"{CLOUD_MIGRATE_IMAGE_NAME}:{manifest_sha[:8]}-{stamp}"
        try:
            self.build_and_import(
                tree=overlay_tree,
                dockerfile=MIGRATE_DOCKERFILE,
                context=MIGRATE_CONTEXT,
                target=target,
                timeout=MIGRATE_BUILD_TIMEOUT_SECONDS,
            )
            checks.append(
                ModelLabOverlayCheck(
                    name="cloud_migrate_repair_image_built",
                    ok=True,
                    evidence=(
                        f"docker build -f {MIGRATE_DOCKERFILE} -t {target} "
                        f"{MIGRATE_CONTEXT} from omninode_infra@{manifest_sha}; "
                        "resident on the host Docker daemon. It is NOT delivered "
                        "to the lane: ONEX_CLOUD_MIGRATE_IMAGE is operator-held "
                        "and nothing in this repository writes it."
                    ),
                )
            )
        except (LabOverlayRefusalError, OSError, subprocess.SubprocessError) as exc:
            checks.append(
                ModelLabOverlayCheck(
                    name="cloud_migrate_repair_image_built",
                    ok=False,
                    evidence=f"{type(exc).__name__}: {_truncate(str(exc))}",
                )
            )

        checks.append(self._repair_lane_not_applied_check())
        return self._write_repair(
            sha=sha,
            started_at=started_at,
            correlation_id=correlation_id,
            checks=checks,
        )

    @staticmethod
    def _repair_lane_not_applied_check() -> ModelLabOverlayCheck:
        """The check that keeps a repair record from reading as a lab pass.

        Always failing, by construction. The receipt derives PASS iff every check
        passed, so a repair record must carry one that did not: the k3s lane was
        never advanced to this sha, and a record saying otherwise would be a
        worse lie than no record at all.
        """
        return ModelLabOverlayCheck(
            name="lab_overlay_applied",
            ok=False,
            evidence=(
                "not attempted: the compose deploy job for this sha FAILED, so "
                "the k3s onex-lab lane was deliberately not advanced to it "
                "(OMN-18545). This run built the cloud-migrate repair image "
                "only. The lane is still running whatever it ran before."
            ),
        )

    def _write_repair(
        self,
        *,
        sha: str,
        started_at: str,
        correlation_id: str | None,
        checks: Sequence[ModelLabOverlayCheck],
    ) -> Path:
        """Write a repair record, under its own key (OMN-18545).

        See ``repair_record_path`` for why it must not share the apply's key.
        """
        return self._write_at(
            repair_record_path(self.state_dir, sha),
            sha=sha,
            started_at=started_at,
            correlation_id=correlation_id,
            checks=checks,
        )

    def _write_at(
        self,
        path: Path,
        *,
        sha: str,
        started_at: str,
        correlation_id: str | None,
        checks: Sequence[ModelLabOverlayCheck],
    ) -> Path:
        record = ModelLabOverlayRecord(
            sha=sha,
            lane=LAB_LANE_VALUE,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            agent_command_id=correlation_id,
            checks=tuple(checks),
        )
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
