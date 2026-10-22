# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-lane branch installs: a declared record the deploy agent re-applies (OMN-17427).

THE FAILURE, MEASURED THREE TIMES ON 2026-10-22
------------------------------------------------
A lane that fixes code the dev lane runs does not wait for the merge: it installs
the fixed file from its branch into the running container and restarts the
process (operator ruling 2026-10-22T12:12:56Z). The file lives in the container's
writable layer. The next dev refresh recreates the container from the image and
the layer, with the fix in it, is gone. The hook-event feed fix (omnimarket#4304
and #4308, omnibase_infra#5269) was lost when its container was recreated around
03:08Z. The consumer-flow writer (omnimarket#4312, sha256 17d6b307) was dropped by
the 11:18Z refresh, reinstalled by hand at 11:31:57Z, and dropped again at 12:39Z.
Nothing recorded that a fix was installed, so nothing could put it back or even
say it was missing.

WHAT THIS IS
------------
A branch install is a declared record, kept in the deploy agent's state directory
and not in the repository, because it has to exist BEFORE the merge:

* the repository, PR, branch and branch head it comes from, and the ticket and
  lane that registered it;
* the compose services whose containers carry it and, for each file, the path
  inside the container and its sha256. The bytes are kept beside the record,
  content addressed, so a re-apply needs neither the lane's worktree nor GitHub;
* an expiry, declared by the registrar and at most seven days out.

The deploy agent runs ``apply`` after every dev job that verified the lane and
on an idle tick (``check``, then ``apply`` under the lane lock when anything is missing), which also covers a recreate the agent did not do (a hand
refresh, the settle worker's pin recreate, a rollback). An apply copies a file
whose sha256 differs, restores its owner and mode, restarts a changed container
once, then READS EVERY FILE BACK and compares its sha256. A record whose files do
not read back is FAILED, never quiet.

RELEASE RULE: MERGED OR EXPIRED
-------------------------------
A record is dropped when its PR is merged and the lane's running image already
carries the merge (so dropping it cannot regress the lane), or when it expires.
The drop is a line in ``released.jsonl`` with the reason. An expired record's
files are left where they are; the next recreate removes them and nothing
re-applies them.

DETECTOR
--------
``check`` is read only. Every recorded install whose files are not in the running
container is one ``BRANCH-INSTALL-MISSING`` line on stderr, which the agent relays
into its job log (``friction_type=branch_install_missing``), and the same facts
are written to ``status.json``. An unreadable container is MISSING, never OK.

DEV LANE ONLY
-------------
The handler takes a compose project and refuses one that names prod or stability.
A prod image is promoted through the gated node path with a fresh approval
(OMN-13418); a file is never copied into a prod container by a record.

NO DEFAULTS (rule 8). The state directory and the compose project are always
named by the caller.

Command line::

    python -m omnibase_infra.handlers.handler_dev_lane_branch_install register \\
        --state-dir D --lane dev --id ID --repo omnimarket --pr 4312 --branch B \\
        --head SHA --ticket OMN-n --registered-by LANE --service omninode-runtime \\
        --expires-in-hours 48 --file /app/.../handler.py=/local/copy/handler.py
    python -m omnibase_infra.handlers.handler_dev_lane_branch_install check \\
        --state-dir D --compose-project P
    python -m omnibase_infra.handlers.handler_dev_lane_branch_install apply|release-due \\
        --state-dir D --compose-project P [--clone REPO=PATH] [--running-ref REPO=SHA]
    python -m omnibase_infra.handlers.handler_dev_lane_branch_install release \\
        --state-dir D --id ID --reason R
    python -m omnibase_infra.handlers.handler_dev_lane_branch_install list --state-dir D
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_dev_lane_branch_install_file_state import (
    EnumDevLaneBranchInstallFileState,
)
from omnibase_infra.enums.enum_dev_lane_branch_install_release_reason import (
    EnumDevLaneBranchInstallReleaseReason,
)
from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.models.model_dev_lane_branch_install import (
    COMMIT_PATTERN,
    ModelDevLaneBranchInstall,
)
from omnibase_infra.models.model_dev_lane_branch_install_file import (
    SHA256_PATTERN,
    ModelDevLaneBranchInstallFile,
)
from omnibase_infra.models.model_dev_lane_branch_install_file_readback import (
    ModelDevLaneBranchInstallFileReadback,
)
from omnibase_infra.models.model_dev_lane_branch_install_readback import (
    ModelDevLaneBranchInstallReadback,
)
from omnibase_infra.models.model_dev_lane_branch_install_receipt import (
    ModelDevLaneBranchInstallReceipt,
)
from omnibase_infra.models.model_dev_lane_branch_install_records import (
    SCHEMA,
    ModelDevLaneBranchInstallRecords,
)
from omnibase_infra.models.model_dev_lane_branch_install_release import (
    ModelDevLaneBranchInstallRelease,
)

logger = logging.getLogger(__name__)

#: The one lane a branch install is registered on.
DEV_LANE = "dev"
#: A record lives at most this long past its registration.
MAX_LIFETIME = timedelta(days=7)
#: A compose project that names one of these is not the dev lane.
GOVERNED_PROJECT_MARKERS = ("prod", "stability")
#: How many commits of a repository's dev branch are searched for a squash merge.
MERGE_SEARCH_DEPTH = 3000
DEV_BRANCH_REF = "origin/dev"

STORE_DIRNAME = "branch-installs"
RECORDS_FILE = "records.json"
LOCK_FILE = "records.lock"
RELEASED_FILE = "released.jsonl"
STATUS_FILE = "status.json"
CONTENT_DIRNAME = "content"

DOCKER_TIMEOUT_SECONDS = 60.0
RESTART_TIMEOUT_SECONDS = 240.0
RESTART_WAIT_SECONDS = 240.0
RESTART_POLL_SECONDS = 3.0

EXIT_OK = 0
EXIT_REFUSED = 3
EXIT_NOT_WHOLE = 4

#: Runs one argv with a timeout; the default captures text and never raises on a
#: non-zero exit. Tests and the agent pass their own.
Runner = Callable[..., "subprocess.CompletedProcess[str]"]
Sleeper = Callable[[float], None]
#: ``None`` while the install's PR is not merged into what the lane runs;
#: otherwise the evidence, for the release line.
MergedProbe = Callable[[ModelDevLaneBranchInstall], str | None]
Guard = Callable[[], contextlib.AbstractContextManager[object]]

_DEFAULT_RUN: Runner = functools.partial(
    subprocess.run, capture_output=True, text=True, check=False
)

_WHOLE = (
    EnumDevLaneBranchInstallFileState.PRESENT,
    EnumDevLaneBranchInstallFileState.REAPPLIED,
)


class DevLaneBranchInstallRefusedError(Exception):
    """A condition under which nothing is written or copied; the message names it."""


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short(sha: str) -> str:
    return sha[:12] if sha else "absent"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_dev_project(compose_project: str) -> None:
    """Fail closed on anything that is not the dev lane's compose project."""
    if not compose_project or any(
        marker in compose_project.lower() for marker in GOVERNED_PROJECT_MARKERS
    ):
        raise DevLaneBranchInstallRefusedError(
            f"compose project {compose_project!r} is not the dev lane: a branch "
            "install is never applied to prod or stability (OMN-13418)"
        )


class DevLaneBranchInstallBlobs:
    """The bytes of every declared file, kept by sha256 beside the record."""

    def __init__(self, content_dir: Path) -> None:
        self.content = content_dir

    def path(self, sha256: str) -> Path:
        return self.content / sha256

    def save(self, source: Path) -> str:
        """Keep a copy of ``source`` by its sha256 and return the hash."""
        digest = sha256_of(source)
        target = self.path(digest)
        if not target.exists():
            self.content.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(dir=str(self.content), prefix=".blob.")
            os.close(handle)
            tmp = Path(tmp_name)
            shutil.copyfile(source, tmp)
            tmp.chmod(0o644)
            tmp.replace(target)
        if sha256_of(target) != digest:
            raise DevLaneBranchInstallRefusedError(
                f"blob {digest} did not read back from {target}"
            )
        return digest

    def drop_unreferenced(self, records: ModelDevLaneBranchInstallRecords) -> None:
        if not self.content.is_dir():
            return
        keep = {f.sha256 for install in records.installs for f in install.files}
        for path in self.content.iterdir():
            if path.name not in keep and SHA256_PATTERN.match(path.name):
                path.unlink(missing_ok=True)


class DevLaneBranchInstallStore:
    """The records and their content, under ``<state dir>/branch-installs``."""

    def __init__(self, state_dir: Path) -> None:
        self.root = state_dir / STORE_DIRNAME
        self.records_path = self.root / RECORDS_FILE
        self.status_path = self.root / STATUS_FILE
        self.released_path = self.root / RELEASED_FILE
        self.blobs = DevLaneBranchInstallBlobs(self.root / CONTENT_DIRNAME)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Serialise the agent's read-modify-writes with the registrar's."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / LOCK_FILE).open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self) -> ModelDevLaneBranchInstallRecords:
        """The declared installs; an absent file is none, an unreadable one a refusal."""
        try:
            text = self.records_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ModelDevLaneBranchInstallRecords()
        try:
            records = ModelDevLaneBranchInstallRecords.model_validate_json(text)
        except ValueError as exc:
            raise DevLaneBranchInstallRefusedError(
                f"{self.records_path} is not a valid record file: {exc}"
            ) from exc
        if records.schema_version != SCHEMA:
            raise DevLaneBranchInstallRefusedError(
                f"{self.records_path} has schema {records.schema_version!r}, "
                f"expected {SCHEMA!r}"
            )
        return records

    def _write_atomic(self, path: Path, text: str, mode: int = 0o644) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}."
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(text)
                out.flush()
                os.fsync(out.fileno())
            tmp.chmod(mode)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def save(self, records: ModelDevLaneBranchInstallRecords) -> None:
        """Write the records, keep the previous file, and read the new one back."""
        backup = self.records_path.with_suffix(".json.bak")
        if self.records_path.exists():
            shutil.copy2(self.records_path, backup)
        self._write_atomic(self.records_path, records.model_dump_json(indent=2) + "\n")
        if self.load() != records:
            raise DevLaneBranchInstallRefusedError(
                f"{self.records_path} did not read back as written; the previous "
                f"file is {backup}"
            )

    def append_released(
        self,
        install: ModelDevLaneBranchInstall,
        release: ModelDevLaneBranchInstallRelease,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "release": json.loads(release.model_dump_json()),
                "install": json.loads(install.model_dump_json()),
            },
            sort_keys=True,
        )
        with self.released_path.open("a", encoding="utf-8") as out:
            out.write(line + "\n")
            out.flush()
            os.fsync(out.fileno())

    def write_status(self, receipt: ModelDevLaneBranchInstallReceipt) -> None:
        self._write_atomic(self.status_path, receipt.model_dump_json(indent=2) + "\n")

    def read_status(self) -> dict[str, object] | None:
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None


def register(
    store: DevLaneBranchInstallStore,
    *,
    lane: str,
    install_id: str,
    repo: str,
    pr: int,
    branch: str,
    head: str,
    ticket: str,
    registered_by: str,
    services: Sequence[str],
    sources: Mapping[str, Path],
    expires_at: datetime,
    restart: bool = True,
    replace: bool = False,
    expected_sha256: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> ModelDevLaneBranchInstall:
    """Declare one install, keeping its bytes; refuse rather than guess.

    ``sources`` maps a container path to the local file holding the branch's
    version of it. ``expected_sha256`` lets the registrar assert the hash it was
    shown (a lane's own readback of the file it built) and refuses a mismatch.
    """
    if lane != DEV_LANE:
        raise DevLaneBranchInstallRefusedError(
            f"lane {lane!r} refused: branch installs are for the dev lane only"
        )
    when = (now or datetime.now(UTC)).astimezone(UTC)
    expires = expires_at.astimezone(UTC)
    if expires <= when:
        raise DevLaneBranchInstallRefusedError(
            f"expires_at {_utc(expires)} is not in the future"
        )
    if expires > when + MAX_LIFETIME:
        raise DevLaneBranchInstallRefusedError(
            f"expires_at {_utc(expires)} is more than {MAX_LIFETIME.days} days out"
        )
    if not sources:
        raise DevLaneBranchInstallRefusedError("an install names at least one file")
    files: list[ModelDevLaneBranchInstallFile] = []
    with store.locked():
        for container_path, source in sorted(sources.items()):
            if not source.is_file():
                raise DevLaneBranchInstallRefusedError(f"source {source} is not a file")
            digest = sha256_of(source)
            wanted = (expected_sha256 or {}).get(container_path)
            if wanted is not None and wanted != digest:
                raise DevLaneBranchInstallRefusedError(
                    f"{container_path}: source hashes to {digest[:12]}, "
                    f"expected {wanted[:12]}"
                )
            try:
                files.append(
                    ModelDevLaneBranchInstallFile(path=container_path, sha256=digest)
                )
            except ValueError as exc:
                raise DevLaneBranchInstallRefusedError(str(exc)) from exc
        try:
            install = ModelDevLaneBranchInstall(
                id=install_id,
                repo=repo,
                pr=pr,
                branch=branch,
                head=head,
                ticket=ticket,
                registered_by=registered_by,
                services=tuple(services),
                files=tuple(files),
                restart=restart,
                registered_at=when,
                expires_at=expires,
            )
        except ValueError as exc:
            raise DevLaneBranchInstallRefusedError(str(exc)) from exc
        records = store.load()
        existing = next((i for i in records.installs if i.id == install_id), None)
        if existing is not None and not replace:
            raise DevLaneBranchInstallRefusedError(
                f"install {install_id!r} is already declared; release it or pass replace"
            )
        for source in sources.values():
            store.blobs.save(source)
        kept = tuple(i for i in records.installs if i.id != install_id)
        if existing is not None:
            store.append_released(
                existing,
                ModelDevLaneBranchInstallRelease(
                    install_key=install_id,
                    reason=EnumDevLaneBranchInstallReleaseReason.REPLACED,
                    detail=f"replaced by head {head[:12]}",
                    released_at=when,
                ),
            )
        store.save(ModelDevLaneBranchInstallRecords(installs=(*kept, install)))
    return install


def release(
    store: DevLaneBranchInstallStore,
    install_id: str,
    reason: EnumDevLaneBranchInstallReleaseReason,
    detail: str,
    *,
    now: datetime | None = None,
) -> ModelDevLaneBranchInstallRelease | None:
    """Drop one install with a line in the history; ``None`` when it is not declared."""
    when = (now or datetime.now(UTC)).astimezone(UTC)
    with store.locked():
        records = store.load()
        install = next((i for i in records.installs if i.id == install_id), None)
        if install is None:
            return None
        released = ModelDevLaneBranchInstallRelease(
            install_key=install_id, reason=reason, detail=detail, released_at=when
        )
        store.append_released(install, released)
        remaining = ModelDevLaneBranchInstallRecords(
            installs=tuple(i for i in records.installs if i.id != install_id)
        )
        store.save(remaining)
        store.blobs.drop_unreferenced(remaining)
    return released


class DevLaneContainerReader:
    """Reads the files of one compose project's running containers; writes nothing."""

    def __init__(self, compose_project: str, run: Runner) -> None:
        self.compose_project = compose_project
        self._run = run

    def containers(self, service: str) -> tuple[list[str], str]:
        try:
            result = self._run(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--no-trunc",
                    "--filter",
                    f"label=com.docker.compose.project={self.compose_project}",
                    "--filter",
                    f"label=com.docker.compose.service={service}",
                ],
                timeout=DOCKER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], f"docker ps failed: {type(exc).__name__}"
        if result.returncode != 0:
            return [], f"docker ps exited {result.returncode}"
        ids = result.stdout.split()
        if len(ids) != 1:
            return [], f"{len(ids)} running containers for service {service}"
        return ids, ""

    def read_sha(
        self, container: str, path: str
    ) -> tuple[str, EnumDevLaneBranchInstallFileState, str]:
        """The file's sha256 in the container, or why it could not be read."""
        try:
            result = self._run(
                ["docker", "exec", container, "sha256sum", "--", path],
                timeout=DOCKER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return "", EnumDevLaneBranchInstallFileState.UNREADABLE, type(exc).__name__
        if result.returncode != 0:
            if "No such file" in (result.stderr or ""):
                return "", EnumDevLaneBranchInstallFileState.MISSING, "file absent"
            return (
                "",
                EnumDevLaneBranchInstallFileState.UNREADABLE,
                f"sha256sum exited {result.returncode}",
            )
        tokens = result.stdout.split()
        token = tokens[0] if tokens else ""
        if not SHA256_PATTERN.match(token):
            return (
                "",
                EnumDevLaneBranchInstallFileState.UNREADABLE,
                "sha256sum printed no hash",
            )
        return token, EnumDevLaneBranchInstallFileState.PRESENT, ""

    def readback(
        self, install: ModelDevLaneBranchInstall
    ) -> ModelDevLaneBranchInstallReadback:
        """Read every file of one install in the container of every service."""
        rows: list[ModelDevLaneBranchInstallFileReadback] = []
        for service in install.services:
            ids, why = self.containers(service)
            if not ids:
                rows.extend(
                    ModelDevLaneBranchInstallFileReadback(
                        service=service,
                        container="",
                        path=f.path,
                        expected_sha256=f.sha256,
                        observed_sha256="",
                        state=EnumDevLaneBranchInstallFileState.NO_CONTAINER,
                        detail=why,
                    )
                    for f in install.files
                )
                continue
            container = ids[0]
            for f in install.files:
                observed, state, detail = self.read_sha(container, f.path)
                if (
                    state is EnumDevLaneBranchInstallFileState.PRESENT
                    and observed != f.sha256
                ):
                    state, detail = (
                        EnumDevLaneBranchInstallFileState.DRIFTED,
                        "sha256 differs",
                    )
                rows.append(
                    ModelDevLaneBranchInstallFileReadback(
                        service=service,
                        container=container,
                        path=f.path,
                        expected_sha256=f.sha256,
                        observed_sha256=observed,
                        state=state,
                        detail=detail,
                    )
                )
        whole = all(r.state in _WHOLE for r in rows)
        return ModelDevLaneBranchInstallReadback(
            install_key=install.id,
            repo=install.repo,
            pr=install.pr,
            head=install.head,
            ticket=install.ticket,
            expires_at=install.expires_at,
            state=EnumDevLaneBranchInstallState.PRESENT
            if whole
            else EnumDevLaneBranchInstallState.MISSING,
            files=tuple(rows),
        )


class DevLaneContainerWriter:
    """Puts stored bytes into a running container and restarts it."""

    def __init__(
        self, blobs: DevLaneBranchInstallBlobs, run: Runner, sleep: Sleeper
    ) -> None:
        self.blobs = blobs
        self._run = run
        self._sleep = sleep

    def cmd_ok(
        self, argv: Sequence[str], timeout: float = DOCKER_TIMEOUT_SECONDS
    ) -> subprocess.CompletedProcess[str] | None:
        try:
            result = self._run(list(argv), timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result if result.returncode == 0 else None

    def copy_in(self, container: str, f: ModelDevLaneBranchInstallFile) -> str:
        """Put one file into one container keeping the owner and mode it had; "" is success."""
        blob = self.blobs.path(f.sha256)
        if not blob.is_file() or sha256_of(blob) != f.sha256:
            return f"stored copy {f.sha256[:12]} is missing or corrupt"
        owner_mode = "0:0 644"
        stat = self.cmd_ok(
            ["docker", "exec", "-u", "0", container, "stat", "-c", "%u:%g %a", f.path]
        )
        if stat is not None and stat.stdout.strip():
            owner_mode = stat.stdout.strip()
        if self.cmd_ok(["docker", "cp", str(blob), f"{container}:{f.path}"]) is None:
            return "docker cp failed"
        owner, _, mode = owner_mode.partition(" ")
        if not (
            self.cmd_ok(
                ["docker", "exec", "-u", "0", container, "chown", owner, f.path]
            )
            and self.cmd_ok(
                ["docker", "exec", "-u", "0", container, "chmod", mode, f.path]
            )
        ):
            return "owner or mode could not be restored"
        return ""

    def await_running(self, container: str) -> bool:
        deadline = time.monotonic() + RESTART_WAIT_SECONDS
        while True:
            state = self.cmd_ok(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}",
                    container,
                ]
            )
            if state is not None:
                status, _, health = state.stdout.strip().partition("|")
                if status == "running" and health in ("", "healthy"):
                    return True
            if time.monotonic() >= deadline:
                return False
            self._sleep(RESTART_POLL_SECONDS)


class HandlerDevLaneBranchInstall:
    """Check, apply and release the declared installs of one compose project."""

    def __init__(
        self,
        store: DevLaneBranchInstallStore,
        compose_project: str,
        *,
        run: Runner | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Sleeper | None = None,
        merged_probe: MergedProbe | None = None,
    ) -> None:
        assert_dev_project(compose_project)
        self.store = store
        self.compose_project = compose_project
        runner: Runner = run or _DEFAULT_RUN
        self._clock = clock or (lambda: datetime.now(UTC))
        self._merged_probe = merged_probe
        self._reader = DevLaneContainerReader(compose_project, runner)
        self._writer = DevLaneContainerWriter(store.blobs, runner, sleep or time.sleep)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def check(self) -> ModelDevLaneBranchInstallReceipt:
        """Read only: which declared installs are not in the running containers."""
        records = self.store.load()
        receipt = ModelDevLaneBranchInstallReceipt(
            checked_at=self._clock(),
            compose_project=self.compose_project,
            readbacks=tuple(self._reader.readback(i) for i in records.installs),
        )
        self.store.write_status(receipt)
        report(receipt)
        return receipt

    # -- release -----------------------------------------------------------

    def release_due(self) -> tuple[ModelDevLaneBranchInstallRelease, ...]:
        """Drop expired installs and those whose merge the lane already runs."""
        now = self._clock()
        out: list[ModelDevLaneBranchInstallRelease] = []
        for install in self.store.load().installs:
            if install.expires_at <= now:
                done = release(
                    self.store,
                    install.id,
                    EnumDevLaneBranchInstallReleaseReason.EXPIRED,
                    f"expired {_utc(install.expires_at)} before "
                    f"{install.repo}#{install.pr} merged",
                    now=now,
                )
            else:
                evidence = self._merged_probe(install) if self._merged_probe else None
                if evidence is None:
                    continue
                done = release(
                    self.store,
                    install.id,
                    EnumDevLaneBranchInstallReleaseReason.MERGED,
                    evidence,
                    now=now,
                )
            if done is not None:
                logger.info(
                    "branch install RELEASED: id=%s repo=%s pr=#%s reason=%s detail=%s",
                    install.id,
                    install.repo,
                    install.pr,
                    done.reason.value,
                    done.detail,
                )
                out.append(done)
        return tuple(out)

    def _due_for_release(self) -> bool:
        now = self._clock()
        for install in self.store.load().installs:
            if install.expires_at <= now:
                return True
            if self._merged_probe is not None and self._merged_probe(install):
                return True
        return False

    # -- apply -------------------------------------------------------------

    def apply(self) -> ModelDevLaneBranchInstallReceipt:
        """Put back every declared install that is not whole, restart, read it all back.

        Three passes so a container carrying several installs restarts once:
        copy what differs, restart each changed container whose install asks for
        it, then read every file of every install back. A file that does not read
        back at its recorded sha256 is FAILED and logged.
        """
        released = self.release_due()
        installs = self.store.load().installs
        failed: dict[tuple[str, str, str], str] = {}
        was: dict[tuple[str, str, str], EnumDevLaneBranchInstallFileState] = {}
        to_restart: set[str] = set()
        for install in installs:
            before = self._reader.readback(install)
            if before.state is EnumDevLaneBranchInstallState.PRESENT:
                continue
            report_one(before)
            by_path = {f.path: f for f in install.files}
            for row in before.files:
                key = (install.id, row.service, row.path)
                was[key] = row.state
                if row.state is EnumDevLaneBranchInstallFileState.PRESENT:
                    continue
                if row.state in (
                    EnumDevLaneBranchInstallFileState.NO_CONTAINER,
                    EnumDevLaneBranchInstallFileState.UNREADABLE,
                ):
                    failed[key] = row.detail or row.state.value
                    continue
                problem = self._writer.copy_in(row.container, by_path[row.path])
                if problem:
                    failed[key] = problem
                elif install.restart:
                    to_restart.add(row.container)
        restarted: list[str] = []
        for container in sorted(to_restart):
            if (
                self._writer.cmd_ok(
                    ["docker", "restart", container], RESTART_TIMEOUT_SECONDS
                )
                is None
            ):
                logger.error(
                    "branch install restart FAILED: container=%s "
                    "friction_type=branch_install_restart_failed",
                    container[:12],
                )
            elif not self._writer.await_running(container):
                logger.error(
                    "branch install container not running and healthy after restart: "
                    "container=%s friction_type=branch_install_restart_unhealthy",
                    container[:12],
                )
            else:
                restarted.append(container)
        final = tuple(
            self._final_readback(install, was, failed) for install in installs
        )
        receipt = ModelDevLaneBranchInstallReceipt(
            checked_at=self._clock(),
            compose_project=self.compose_project,
            readbacks=final,
            released=released,
            restarted=tuple(restarted),
        )
        self.store.write_status(receipt)
        report(receipt)
        return receipt

    def _final_readback(
        self,
        install: ModelDevLaneBranchInstall,
        was: Mapping[tuple[str, str, str], EnumDevLaneBranchInstallFileState],
        failed: Mapping[tuple[str, str, str], str],
    ) -> ModelDevLaneBranchInstallReadback:
        after = self._reader.readback(install)
        rows: list[ModelDevLaneBranchInstallFileReadback] = []
        for row in after.files:
            key = (install.id, row.service, row.path)
            if row.state is EnumDevLaneBranchInstallFileState.PRESENT:
                if (
                    was.get(key, EnumDevLaneBranchInstallFileState.PRESENT)
                    is not EnumDevLaneBranchInstallFileState.PRESENT
                ):
                    row = row.model_copy(
                        update={"state": EnumDevLaneBranchInstallFileState.REAPPLIED}
                    )
            else:
                row = row.model_copy(
                    update={
                        "state": EnumDevLaneBranchInstallFileState.FAILED,
                        "detail": failed.get(key) or row.detail or row.state.value,
                    }
                )
            rows.append(row)
        if all(r.state is EnumDevLaneBranchInstallFileState.PRESENT for r in rows):
            state = EnumDevLaneBranchInstallState.PRESENT
        elif all(r.state in _WHOLE for r in rows):
            state = EnumDevLaneBranchInstallState.REAPPLIED
        else:
            state = EnumDevLaneBranchInstallState.FAILED
        return after.model_copy(update={"files": tuple(rows), "state": state})

    def reconcile(self, guard: Guard | None = None) -> ModelDevLaneBranchInstallReceipt:
        """Check; when anything is missing or due for release, apply under ``guard``.

        ``guard`` is the lane lock for a caller outside a deploy job; the job path
        already holds it and passes none.
        """
        receipt = self.check()
        if receipt.ok and not self._due_for_release():
            return receipt
        with guard() if guard is not None else contextlib.nullcontext():
            return self.apply()


def report_one(readback: ModelDevLaneBranchInstallReadback) -> None:
    """The detector line for each file of an install that is not whole."""
    for row in readback.files:
        if row.state in _WHOLE:
            continue
        logger.error(
            "BRANCH-INSTALL-MISSING: id=%s repo=%s pr=#%s head=%s ticket=%s service=%s "
            "container=%s file=%s state=%s expected_sha256=%s observed_sha256=%s "
            "detail=%s friction_type=branch_install_missing",
            readback.install_key,
            readback.repo,
            readback.pr,
            readback.head[:12],
            readback.ticket,
            row.service,
            row.container[:12] or "none",
            row.path,
            row.state.value,
            _short(row.expected_sha256),
            _short(row.observed_sha256),
            row.detail or "-",
        )


def report(receipt: ModelDevLaneBranchInstallReceipt) -> None:
    """One summary line, and the detector line for every file that is not whole."""
    for readback in receipt.readbacks:
        if readback.state is EnumDevLaneBranchInstallState.PRESENT:
            continue
        if readback.state is EnumDevLaneBranchInstallState.REAPPLIED:
            logger.warning(
                "branch install REAPPLIED: id=%s repo=%s pr=#%s head=%s files=%s "
                "readback=sha256-match restarted=%s",
                readback.install_key,
                readback.repo,
                readback.pr,
                readback.head[:12],
                len(readback.files),
                ",".join(c[:12] for c in receipt.restarted) or "none",
            )
        else:
            report_one(readback)
    logger.info(
        "branch installs: declared=%s present=%s reapplied=%s missing=%s released=%s",
        len(receipt.readbacks),
        sum(
            1
            for r in receipt.readbacks
            if r.state is EnumDevLaneBranchInstallState.PRESENT
        ),
        sum(
            1
            for r in receipt.readbacks
            if r.state is EnumDevLaneBranchInstallState.REAPPLIED
        ),
        len(receipt.missing),
        len(receipt.released),
    )


def status_payload(
    store: DevLaneBranchInstallStore, now: datetime
) -> dict[str, object]:
    """The declared installs and the last readback, for a reader of the lane's state."""
    try:
        records = store.load()
    except DevLaneBranchInstallRefusedError as exc:
        return {"installs": None, "reason": str(exc)}
    status = store.read_status()
    readbacks = status.get("readbacks") if status else None
    missing = [
        str(r.get("install_key"))
        for r in (readbacks if isinstance(readbacks, list) else [])
        if isinstance(r, dict) and r.get("state") in ("missing", "failed")
    ]
    return {
        "schema": SCHEMA,
        "read_at": _utc(now),
        "declared": [json.loads(i.model_dump_json()) for i in records.installs],
        "last_check": status,
        "missing": missing,
    }


def git_merged_probe(
    clone_for: Callable[[str], Path | None],
    running_ref_for: Callable[[str], str | None],
    *,
    run: Runner | None = None,
    dev_ref: str = DEV_BRANCH_REF,
) -> MergedProbe:
    """Merged means the PR is on the dev branch and the lane's image already runs it.

    A squash merge leaves a commit whose subject ends ``(#<pr>)`` and never makes
    the branch head an ancestor, so both shapes are read. Anything unread (no
    clone, no running ref, a git error) keeps the install.
    """
    runner: Runner = run or _DEFAULT_RUN

    def probe(install: ModelDevLaneBranchInstall) -> str | None:
        clone = clone_for(install.repo)
        running = running_ref_for(install.repo)
        if clone is None or not running:
            return None
        merge = _merge_commit(runner, clone, dev_ref, install)
        if merge is None:
            return None
        try:
            contained = runner(
                [
                    "git",
                    "-C",
                    str(clone),
                    "merge-base",
                    "--is-ancestor",
                    merge,
                    running,
                ],
                timeout=DOCKER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if contained.returncode != 0:
            return None
        return (
            f"{install.repo}#{install.pr} merged as {merge[:12]} on {dev_ref}; the "
            f"running image carries it (build ref {running[:12]})"
        )

    return probe


def _merge_commit(
    runner: Runner, clone: Path, dev_ref: str, install: ModelDevLaneBranchInstall
) -> str | None:
    suffix = f"(#{install.pr})"
    try:
        log = runner(
            [
                "git",
                "-C",
                str(clone),
                "log",
                f"-n{MERGE_SEARCH_DEPTH}",
                "--format=%H%x1f%s",
                dev_ref,
            ],
            timeout=DOCKER_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if log.returncode == 0:
        for line in log.stdout.splitlines():
            sha, _, subject = line.partition("\x1f")
            if COMMIT_PATTERN.match(sha) and subject.rstrip().endswith(suffix):
                return sha
    try:
        ancestor = runner(
            [
                "git",
                "-C",
                str(clone),
                "merge-base",
                "--is-ancestor",
                install.head,
                dev_ref,
            ],
            timeout=DOCKER_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return install.head if ancestor.returncode == 0 else None


# -- command line ------------------------------------------------------------


def _pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in values:
        key, sep, value = pair.partition("=")
        if not sep or not key or not value:
            raise DevLaneBranchInstallRefusedError(
                f"{flag} {pair!r} must be <key>=<value>"
            )
        out[key] = value
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omnibase_infra.handlers.handler_dev_lane_branch_install",
        description="Declare, check, apply and release dev-lane branch installs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "release", "list", "check", "apply", "release-due"):
        p = sub.add_parser(name)
        p.add_argument("--state-dir", required=True, type=Path)
        if name in ("check", "apply", "release-due"):
            p.add_argument("--compose-project", required=True)
        if name in ("apply", "release-due"):
            p.add_argument(
                "--clone", action="append", default=[], help="REPO=PATH of a clone"
            )
            p.add_argument(
                "--running-ref",
                action="append",
                default=[],
                help="REPO=SHA the running image was built from",
            )
    reg = sub.choices["register"]
    reg.add_argument("--lane", required=True)
    reg.add_argument("--id", required=True, dest="install_id")
    reg.add_argument("--repo", required=True)
    reg.add_argument("--pr", required=True, type=int)
    reg.add_argument("--branch", required=True)
    reg.add_argument("--head", required=True)
    reg.add_argument("--ticket", required=True)
    reg.add_argument("--registered-by", required=True)
    reg.add_argument("--service", required=True, action="append", dest="services")
    reg.add_argument("--file", required=True, action="append", dest="files")
    reg.add_argument("--expect-sha256", action="append", default=[], dest="expect")
    reg.add_argument("--expires-in-hours", required=True, type=float)
    reg.add_argument("--no-restart", action="store_true")
    reg.add_argument("--replace", action="store_true")
    reg.add_argument("--compose-project", help="apply at once to this project")
    rel = sub.choices["release"]
    rel.add_argument("--id", required=True, dest="install_id")
    rel.add_argument("--reason", required=True)
    return parser


def _probe(args: argparse.Namespace) -> MergedProbe | None:
    clones = _pairs(getattr(args, "clone", []), "--clone")
    refs = _pairs(getattr(args, "running_ref", []), "--running-ref")
    if not clones or not refs:
        return None
    return git_merged_probe(
        lambda repo: Path(clones[repo]) if repo in clones else None, refs.get
    )


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Runner | None = None,
    clock: Callable[[], datetime] | None = None,
    sleep: Sleeper | None = None,
) -> int:
    """Run one subcommand. Logs go to stderr; the result is one stdout line."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s branch-installs: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    store = DevLaneBranchInstallStore(args.state_dir)
    try:
        if args.command == "register":
            install = register(
                store,
                lane=args.lane,
                install_id=args.install_id,
                repo=args.repo,
                pr=args.pr,
                branch=args.branch,
                head=args.head,
                ticket=args.ticket,
                registered_by=args.registered_by,
                services=args.services,
                sources={k: Path(v) for k, v in _pairs(args.files, "--file").items()},
                expires_at=(clock or (lambda: datetime.now(UTC)))()
                + timedelta(hours=args.expires_in_hours),
                restart=not args.no_restart,
                replace=args.replace,
                expected_sha256=_pairs(args.expect, "--expect-sha256"),
                now=clock() if clock else None,
            )
            _emit(f"REGISTERED {install.model_dump_json()}")
            if args.compose_project:
                receipt = HandlerDevLaneBranchInstall(
                    store, args.compose_project, run=run, clock=clock, sleep=sleep
                ).apply()
                _emit(f"READBACK {receipt.model_dump_json()}")
                return EXIT_OK if receipt.ok else EXIT_NOT_WHOLE
            return EXIT_OK
        if args.command == "release":
            released = release(
                store,
                args.install_id,
                EnumDevLaneBranchInstallReleaseReason.OPERATOR,
                args.reason,
            )
            _emit(
                f"RELEASED {released.model_dump_json()}"
                if released
                else f"NOT-DECLARED {args.install_id}"
            )
            return EXIT_OK if released else EXIT_NOT_WHOLE
        if args.command == "list":
            _emit(json.dumps(status_payload(store, datetime.now(UTC)), indent=2))
            return EXIT_OK
        handler = HandlerDevLaneBranchInstall(
            store,
            args.compose_project,
            run=run,
            clock=clock,
            sleep=sleep,
            merged_probe=_probe(args),
        )
        if args.command == "release-due":
            for dropped in handler.release_due():
                _emit(f"RELEASED {dropped.model_dump_json()}")
            return EXIT_OK
        receipt = handler.check() if args.command == "check" else handler.apply()
        _emit(f"RECEIPT {receipt.model_dump_json()}")
        return EXIT_OK if receipt.ok else EXIT_NOT_WHOLE
    except DevLaneBranchInstallRefusedError as exc:
        _emit(f"REFUSED {exc}")
        return EXIT_REFUSED


__all__ = [
    "DevLaneBranchInstallBlobs",
    "DevLaneBranchInstallRefusedError",
    "DevLaneBranchInstallStore",
    "DevLaneContainerReader",
    "DevLaneContainerWriter",
    "HandlerDevLaneBranchInstall",
    "git_merged_probe",
    "main",
    "register",
    "release",
    "status_payload",
]


if __name__ == "__main__":
    sys.exit(main())
