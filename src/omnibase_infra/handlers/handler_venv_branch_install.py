# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Install one omnimarket pull request head into a host's dispatch venv, and take it out (OMN-26102).

THE GAP
-------
Code a venv-resident unit runs reaches the host only when its pull request merges
and the reconcile tick carries it, so a proof that "it runs from the branch" has
to wait for the merge. No declared installer put one pull request head into the
dispatch venv and took it out again.

THE SHAPE
---------
An in-place exchange of only the omnimarket package of the head through the
existing atomic venv install (``venv_atomic_install``): stage into a side folder
under the staging lock, exchange the package directories in with a journal, read
the installed revision back. The venv is shared by every unit and lane on the
host, so the install moves all of them until it is reverted; the units it names
are the record of who that is, and one install per host may be active at a time.

The install writes a receipt that holds the revision it replaced. The revert
restores that revision from the stage folder's journal and reads it back; a
revision that does not read back is a failure, never quiet.

WHAT IT NEVER DOES
------------------
It touches no shared clone of any repository, runs no git command, and does not
start, stop or restart any unit or runtime: the only processes it runs are the
venv's own interpreter (a read) and ``uv`` (through the atomic engine).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.venv_atomic_install import (
    atomic_venv_install as engine,
)
from omnibase_infra.handlers.venv_atomic_install import (
    dispatch_venv_promotion as promotion,
)
from omnibase_infra.models.model_venv_branch_install_receipt import (
    ModelVenvBranchInstallReceipt,
)
from omnibase_infra.models.model_venv_branch_install_request import (
    ModelVenvBranchInstallRequest,
)
from omnibase_infra.models.model_venv_branch_install_result import (
    ModelVenvBranchInstallResult,
)

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE = "omnimarket"
_Status = Literal["installed", "reverted", "noop", "refused", "failed"]
_EXIT_REFUSED = 3
_EXIT_FAILED = 5
_RECORD_PROBE = r"""
import json
from importlib.metadata import PackageNotFoundError, distribution
try:
    text = distribution("omnimarket").read_text("direct_url.json")
except PackageNotFoundError:
    text = None
record = json.loads(text) if text else {}
print(json.dumps([record.get("url", ""), record.get("vcs_info", {}).get("commit_id", "")]))
"""


def _read_record(python_bin: str) -> tuple[str, str]:
    out = subprocess.run(
        [python_bin, "-I", "-c", _RECORD_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return "", ""
    url, revision = json.loads(out.stdout)
    return url, revision if _FULL_SHA.match(revision) else ""


def read_installed_revision(python_bin: str) -> str:
    """The commit the venv's omnimarket was installed from; empty when none is recorded."""
    return _read_record(python_bin)[1]


def read_installed_source(python_bin: str) -> str:
    """The repository the venv's omnimarket was installed from; empty when none is recorded."""
    return _read_record(python_bin)[0].removeprefix("git+")


def head_from_watcher_facts(
    state: Mapping[str, object], *, repo: str, number: int
) -> str:
    """The head sha the pull request watcher read, for an open pull request of ``repo``.

    ``state`` is the watcher record of the pull request (``facts`` holds the
    pull request's own fields). Anything else is refused, so the head is never a
    value somebody typed.
    """
    facts = state.get("facts")
    if not isinstance(facts, Mapping):
        raise ValueError("the watcher record carries no facts")
    if facts.get("repo") != repo or facts.get("number") != number:
        raise ValueError(f"the watcher record is not {repo}#{number}")
    if facts.get("state") != "OPEN":
        raise ValueError(f"{repo}#{number} is {facts.get('state')}, not OPEN")
    head = facts.get("head_sha", "")
    if not isinstance(head, str) or not _FULL_SHA.match(head):
        raise ValueError(f"{repo}#{number} has no full head sha in the watcher record")
    return head


def _omnimarket_requirement(source: str, sha: str) -> str:
    """The head of the repository the venv's omnimarket came from."""
    return f"{_PACKAGE} @ git+{source}@{sha}"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _receipts_dir(request: ModelVenvBranchInstallRequest) -> Path:
    return request.state_dir / "receipts"


def _write_receipt(path: Path, receipt: ModelVenvBranchInstallReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = receipt.model_dump(mode="json") | {"installed": receipt.installed}
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scratch.replace(path)


def _discard_stage(stage_dir: Path) -> None:
    engine.prune_git_checkouts(stage_dir)
    shutil.rmtree(stage_dir, ignore_errors=True)


class HandlerVenvBranchInstall:
    """Install a pull request head atomically; revert it to the replaced revision."""

    def __init__(
        self,
        container: ModelONEXContainer,
        *,
        requirement_for: Callable[[str, str], str] = _omnimarket_requirement,
        revision_reader: Callable[[str], str] = read_installed_revision,
        source_reader: Callable[[str], str] = read_installed_source,
    ) -> None:
        """Hold the container; the requirement and the two readers are seams for tests."""
        self._container = container
        self._requirement_for = requirement_for
        self._read_revision = revision_reader
        self._read_source = source_reader

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify the package-tool adapter."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """File system and package operations are effects."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelVenvBranchInstallRequest
    ) -> ModelVenvBranchInstallResult:
        """Run the install or the revert and return its verdict with the output."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result = (
                self._install(request)
                if request.command == "install"
                else self._revert(request)
            )
        return result.model_copy(
            update={"output": result.output + out.getvalue() + err.getvalue()}
        )

    @staticmethod
    def _result(
        status: _Status, code: int, output: str = "", **fields: str
    ) -> ModelVenvBranchInstallResult:
        return ModelVenvBranchInstallResult(
            status=status,
            exit_code=code,
            output=output + "\n" if output else "",
            **fields,
        )

    def _active_receipt(
        self, request: ModelVenvBranchInstallRequest
    ) -> tuple[Path, ModelVenvBranchInstallReceipt] | None:
        for path in sorted(_receipts_dir(request).glob(f"{request.host}-*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("installed", None)
            receipt = ModelVenvBranchInstallReceipt.model_validate(data)
            if receipt.host == request.host and receipt.state == "active":
                return path, receipt
        return None

    def _install(
        self, request: ModelVenvBranchInstallRequest
    ) -> ModelVenvBranchInstallResult:
        python = request.venv / "bin" / "python"
        if not python.is_file():
            return self._result("refused", _EXIT_REFUSED, f"no venv at {request.venv}")
        active = self._active_receipt(request)
        if active is not None:
            return self._result(
                "refused",
                _EXIT_REFUSED,
                f"{request.host} already has an active branch install "
                f"({active[1].installed}); revert it first",
            )
        replaced = self._read_revision(str(python))
        if not _FULL_SHA.match(replaced):
            return self._result(
                "refused",
                _EXIT_REFUSED,
                f"no recorded rollback target: {request.venv} holds no omnimarket "
                "revision to put back",
            )
        if replaced == request.head_sha:
            return self._result(
                "noop",
                0,
                f"{request.host} already runs {request.head_sha}",
                installed=f"installed={request.host}:{request.head_sha}",
                replaced_revision=replaced,
            )
        stage_dir = (
            request.state_dir / "stages" / f"{request.host}-{request.head_sha[:12]}"
        ).resolve()
        if stage_dir.exists():
            return self._result(
                "refused", _EXIT_REFUSED, f"a stage folder is already at {stage_dir}"
            )
        lock = Path(f"{request.venv}.rebuilding.lock")
        if promotion.staging_lock_acquire(lock, os.getpid()) != 0:
            return self._result(
                "refused", _EXIT_REFUSED, "a peer holds the staging lock"
            )
        try:
            return self._stage_and_commit(request, python, replaced, stage_dir)
        finally:
            promotion.staging_lock_release(lock, os.getpid())

    def _stage_and_commit(
        self,
        request: ModelVenvBranchInstallRequest,
        python: Path,
        replaced: str,
        stage_dir: Path,
    ) -> ModelVenvBranchInstallResult:
        stage_dir.parent.mkdir(parents=True, exist_ok=True)
        label = f"omnimarket#{request.pr_number}"
        source = self._read_source(str(python))
        if not source:
            return self._result(
                "refused",
                _EXIT_REFUSED,
                f"{request.venv} records no source repository for omnimarket",
            )
        code = engine.stage(
            str(python),
            stage_dir,
            [self._requirement_for(source, request.head_sha)],
            no_deps=True,
            label=label,
        )
        if code != 0 or not (stage_dir / "plan.json").exists():
            _discard_stage(stage_dir)
            return self._result(
                "refused" if code == _EXIT_REFUSED else "failed",
                code or _EXIT_FAILED,
                f"staging {label} moved nothing; the venv is untouched",
            )
        code = engine.commit_stage(str(python), stage_dir)
        if code != 0:
            _discard_stage(stage_dir)
            return self._result(
                "failed", code, "the commit failed and restored every path"
            )
        installed = self._read_revision(str(python))
        if installed != request.head_sha:
            engine.rollback_stage(stage_dir)
            _discard_stage(stage_dir)
            return self._result(
                "failed",
                _EXIT_FAILED,
                f"the venv reads back {installed or 'no revision'}, not "
                f"{request.head_sha}; rolled back",
            )
        receipt = ModelVenvBranchInstallReceipt(
            host=request.host,
            pr_number=request.pr_number,
            head_sha=request.head_sha,
            replaced_revision=replaced,
            units=request.units,
            venv=str(request.venv),
            stage_dir=str(stage_dir),
            installed_at=_utc_now(),
        )
        path = _receipts_dir(request) / f"{receipt.host}-{receipt.head_sha}.json"
        _write_receipt(path, receipt)
        return self._result(
            "installed",
            0,
            f"installed={receipt.installed}; replaced {replaced}",
            installed=f"installed={receipt.installed}",
            receipt_path=str(path),
            replaced_revision=replaced,
        )

    def _revert(
        self, request: ModelVenvBranchInstallRequest
    ) -> ModelVenvBranchInstallResult:
        active = self._active_receipt(request)
        if active is None:
            return self._result(
                "refused", _EXIT_REFUSED, f"{request.host} has no active branch install"
            )
        path, receipt = active
        python = request.venv / "bin" / "python"
        lock = Path(f"{request.venv}.rebuilding.lock")
        if promotion.staging_lock_acquire(lock, os.getpid()) != 0:
            return self._result(
                "refused", _EXIT_REFUSED, "a peer holds the staging lock"
            )
        try:
            stage_dir = Path(receipt.stage_dir)
            if engine.rollback_stage(stage_dir) != 0:
                return self._result(
                    "failed", _EXIT_FAILED, "the rollback from the journal failed"
                )
            now = self._read_revision(str(python))
            if now != receipt.replaced_revision:
                return self._result(
                    "failed",
                    _EXIT_FAILED,
                    f"the venv reads back {now or 'no revision'}, not the recorded "
                    f"{receipt.replaced_revision}",
                    receipt_path=str(path),
                )
            _discard_stage(stage_dir)
            _write_receipt(
                path,
                receipt.model_copy(
                    update={"state": "reverted", "reverted_at": _utc_now()}
                ),
            )
        finally:
            promotion.staging_lock_release(lock, os.getpid())
        return self._result(
            "reverted",
            0,
            f"reverted to {receipt.replaced_revision}",
            receipt_path=str(path),
            replaced_revision=receipt.replaced_revision,
        )
