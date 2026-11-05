# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reconcile a hook venv without discarding installer-selected Git revisions.

The existing host reconciler calls this effect through the node bootstrap runtime:
dispatch itself depends on a reconciled environment. The same typed handler is
bound by node_reconcile_workspace_venvs for local bootstrap (OMN-19998).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_hook_venv_reconcile_policy import (
    ModelHookVenvReconcilePolicy,
)
from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)
from omnibase_infra.models.model_hook_venv_reconcile_result import (
    ModelHookVenvReconcileResult,
)
from omnibase_infra.protocols.protocol_hook_venv_reconcile_host import (
    HOOK_VENV_HOST_SERVICE,
    ProtocolHookVenvReconcileHost,
)
from omnibase_infra.protocols.protocol_hook_venv_reconcile_host import (
    HookVenvRevision as _Revision,
)

_CHANGE = re.compile(r"^\s*([-+~])\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?:==|\s+@)")
_GIT_SOURCE = re.compile(r"git\+([^\s)]+)@([0-9a-fA-F]{40})(?:\)|$)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _approved(
    root: Path,
    expected: dict[str, _Revision],
    plan: str,
    host: ProtocolHookVenvReconcileHost,
) -> dict[str, _Revision]:
    changed: set[str] = set()
    replacements: dict[str, _Revision] = {}
    for line in plan.splitlines():
        match = _CHANGE.match(line)
        if match is None:
            continue
        sign, name = match.group(1), _normalize(match.group(2))
        changed.add(name)
        source = _GIT_SOURCE.search(line)
        if sign == "+" and source:
            replacements[name] = (source.group(1), source.group(2))
    if not changed and not re.search(
        r"Would make no changes|Audited [0-9]+ package", plan
    ):
        raise ValueError("unreadable branch install sync plan")
    approved = dict(expected)
    for name in sorted(changed.intersection(expected)):
        after = replacements.get(name)
        if after is None or not host.carries(root, expected[name], after):
            raise ValueError("conflicting branch install would be replaced: " + name)
        approved[name] = after
    return approved


class HandlerReconcileWorkspaceVenvs:
    """Own the existing hook-venv sync operation and its revision readback."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Resolve the node's host effect port from the container."""
        self._host: ProtocolHookVenvReconcileHost = container.get_service(
            HOOK_VENV_HOST_SERVICE
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        """Classify the host package-tool adapter."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Filesystem and package operations are effects."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelHookVenvReconcileRequest
    ) -> ModelHookVenvReconcileResult:
        """Plan before writing and verify the selected revisions after writing."""
        project = request.project
        try:
            expected = self._host.installed(project)
            if expected:
                plan_code, plan_output = self._host.uv(request, "--dry-run")
                if plan_code:
                    return ModelHookVenvReconcileResult(
                        status="refused",
                        detail="branch install sync plan failed",
                        output=plan_output,
                    )
                expected = _approved(
                    request.workspace_root,
                    expected,
                    plan_output,
                    self._host,
                )
            already = self._host.uv(request, "--check")[0] == 0
            output = ""
            if not already:
                sync_code, output = self._host.uv(request)
                if sync_code:
                    return ModelHookVenvReconcileResult(
                        status="refused",
                        detail=f"hook venv lock sync did not complete for {project}",
                        output=output,
                    )
            actual = self._host.installed(project)
            for name, revision in expected.items():
                if actual.get(name) != revision:
                    return ModelHookVenvReconcileResult(
                        status="refused",
                        detail="branch install revision changed after sync: " + name,
                        output=output,
                    )
            return ModelHookVenvReconcileResult(
                status="already_in_sync" if already else "reconciled",
                detail=f"hook venv: {'already in sync' if already else 'reconciled'} ({project}/.venv)",
                output=output,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            StopIteration,
        ) as exc:
            return ModelHookVenvReconcileResult(
                status="refused",
                detail="hook venv branch intent: " + str(exc),
            )


class HandlerHookVenvReconcileHost:
    """Keep raw filesystem and process I/O inside the declared EFFECT node."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._policy = container.get_service(ModelHookVenvReconcilePolicy)

    def installed(self, project: Path) -> dict[str, _Revision]:
        """Read the installer's PEP 610 records without importing the target venv."""
        result: dict[str, _Revision] = {}
        for path in (project / ".venv").glob(
            "lib/python*/site-packages/*.dist-info/direct_url.json"
        ):
            data = json.loads(path.read_text())
            vcs = data.get("vcs_info", {})
            if vcs.get("vcs") != "git":
                continue
            metadata = (path.parent / "METADATA").read_text()
            name = _normalize(
                next(
                    line[6:]
                    for line in metadata.splitlines()
                    if line.startswith("Name: ")
                )
            )
            commit, url = vcs.get("commit_id", ""), data.get("url", "")
            if (
                not isinstance(commit, str)
                or not re.fullmatch(r"[0-9a-fA-F]{40}", commit)
                or not isinstance(url, str)
                or not url
                or name in result
            ):
                raise ValueError("unreadable or ambiguous Git revision for " + name)
            result[name] = (url, commit)
        return result

    def carries(self, root: Path, before: _Revision, after: _Revision) -> bool:
        """Allow ordinary forward updates; refuse unknown or discarded commits."""
        if before == after:
            return True
        source = urlsplit(before[0])
        clone = (
            Path(unquote(source.path))
            if source.scheme == "file"
            else root / Path(source.path).name.removesuffix(".git")
        )
        if before[0] != after[0] or not clone.is_dir():
            return False
        try:
            probe = subprocess.run(
                [
                    "git",
                    "-C",
                    str(clone),
                    "merge-base",
                    "--is-ancestor",
                    before[1],
                    after[1],
                ],
                env=scrub_git_location_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=self._policy.git_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("branch install ancestry read timed out") from exc
        if probe.returncode not in (0, 1):
            raise ValueError("branch install ancestry is unreadable")
        return probe.returncode == 0

    def uv(
        self, request: ModelHookVenvReconcileRequest, flag: str | None = None
    ) -> tuple[int, str]:
        """Run uv against exactly the requested project as the host owner."""
        env = scrub_git_location_env()
        env.pop("PYTHONPATH", None)
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        args = [
            str(request.uv_bin),
            "sync",
        ]
        # uv before 0.7 refuses --frozen together with --dry-run (CI pins 0.6.14).
        if flag != "--dry-run":
            args.append("--frozen")
        args += ["--project", str(request.project)]
        if flag:
            args.append(flag)
        try:
            result = subprocess.run(
                args,
                cwd=request.project,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._policy.sync_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("hook venv package operation timed out") from exc
        return result.returncode, result.stdout + result.stderr
