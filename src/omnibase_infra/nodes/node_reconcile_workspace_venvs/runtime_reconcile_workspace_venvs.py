# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bootstrap runtime for the host reconciler's contract-bound hook sync.

This operation runs before the dispatcher is available, after its gate venv
has been reconciled. Host ownership remains the shell adapter's responsibility.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_reconcile_workspace_venvs import (
    HandlerReconcileWorkspaceVenvs,
    _normalize,
)
from omnibase_infra.models.model_hook_venv_reconcile_policy import (
    ModelHookVenvReconcilePolicy,
)
from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)
from omnibase_infra.protocols.protocol_hook_venv_reconcile_host import (
    HOOK_VENV_HOST_SERVICE,
)
from omnibase_infra.protocols.protocol_hook_venv_reconcile_host import (
    HookVenvRevision as _Revision,
)


class AdapterHookVenvReconcileHost:
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
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        args = [
            str(request.uv_bin),
            "sync",
            "--frozen",
            "--project",
            str(request.project),
        ]
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


async def _bootstrap(contract_path: Path) -> ModelONEXContainer:
    """Bind contract configuration and the effect at the composition root."""
    contract = yaml.safe_load(contract_path.read_text())
    policy = ModelHookVenvReconcilePolicy.model_validate(contract["reconcile_policy"])
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        ModelHookVenvReconcilePolicy, policy
    )
    await container.service_registry.register_instance(
        HOOK_VENV_HOST_SERVICE, AdapterHookVenvReconcileHost(container)
    )
    await container.service_registry.register_instance(
        HandlerReconcileWorkspaceVenvs, HandlerReconcileWorkspaceVenvs(container)
    )
    return container


def main() -> int:
    """Adapt the pre-dispatch shell reconciler to the contract-bound handler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    container = asyncio.run(_bootstrap(args.contract))
    result = container.get_service(HandlerReconcileWorkspaceVenvs).handle(
        ModelHookVenvReconcileRequest(
            workspace_root=args.workspace,
            project=args.project,
            uv_bin=args.uv,
        )
    )
    if result.output:
        sys.stderr.write(result.output)
    stream = sys.stderr if result.status == "refused" else sys.stdout
    stream.write("[reconcile] " + result.detail + "\n")
    return 2 if result.status == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
