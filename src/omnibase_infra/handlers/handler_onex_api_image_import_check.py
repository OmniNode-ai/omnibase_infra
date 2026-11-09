# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Import-check a candidate onex-api image and put the previous pin back (OMN-23039).

``deliver_onex_api_pin`` advanced ``ONEX_API_IMAGE`` and recreated ``onex-api``
on the strength of the image's provenance alone. On 2026-11-09T03:58Z it
delivered the onex-api image tagged ``145fd4d6-20261109T032509Z``, whose omnimarket
no longer carried ``omnimarket.routing.byok_provider_backends`` (omnimarket#4810
moved it). ``onex-api`` crash-looped 246 times on ``ModuleNotFoundError``, every
other compose service waits on it being healthy, and every later job failed.

This handler runs after the repoint has written the new pin and before anything
is recreated. One throwaway container on the candidate image, no network, no
entrypoint: the same interpreter and the same installed omnimarket the service
would start with. A failure writes the previous pin back, changing that one line.

FAILS CLOSED, with the reason said. A check that could not run (docker refused,
the interpreter is absent, the timeout passed, the output is not the script's
verdict) is a failure worded "could not run", never "found a missing module", so
the journal tells the two apart. Keeping the previous pin on an unreadable check
costs a stale onex-api; advancing it on one is the incident this exists to stop.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from omnibase_infra.models.model_onex_api_image_import_request import (
    ModelOnexApiImageImportRequest,
)
from omnibase_infra.models.model_onex_api_image_import_verdict import (
    ModelOnexApiImageImportVerdict,
)

logger = logging.getLogger(__name__)

#: Run INSIDE the candidate image by ``python -c`` from its ``WORKDIR`` (/app).
#: It walks the application's own ``.py`` files (never tests), reads the
#: MODULE-LEVEL ``omnimarket`` imports out of each with ``ast``, and asks the
#: image's interpreter whether each module can be found. Module-level only,
#: because those are the imports that crash the container before it can serve: a
#: guarded or function-level import is allowed to be absent. A file that does not
#: parse is skipped, not a verdict. ``find_spec`` rather than a real import, so
#: the check needs no env and opens no connection. Exit 0 with ``missing == []``
#: is the only pass; exit 3 names every module that is gone.
IMPORT_CHECK_SCRIPT = """
import ast, importlib.util, json, pathlib, sys
checked, missing = [], []
for path in sorted(pathlib.Path(".").rglob("*.py")):
    if any(p in ("tests", "__pycache__") or p.startswith(".") for p in path.parts[:-1]):
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        continue
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            if name.split(".")[0] != "omnimarket" or name in checked:
                continue
            checked.append(name)
            try:
                found = importlib.util.find_spec(name) is not None
            except ImportError:
                found = False
            if not found:
                missing.append({"module": name, "imported_by": str(path)})
print(json.dumps({"checked": checked, "missing": missing}))
sys.exit(3 if missing else 0)
"""

_PIN_PREFIX = "ONEX_API_IMAGE="

#: ``(argv, timeout=...) -> CompletedProcess``: the executor's own ``_run``.
RunCommand = Callable[..., "subprocess.CompletedProcess[str]"]


class HandlerOnexApiImageImportCheck:
    """Decide whether a candidate onex-api image may keep the advanced pin."""

    def __init__(self, *, run: RunCommand) -> None:
        self._run = run

    def handle(
        self, request: ModelOnexApiImageImportRequest
    ) -> ModelOnexApiImageImportVerdict:
        importable, detail = self._check(request)
        if importable:
            return ModelOnexApiImageImportVerdict(
                importable=True,
                rolled_back=False,
                reason=detail,
                candidate=request.image,
                pin_after=request.image,
            )
        restored, outcome = self._restore(request)
        return ModelOnexApiImageImportVerdict(
            importable=False,
            rolled_back=restored,
            reason=f"IMPORT_CHECK_FAILED_ROLLED_BACK: {detail}; {outcome}",
            candidate=request.image,
            pin_after=request.previous_pin if restored else request.image,
        )

    def _check(self, request: ModelOnexApiImageImportRequest) -> tuple[bool, str]:
        argv = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--entrypoint",
            "python",
            request.image,
            "-c",
            IMPORT_CHECK_SCRIPT,
        ]
        try:
            result = self._run(argv, timeout=request.timeout_seconds)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, (
                f"the import check could not run on {request.image}: "
                f"{type(exc).__name__}: {exc}"
            )
        try:
            verdict = json.loads(result.stdout.strip().splitlines()[-1])
            missing = verdict["missing"]
            checked = verdict["checked"]
        except (IndexError, ValueError, KeyError, TypeError):
            return False, (
                f"the import check could not run on {request.image}: docker exited "
                f"{result.returncode} with no verdict: "
                f"{(result.stderr or result.stdout).strip()[:400]}"
            )
        if result.returncode == 0 and not missing:
            return True, f"{len(checked)} omnimarket module(s) import-resolvable"
        names = ", ".join(
            f"{m.get('module')} (imported by {m.get('imported_by')})" for m in missing
        )
        return False, f"{request.image} cannot import: {names}"

    def _restore(self, request: ModelOnexApiImageImportRequest) -> tuple[bool, str]:
        """Write ``previous_pin`` back, changing that one line and no other.

        Refuses rather than guesses when the pin line is not unique: the repoint
        refused a duplicate or missing line before it wrote, so a file that now
        has one is not the file it wrote, and a rollback that guesses is worse
        than the advance it undoes.
        """
        if not request.previous_pin:
            return False, "the repoint recorded no previous pin, so none was restored"
        try:
            lines = request.env_file.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
            hits = [i for i, line in enumerate(lines) if line.startswith(_PIN_PREFIX)]
            if len(hits) != 1:
                return False, (
                    f"the pin was not restored: {request.env_file} carries "
                    f"{len(hits)} {_PIN_PREFIX} lines"
                )
            end = "\n" if lines[hits[0]].endswith("\n") else ""
            lines[hits[0]] = f"{_PIN_PREFIX}{request.previous_pin}{end}"
            request.env_file.write_text("".join(lines), encoding="utf-8")
        except OSError as exc:
            return False, f"the pin was not restored: {type(exc).__name__}: {exc}"
        return True, (
            f"the pin was put back to {request.previous_pin} and onex-api was "
            "not recreated"
        )


def build_import_check_handler(run: RunCommand) -> HandlerOnexApiImageImportCheck:
    """Wire the handler to the caller's command runner (the agent has no container)."""
    return HandlerOnexApiImageImportCheck(run=run)


def import_gate(
    run: RunCommand, record: dict[str, object], env_file: Path
) -> dict[str, object] | None:
    """The deploy agent's one call: ``None`` to proceed, or the refused record fields.

    ``record`` is the repoint's own record of an advance, so ``pin_after`` is the
    candidate and ``pin_before`` the pin to put back.
    """
    verdict = build_import_check_handler(run).handle(
        ModelOnexApiImageImportRequest(
            image=str(record.get("pin_after") or ""),
            env_file=env_file,
            previous_pin=str(record.get("pin_before") or ""),
        )
    )
    if verdict.importable:
        logger.info("onex-api candidate import check passed: %s", verdict.reason)
        return None
    logger.error("onex-api candidate image rejected: %s", verdict.reason)
    return verdict.record_fields()
