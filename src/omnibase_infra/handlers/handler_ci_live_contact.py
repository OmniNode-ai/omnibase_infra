# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Admit CI-tooling changes with recorded contact or a stated exception.

OMN-18648: read object content at the inspected head (or the index for a
commit), rather than borrowing a marker from an unrelated working-tree test.
The shared runtime hosts this effect handler on the contract's bus topics.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_ci_live_contact_request import (
    ModelCILiveContactRequest,
)
from omnibase_infra.models.model_ci_live_contact_result import (
    ModelCILiveContactResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

_EXCEPTION = re.compile(r"^Live-contact exception:[ \t]*(\S[^\n]*)$", re.MULTILINE)
# A pin-only workflow edit swaps a full commit sha on an existing `uses:` or
# `ref:` line (or the comment naming it) and changes nothing else: no step,
# command, trigger, permission or logic, so it reaches no new contact surface.
_SHA_TOKEN = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")
_FULL_SHA = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{40}(?![0-9A-Za-z])")
_PIN_LINE = re.compile(r"^\s*(?:-\s+)?(?:uses|ref):\s")
_COMMENT_LINE = re.compile(r"^\s*#")


def _is_pin_only_hunk(removed: list[str], added: list[str]) -> bool:
    if not removed or len(removed) != len(added):
        return False
    for old, new in zip(removed, added, strict=True):
        if old == new or _SHA_TOKEN.sub("<sha>", old) != _SHA_TOKEN.sub("<sha>", new):
            return False
        if _PIN_LINE.match(new):
            if not (_FULL_SHA.search(old) and _FULL_SHA.search(new)):
                return False
        elif not _COMMENT_LINE.match(new):
            return False
    return True


def _is_pin_only_diff(diff: str) -> bool:
    """True when a zero-context diff of one modified file only swaps pins."""
    hunks: list[tuple[list[str], list[str]]] = []
    for line in diff.splitlines():
        if line.startswith("@@"):
            hunks.append(([], []))
        elif not hunks:
            continue
        elif line.startswith("-"):
            hunks[-1][0].append(line[1:])
        elif line.startswith("+"):
            hunks[-1][1].append(line[1:])
        else:
            return False
    return bool(hunks) and all(_is_pin_only_hunk(r, a) for r, a in hunks)


class HandlerCILiveContact:
    """Fail closed on unreadable inputs and on marker-only seam mocks."""

    def __init__(self, container: ModelONEXContainer | None = None) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelCILiveContactRequest
    ) -> ModelCILiveContactResult:
        repository = Path(request.repository).resolve()

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repository), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

        def read(path: str) -> str:
            # No shell, no working-tree read and no user-supplied revision syntax.
            pure = Path(path)
            if pure.is_absolute() or ".." in pure.parts or path.startswith("-"):
                raise ValueError(f"artifact must be a repository-relative path: {path}")
            revision = f"{head}:{path}" if request.base_ref else f":{path}"
            return git("show", revision)

        try:
            if request.base_ref:
                base = git(
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    f"{request.base_ref}^{{commit}}",
                ).strip()
                head = git(
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    f"{request.head_ref}^{{commit}}",
                ).strip()
                paths = git(
                    "diff",
                    "--no-renames",
                    "--name-only",
                    "-z",
                    f"{base}...{head}",
                    "--",
                ).split("\0")
                present = set(
                    git("ls-tree", "-r", "--name-only", "-z", head).split("\0")
                )
            else:
                paths = git(
                    "diff", "--cached", "--no-renames", "--name-only", "-z", "--"
                ).split("\0")
                present = set(git("ls-files", "-z").split("\0"))
            tooling = tuple(
                p for p in paths if p.startswith(("scripts/ci/", ".github/workflows/"))
            )
            if not tooling:
                return ModelCILiveContactResult(
                    success=True, reason="No CI-tooling change in the inspected diff"
                )
            revisions = (f"{base}...{head}",) if request.base_ref else ("--cached",)
            if all(
                p.startswith(".github/workflows/")
                and p.endswith((".yml", ".yaml"))
                and _is_pin_only_diff(
                    git(
                        "diff",
                        "--no-renames",
                        "--no-ext-diff",
                        "--no-color",
                        "--diff-filter=M",
                        "-U0",
                        *revisions,
                        "--",
                        p,
                    )
                )
                for p in tooling
            ):
                return ModelCILiveContactResult(
                    success=True,
                    reason="Pin-only workflow edit: every changed line swaps a full commit sha on an existing uses:/ref: line or its comment",
                    changed_tooling=tooling,
                )
            body = request.pr_body
            # The local hook has no PR event. A tracked PR body lets the same
            # explicit exception grammar apply locally; CI reads the real body.
            if (
                not request.base_ref
                and "PR_BODY.md" in paths
                and "PR_BODY.md" in present
            ):
                body = read("PR_BODY.md")
            body = re.sub(r"```.*?```|<!--.*?-->", "", body, flags=re.DOTALL)
            exception = _EXCEPTION.search(body)
            if exception and len(exception.group(1).split()) >= 8:
                return ModelCILiveContactResult(
                    success=True,
                    reason=f"Declared live-contact exception: {exception.group(1)}",
                    changed_tooling=tooling,
                )
            contacts: list[str] = []
            problems: list[str] = []
            for path in paths:
                if not (
                    path.startswith("tests/")
                    and Path(path).name.startswith("test_")
                    and path.endswith(".py")
                    and path in present
                ):
                    continue
                tree = ast.parse(read(path), filename=path)
                for test in ast.walk(tree):
                    if not isinstance(
                        test, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) or not test.name.startswith("test_"):
                        continue
                    for decorator in test.decorator_list:
                        if not (
                            isinstance(decorator, ast.Call)
                            and ast.unparse(decorator.func)
                            == "pytest.mark.live_contact"
                        ):
                            continue
                        label = f"{path}::{test.name}"
                        parameters = {
                            a.arg
                            for a in (
                                *test.args.posonlyargs,
                                *test.args.args,
                                *test.args.kwonlyargs,
                            )
                        }
                        calls = [
                            ast.unparse(n.func)
                            for n in ast.walk(test)
                            if isinstance(n, ast.Call)
                        ]
                        uses_recording = any(
                            isinstance(n, ast.Name)
                            and n.id == "recorded_response"
                            and isinstance(n.ctx, ast.Load)
                            for n in ast.walk(test)
                        )
                        if (
                            "recorded_response" not in parameters
                            or not uses_recording
                            or {"monkeypatch", "mocker"} & parameters
                            or any(
                                ast.unparse(d).startswith(
                                    ("pytest.mark.skip", "pytest.mark.xfail")
                                )
                                for d in test.decorator_list
                            )
                            or any(
                                name.rsplit(".", 1)[-1]
                                in {
                                    "Mock",
                                    "MagicMock",
                                    "AsyncMock",
                                    "patch",
                                    "setattr",
                                    "setitem",
                                    "skip",
                                    "xfail",
                                }
                                for name in calls
                            )
                        ):
                            problems.append(
                                f"{label}: marked test must consume recorded_response without an in-process seam replacement"
                            )
                            continue
                        if (
                            len(decorator.args) != 1
                            or not isinstance(decorator.args[0], ast.Constant)
                            or not isinstance(decorator.args[0].value, str)
                        ):
                            problems.append(
                                f"{label}: live_contact requires one literal artifact path"
                            )
                            continue
                        artifact = decorator.args[0].value
                        if artifact not in present:
                            problems.append(
                                f"{label}: missing recorded artifact {artifact}"
                            )
                            continue
                        recording = json.loads(read(artifact))
                        provenance = (
                            recording.get("_provenance")
                            if isinstance(recording, dict)
                            else None
                        )
                        if (
                            not isinstance(provenance, dict)
                            or not provenance.get("source")
                            or not any(
                                provenance.get(k)
                                for k in ("captured_utc", "source_sha256")
                            )
                        ):
                            problems.append(
                                f"{label}: recorded artifact lacks source and capture provenance"
                            )
                            continue
                        contacts.append(label)
            if contacts:
                return ModelCILiveContactResult(
                    success=True,
                    reason="Changed tests consume recorded live-contact evidence",
                    changed_tooling=tooling,
                    live_contact_tests=tuple(contacts),
                )
            reason = "CI-tooling change requires a live-contact test with a recorded response/replay, or a Live-contact exception: reason in the PR body"
            if problems:
                reason += "; " + "; ".join(problems)
            return ModelCILiveContactResult(
                success=False,
                reason=reason,
                changed_tooling=tooling,
                error_message=reason,
            )
        except (subprocess.CalledProcessError, OSError, ValueError, SyntaxError) as exc:
            reason = f"Cannot establish live-contact evidence: {exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                reason += f"; {exc.stderr.strip()}"
            return ModelCILiveContactResult(
                success=False, reason=reason, error_message=reason
            )
