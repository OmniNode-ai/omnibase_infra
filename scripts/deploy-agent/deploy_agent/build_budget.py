# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Derive the runtime IMAGE-BUILD ceiling from the build model (OMN-18072).

WHY THIS EXISTS
---------------

OMN-18057 derived the runtime **compose-up** ceiling from the compose model and
deliberately left ``PHASE_TIMEOUTS[Phase.RUNTIME] = 300`` bounding the runtime
**image build**, the pinned-digest pull and the migration preflight. The build
half of that constant was wrong for the same reason the compose-up half was: it
is a number with no relationship to the work it bounds.

MEASURED, from the agent's own job history on the dev lane (state dir
``/data/omninode/deploy-agent/state/jobs-dev``, journal ``deploy-agent-dev``):

    command   6c323639  build start 2026-09-09T01:44:30.717Z
                        first core container Created 2026-09-09T01:48:24.652Z
                        => runtime image build <= 233.9s, job SUCCEEDED
    command   79171e79  build start 2026-09-09T09:05:39.640Z
                        all 9 runtime images exported 09:09:08.891Z (t+209.3s)
                        killed 09:10:39.674Z => 300.0s, command had NOT returned
    command   2788af33  build start 2026-09-09T09:12:08.616Z
                        killed 09:17:08.902Z => 300.3s, WARM BuildKit cache

The warm-cache repeat is the load-bearing observation: the cache is not the
variable. A ``BUILD_SOURCE=workspace`` build of this image over three staged
sibling repos runs within a couple of minutes of the constant that bounds it,
so ordinary host load decides whether the deploy lives or dies. Two of the
three observations are RIGHT-CENSORED at 300s -- the build was killed, not
measured -- which is precisely why a p95 over recorded builds is not derivable
from this history today and the ceiling is derived from the build MODEL
instead, the way ``deploy_agent.compose_budget`` derives from the compose model.

THE MODEL
---------

``docker compose --profile <p> build`` does two separable things:

1. ONE BuildKit solve over the shared Dockerfile and context. Every buildable
   service in the runtime profile declares the same ``context: ..`` and the
   same ``dockerfile: docker/Dockerfile.runtime``, so there is a single solve
   whose cost scales with the number of instructions in that Dockerfile -- not
   with the number of services. (Proof it is one solve: all nine dev-lane
   images carry the byte-identical config creation timestamp
   ``2026-09-09T05:09:08.891295047-04:00``.)
2. One image export per selected service off that solve.

So:

    ceiling = max(floor, build_steps * per_step + buildable_services * per_image)

Both terms are READ from the same files the build is about to invoke, so adding
a runtime service or an expensive Dockerfile step moves the ceiling with it
instead of silently re-opening this failure one number later.

THE FLOOR
---------

The floor is twice the constant that was proven insufficient. A compose or
Dockerfile model that read as unexpectedly small must never re-derive a ceiling
at or below the one two consecutive deploys already died on.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from deploy_agent.compose_budget import ComposeBudgetError, _ComposeLoader

logger = logging.getLogger(__name__)

# Dockerfile instructions that can execute work or move bytes, and therefore
# cost wall time in a solve. Metadata-only instructions (FROM/ARG/ENV/LABEL/
# WORKDIR/USER/EXPOSE/CMD/ENTRYPOINT/HEALTHCHECK/SHELL/VOLUME/STOPSIGNAL) are
# excluded: they are resolved without doing any of the work this bounds.
_WORK_INSTRUCTIONS = frozenset({"RUN", "COPY", "ADD"})

_INSTRUCTION_RE = re.compile(r"^\s*([A-Za-z]+)\b")
# `RUN cat > f <<'EOF'` / `<<EOF` / `<<-"EOF"`: the body that follows is shell
# input, not Dockerfile instructions, and must not be scanned for RUN/COPY.
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


class BuildBudgetError(RuntimeError):
    """Raised when the build model cannot be read to derive a ceiling."""


class ModelBuildBudget(BaseModel):
    """A derived image-build ceiling and the build facts it was derived from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: int
    floor_seconds: int
    per_step_seconds: int
    per_image_seconds: int
    build_steps: int
    buildable_services: tuple[str, ...]
    dockerfile: str | None
    profile: str
    compose_files: tuple[str, ...]

    def describe(self) -> str:
        """One-line, log-ready statement of the ceiling and its source."""
        if self.dockerfile is None:
            return (
                f"{self.timeout_seconds}s (floor {self.floor_seconds}s; no service "
                f"in profile {self.profile!r} declares a build: section, so there "
                f"is no image build to bound)"
            )
        return (
            f"{self.timeout_seconds}s = {self.build_steps} work steps of "
            f"{self.dockerfile!r} x {self.per_step_seconds}s/step (one shared "
            f"BuildKit solve) + {len(self.buildable_services)} buildable "
            f"service(s) in profile {self.profile!r} x {self.per_image_seconds}s/"
            f"image (export), floor {self.floor_seconds}s"
        )


def count_build_steps(dockerfile: Path | str) -> int:
    """Count the work-performing instructions (RUN/COPY/ADD) in a Dockerfile.

    Backslash continuations and heredoc bodies are consumed rather than
    scanned, so a shell line inside a ``RUN ... <<'EOF'`` block that happens to
    begin with ``RUN`` is not counted as a second instruction.

    Raises ``BuildBudgetError`` when the file cannot be read: a ceiling that
    silently fell back to its floor because the model was unreadable is the
    same class of undetectable wrongness as the bare constant this replaces.
    """
    path = Path(dockerfile)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BuildBudgetError(
            f"cannot read Dockerfile {path} to derive the image-build ceiling: {exc}"
        ) from exc

    steps = 0
    continuing = False
    heredoc_terminator: str | None = None
    for raw_line in text.splitlines():
        if heredoc_terminator is not None:
            if raw_line.strip() == heredoc_terminator:
                heredoc_terminator = None
            continue
        stripped = raw_line.strip()
        if continuing:
            continuing = stripped.endswith("\\")
            continue
        if not stripped or stripped.startswith("#"):
            continue
        match = _INSTRUCTION_RE.match(stripped)
        if match is not None and match.group(1).upper() in _WORK_INSTRUCTIONS:
            steps += 1
        heredoc = _HEREDOC_RE.search(stripped)
        if heredoc is not None:
            heredoc_terminator = heredoc.group(2)
            continue
        continuing = stripped.endswith("\\")
    return steps


def _load_compose_documents(compose_files: tuple[str, ...] | list[str]) -> list[Any]:
    documents: list[dict[str, Any]] = []
    for compose_file in compose_files:
        path = Path(compose_file)
        try:
            document = yaml.load(
                path.read_text(encoding="utf-8"),
                Loader=_ComposeLoader,  # noqa: S506 -- see compose_budget
            )
        except OSError as exc:
            raise BuildBudgetError(
                f"cannot read compose file {path} to derive the image-build "
                f"ceiling: {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise BuildBudgetError(
                f"cannot parse compose file {path} to derive the image-build "
                f"ceiling: {exc}"
            ) from exc
        if isinstance(document, dict):
            documents.append(document)
    return documents


def _buildable_services(
    documents: list[dict[str, Any]], profile: str
) -> list[tuple[str, dict[str, Any]]]:
    """Return (name, build-spec) for every service the profile selects that builds."""
    merged: dict[str, dict[str, Any]] = {}
    for document in documents:
        services = document.get("services") or {}
        if not isinstance(services, dict):
            continue
        for name, spec in services.items():
            if isinstance(spec, dict):
                merged.setdefault(str(name), {}).update(spec)

    selected: list[tuple[str, dict[str, Any]]] = []
    for name, spec in merged.items():
        profiles = spec.get("profiles")
        # A service with no profiles is always selected; compose only excludes a
        # service that declares profiles none of which are active.
        if isinstance(profiles, list) and profile not in profiles:
            continue
        build = spec.get("build")
        if isinstance(build, str):
            build = {"context": build}
        if isinstance(build, dict):
            selected.append((name, build))
    return sorted(selected)


def _resolve_dockerfiles(
    compose_file: str, builds: list[tuple[str, dict[str, Any]]]
) -> dict[str, Path]:
    """Map each buildable service to its Dockerfile path.

    Compose resolves ``build.context`` relative to the compose file's own
    directory and ``build.dockerfile`` relative to the resolved context, which
    is how the runtime services' ``context: ..`` + ``dockerfile:
    docker/Dockerfile.runtime`` lands back on the repo's own Dockerfile.
    """
    base = Path(compose_file).resolve().parent
    resolved: dict[str, Path] = {}
    for name, build in builds:
        context = (base / str(build.get("context", "."))).resolve()
        resolved[name] = (
            context / str(build.get("dockerfile", "Dockerfile"))
        ).resolve()
    return resolved


def derive_image_build_budget(
    compose_files: tuple[str, ...] | list[str],
    profile: str,
    *,
    per_step_seconds: int,
    per_image_seconds: int,
    floor_seconds: int,
) -> ModelBuildBudget:
    """Derive the ``docker compose --profile <profile> build`` ceiling.

    The solve term is counted once over the LONGEST Dockerfile any selected
    service builds from, because compose issues one solve per distinct
    Dockerfile+context and the runtime profile's services all share one. When a
    profile mixes Dockerfiles the longest is the honest bound for the shared
    solve, and the per-image term still scales with the service count.

    Raises ``BuildBudgetError`` when a declared compose file or Dockerfile
    cannot be read or parsed.
    """
    documents = _load_compose_documents(compose_files)
    builds = _buildable_services(documents, profile)

    if not builds:
        return ModelBuildBudget(
            timeout_seconds=floor_seconds,
            floor_seconds=floor_seconds,
            per_step_seconds=per_step_seconds,
            per_image_seconds=per_image_seconds,
            build_steps=0,
            buildable_services=(),
            dockerfile=None,
            profile=profile,
            compose_files=tuple(str(f) for f in compose_files),
        )

    if not compose_files:
        raise BuildBudgetError(
            "cannot derive the image-build ceiling without at least one compose file"
        )
    dockerfiles = _resolve_dockerfiles(str(compose_files[0]), builds)

    source_dockerfile: Path | None = None
    build_steps = 0
    for path in sorted(set(dockerfiles.values())):
        steps = count_build_steps(path)
        if steps > build_steps:
            build_steps = steps
            source_dockerfile = path

    derived = build_steps * per_step_seconds + len(builds) * per_image_seconds
    return ModelBuildBudget(
        timeout_seconds=max(floor_seconds, derived),
        floor_seconds=floor_seconds,
        per_step_seconds=per_step_seconds,
        per_image_seconds=per_image_seconds,
        build_steps=build_steps,
        buildable_services=tuple(name for name, _ in builds),
        dockerfile=str(source_dockerfile) if source_dockerfile else None,
        profile=profile,
        compose_files=tuple(str(f) for f in compose_files),
    )
