# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A per-deploy build argument is declared after the layers that do not use it.

A declared ``ARG`` is part of the build environment of every ``RUN`` after it in
its stage. ``BUILD_DATE``, ``VCS_REF`` and ``RUNTIME_VERSION`` change on every
deploy, so declaring them at the top of a stage made the apt, node, coding-agent
CLI and venv layers miss the cache on every dev build: 42 of 48 image layers
differed between the two dev builds of 2026-11-24 (job d1afd57d against job
7f7f5bcd on the dev lane), and the image build was 408 s and 481 s of the 710 s
and 746 s from accept to a healthy container.

The test reads ``docker/Dockerfile.runtime`` and fails when one of those
arguments is declared before a cache-mounted ``RUN``, ``apt-get`` layer or
builder-venv ``COPY`` that precedes the stage's ``COPY src/`` (every layer after
that one is rebuilt for each commit anyway), or when a reference to one has no
declaration before it in its stage.

Ticket: OMN-26316
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

PER_DEPLOY_ARGS = ("BUILD_DATE", "VCS_REF", "RUNTIME_VERSION")

_ARG_RE = re.compile(r"^ARG\s+(?P<name>[A-Z_][A-Z0-9_]*)")
_REF_RE = re.compile(r"\$\{?(?P<name>[A-Z_][A-Z0-9_]*)")


@dataclass(frozen=True)
class Instruction:
    stage: str
    index: int
    keyword: str
    text: str


def _instructions(dockerfile: str) -> list[Instruction]:
    """Split a Dockerfile into instructions, joining continuations, dropping comments."""
    found: list[Instruction] = []
    stage = ""
    pending = ""
    for raw in dockerfile.splitlines():
        stripped = raw.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if pending and stripped.startswith("#"):
            continue
        joined = f"{pending} {stripped}" if pending else stripped
        if joined.endswith("\\"):
            pending = joined[:-1].rstrip()
            continue
        pending = ""
        keyword = joined.split(None, 1)[0].upper()
        if keyword == "FROM":
            match = re.search(r"\bAS\s+(\S+)\s*$", joined, re.IGNORECASE)
            stage = match.group(1) if match else f"stage-{len(found)}"
        found.append(Instruction(stage, len(found), keyword, joined))
    return found


def _is_commit_boundary(instruction: Instruction) -> bool:
    """``COPY src/ ...`` from the build context: every layer after it is rebuilt per commit."""
    return (
        instruction.keyword == "COPY"
        and "--from=" not in instruction.text
        and re.search(r"\ssrc/\s", instruction.text + " ") is not None
    )


def _is_heavy(instruction: Instruction) -> bool:
    text = instruction.text
    if instruction.keyword == "RUN":
        return "--mount=type=cache" in text or "apt-get" in text
    if instruction.keyword == "COPY":
        return "--from=builder" in text and "/app/.venv /app/.venv" in text
    return False


def cache_order_violations(dockerfile: str) -> list[str]:
    """Return one line per per-deploy ARG that sits ahead of a cacheable layer."""
    violations: list[str] = []
    declared_at: dict[tuple[str, str], int] = {}
    past_boundary: set[str] = set()
    for instruction in _instructions(dockerfile):
        if _is_commit_boundary(instruction):
            past_boundary.add(instruction.stage)
        if instruction.keyword == "ARG":
            match = _ARG_RE.match(instruction.text)
            if match and match.group("name") in PER_DEPLOY_ARGS:
                declared_at.setdefault(
                    (instruction.stage, match.group("name")), instruction.index
                )
            continue
        if instruction.keyword == "RUN" or instruction.keyword in {"LABEL", "ENV"}:
            for ref in _REF_RE.finditer(instruction.text):
                name = ref.group("name")
                if (
                    name in PER_DEPLOY_ARGS
                    and (instruction.stage, name) not in declared_at
                ):
                    violations.append(
                        f"{instruction.stage}: {name} is used at instruction "
                        f"{instruction.index} with no ARG {name} before it"
                    )
        if _is_heavy(instruction) and instruction.stage not in past_boundary:
            for (stage, name), index in declared_at.items():
                if stage == instruction.stage and index < instruction.index:
                    violations.append(
                        f"{stage}: ARG {name} (instruction {index}) is declared before the "
                        f"cacheable layer at instruction {instruction.index}: {instruction.text[:80]}"
                    )
    return violations


def test_per_deploy_args_follow_every_cacheable_layer(dockerfile_content: str) -> None:
    assert cache_order_violations(dockerfile_content) == []


def test_every_per_deploy_arg_is_still_declared_for_its_consumers(
    dockerfile_content: str,
) -> None:
    instructions = _instructions(dockerfile_content)
    for stage, name in (
        ("builder", "BUILD_DATE"),
        ("builder", "VCS_REF"),
        ("runtime", "RUNTIME_VERSION"),
        ("runtime", "BUILD_DATE"),
        ("runtime", "VCS_REF"),
    ):
        assert any(
            i.stage == stage
            and i.keyword == "ARG"
            and _ARG_RE.match(i.text)
            and name in i.text.split()[1]
            for i in instructions
        ), f"{stage} stage no longer declares ARG {name}"


def test_runtime_stage_has_cacheable_layers_for_the_check_to_guard(
    dockerfile_content: str,
) -> None:
    """Positive control: the walk sees the apt and venv layers it protects."""
    runtime = [i for i in _instructions(dockerfile_content) if i.stage == "runtime"]
    assert sum(1 for i in runtime if _is_heavy(i)) >= 2
    assert any(
        i.keyword == "COPY" and "/app/.venv /app/.venv" in i.text for i in runtime
    )


def test_checker_flags_the_pre_fix_layout() -> None:
    """Negative control: the layout that cost the cache is reported."""
    pre_fix = "\n".join(
        [
            "FROM python:3.12-slim AS runtime",
            "ARG RUNTIME_VERSION=0.1.0",
            "ARG BUILD_DATE",
            "ARG VCS_REF",
            "RUN apt-get update && apt-get install -y curl",
            "COPY --from=builder --chown=omniinfra:omniinfra /app/.venv /app/.venv",
            'LABEL org.opencontainers.image.revision="${VCS_REF}"',
        ]
    )
    violations = cache_order_violations(pre_fix)
    assert len(violations) == 6, violations
    assert any("ARG BUILD_DATE" in v and "apt-get" in v for v in violations)


def test_checker_flags_a_use_with_no_declaration() -> None:
    undeclared = "\n".join(
        [
            "FROM python:3.12-slim AS runtime",
            'LABEL org.opencontainers.image.created="${BUILD_DATE}"',
        ]
    )
    assert cache_order_violations(undeclared) == [
        "runtime: BUILD_DATE is used at instruction 1 with no ARG BUILD_DATE before it"
    ]


def test_checker_accepts_the_declaration_after_the_heavy_layers() -> None:
    fixed = "\n".join(
        [
            "FROM python:3.12-slim AS runtime",
            "RUN apt-get update && apt-get install -y curl",
            "COPY --from=builder --chown=omniinfra:omniinfra /app/.venv /app/.venv",
            "ARG VCS_REF",
            'LABEL org.opencontainers.image.revision="${VCS_REF}"',
        ]
    )
    assert cache_order_violations(fixed) == []


def test_dockerfile_fixture_is_the_runtime_dockerfile(dockerfile_path: Path) -> None:
    assert dockerfile_path.name == "Dockerfile.runtime"
