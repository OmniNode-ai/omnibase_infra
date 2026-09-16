# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner image must build for both fleet architectures (OMN-17477).

The lab fleet stopped being a single amd64 host. `.101` (Mac mini, M2 Pro) and
`.105` (MacBook Air, M4) are arm64, so an image whose external binary downloads
are pinned to amd64 cannot run there at all -- it is not a slow image on those
hosts, it is an image that fails at build time on the first `curl`.

These tests pin the two halves of the fix:

1. Every external binary download resolves its architecture from Docker's
   ``TARGETARCH`` build argument rather than a literal. A literal is the whole
   defect and is the thing that silently comes back when someone adds a tool.
2. The CI build proof actually builds BOTH platforms. A multi-arch Dockerfile
   that only ever gets built for one arch is an untested claim: the arm64 leg
   breaks on an apt package that has no arm64 build and nobody finds out until
   a host is being provisioned.

Deliberately a TEXT-level test on the Dockerfile rather than a container probe:
the failure it guards is a source-level literal, and a text test is the only
one that can run in the hermetic unit lane where the fleet has no builder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"
BUILD_SMOKE_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "runner-image-build-smoke.yml"
)

# Architecture literals that must never appear inside a DOWNLOAD URL or a
# release-asset file name. `x64`/`X64` are excluded: they are also GitHub's own
# runner LABEL for an amd64 host, which the amd64 fleet legitimately still sets.
ARCH_LITERALS = ("amd64", "x86_64", "linux-x64", "aarch64", "arm64")

# Markers of a line that RESOLVES an architecture rather than pinning one.
ARCH_RESOLVING_MARKERS = ("TARGETARCH", "dpkg --print-architecture")

# The per-tool architecture maps. Four upstreams spell the same two CPUs four
# different ways, so the map is the fix, not a violation -- but it is only the
# fix if BOTH arms of every entry exist, which is what the map test below pins.
ARCH_MAPS: dict[str, dict[str, str]] = {
    "UV_ARCH": {
        "amd64": "x86_64-unknown-linux-gnu",
        "arm64": "aarch64-unknown-linux-gnu",
    },
    "AWS_ARCH": {"amd64": "x86_64", "arm64": "aarch64"},
    "RUNNER_ARCH": {"amd64": "x64", "arm64": "arm64"},
}


def _dockerfile_lines() -> list[tuple[int, str]]:
    return [
        (n, line)
        for n, line in enumerate(DOCKERFILE.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _download_lines() -> list[tuple[int, str]]:
    """Lines that name a remote artifact: a URL, or a path extracted from one."""
    return [
        (n, line)
        for n, line in _dockerfile_lines()
        if "http" in line or "--strip-components" in line
    ]


def test_dockerfile_declares_targetarch() -> None:
    """``ARG TARGETARCH`` must be declared in the build stage that uses it.

    Docker pre-declares TARGETARCH as a global build arg, but a stage only sees
    it after its own `ARG TARGETARCH`. Without the redeclaration every
    substitution silently expands to the empty string and the URLs 404.
    """
    body = DOCKERFILE.read_text(encoding="utf-8")
    after_from = body[body.index("\nFROM ") :]
    assert re.search(r"^ARG TARGETARCH\b", after_from, re.MULTILINE), (
        "docker/runners/Dockerfile must redeclare `ARG TARGETARCH` after FROM; "
        "a global ARG is not visible inside a build stage and expands empty."
    )


@pytest.mark.parametrize("literal", ARCH_LITERALS)
def test_no_hardcoded_arch_in_download_urls(literal: str) -> None:
    """No download URL or extracted asset path may name an architecture."""
    offenders = [
        f"{n}: {line.strip()}"
        for n, line in _download_lines()
        if literal in line
        and not any(marker in line for marker in ARCH_RESOLVING_MARKERS)
    ]
    assert not offenders, (
        f"docker/runners/Dockerfile pins the architecture literal {literal!r} in "
        "a download URL; resolve it from TARGETARCH so the image builds on the "
        "arm64 lab hosts.\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("prefix", sorted(ARCH_MAPS))
def test_arch_map_declares_both_architectures(prefix: str) -> None:
    """Each per-tool map has an arm arm AND an amd arm, with the right spelling.

    A half-populated map is worse than a literal: the indirect lookup resolves
    to the empty string and the build fails several layers later on a 404 with
    no mention of the architecture.
    """
    body = DOCKERFILE.read_text(encoding="utf-8")
    for arch, value in ARCH_MAPS[prefix].items():
        assert re.search(
            rf"^ARG {prefix}_{arch}={re.escape(value)}$", body, re.MULTILINE
        ), f"docker/runners/Dockerfile must declare ARG {prefix}_{arch}={value}"


def test_runner_tarball_checksum_is_declared_per_architecture() -> None:
    """Both runner tarballs are checksum-verified, not just the amd64 one.

    Dropping the check on the arm64 leg would be the easy way to make this
    file pass, and it would remove the supply-chain verification that is the
    reason the amd64 leg has a pinned digest at all.
    """
    body = DOCKERFILE.read_text(encoding="utf-8")
    for var in ("RUNNER_SHA256_AMD64", "RUNNER_SHA256_ARM64"):
        assert re.search(rf"^ENV {var}=[0-9a-f]{{64}}$", body, re.MULTILINE), (
            f"docker/runners/Dockerfile must declare {var} as a 64-hex digest"
        )
    assert "sha256sum --check --strict" in body, (
        "the runner tarball must still be checksum-verified after the multi-arch change"
    )


def test_build_smoke_workflow_builds_both_architectures() -> None:
    """The CI proof builds linux/amd64 AND linux/arm64 under one manifest."""
    workflow = yaml.safe_load(BUILD_SMOKE_WORKFLOW.read_text(encoding="utf-8"))
    platform_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "linux/arm64" in str(step.get("with", {}).get("platforms", ""))
        or "linux/arm64" in str(step.get("run", ""))
    ]
    assert platform_steps, (
        ".github/workflows/runner-image-build-smoke.yml must build linux/arm64 "
        "as well as linux/amd64; a multi-arch Dockerfile nothing builds for "
        "arm64 is an unproven claim"
    )
    joined = "\n".join(str(step) for step in platform_steps)
    assert "linux/amd64" in joined, (
        "the arm64 build must not replace the amd64 build; both platforms "
        "belong in the same buildx invocation"
    )
