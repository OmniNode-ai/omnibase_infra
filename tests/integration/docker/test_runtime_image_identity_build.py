# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime image identity stamping (OMN-12965).

These tests build the runtime-stage identity block (ARG + LABEL + workspace
guard) extracted verbatim from ``docker/Dockerfile.runtime`` against a tiny base
image and assert, via ``docker inspect``, that:

1. A workspace build WITH the identity quad populates
   ``org.opencontainers.image.version`` and ``.revision`` (the fixed behavior).
2. A workspace build WITHOUT the identity args FAILS the guard (the broken
   blank-identity image is refused, exit 64).
3. A release build without args is allowed to carry the placeholder default
   (no regression for release tooling).

This makes the local throwaway-build proof a permanent gate. The block under
test is the real runtime-stage logic, so a regression in the Dockerfile guard or
label wiring is caught here.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"


def _runtime_identity_block() -> str:
    """Extract the runtime-stage ARG/guard/LABEL identity block from the real Dockerfile.

    Pulls the exact lines under test so the integration build exercises the
    shipped logic, not a hand-copied paraphrase. Spans from the runtime-stage
    ``ARG RUNTIME_VERSION`` line through the end of the OCI ``LABEL`` block.
    """
    text = _DOCKERFILE.read_text(encoding="utf-8")
    # Runtime stage begins at the second `FROM python:...-slim`. Slice from the
    # runtime-stage RUNTIME_VERSION arg to the end of the first LABEL block after
    # it (the OCI image labels including version/revision).
    runtime_start = text.index("FROM python:${PYTHON_VERSION}-slim AS runtime")
    runtime_text = text[runtime_start:]
    arg_idx = runtime_text.index("ARG RUNTIME_VERSION=0.1.0")
    label_idx = runtime_text.index(
        'com.omninode.workspace_provenance_manifest="/app/build-provenance.json"'
    )
    # Extend to the end of that physical line.
    label_end = runtime_text.index("\n", label_idx)
    return runtime_text[arg_idx : label_end + 1]


def _write_proof_dockerfile(tmp_path: Path) -> Path:
    """Write a throwaway Dockerfile wrapping the real identity block on busybox."""
    block = _runtime_identity_block()
    # The block references BUILD_SOURCE/EXPECTED_BUILD_SOURCE/OMNI_HOME args that
    # the runtime stage declares earlier; declare them here so the slice builds.
    proof = "FROM busybox:latest AS runtime\n" + block
    dockerfile = tmp_path / "Dockerfile.identity-proof"
    dockerfile.write_text(proof, encoding="utf-8")
    return dockerfile


def _build(
    dockerfile: Path, context: Path, tag: str, build_args: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag]
    for key, value in build_args.items():
        cmd += ["--build-arg", f"{key}={value}"]
    cmd.append(str(context))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _inspect_label(tag: str, label: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            tag,
            "--format",
            f'{{{{index .Config.Labels "{label}"}}}}',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.integration
def test_workspace_build_with_identity_args_populates_labels(
    skip_if_no_docker: None, tmp_path: Path
) -> None:
    """Workspace build + identity quad → populated version + revision labels."""
    dockerfile = _write_proof_dockerfile(tmp_path)
    tag = f"omn12965-good-{uuid.uuid4().hex[:8]}"
    try:
        result = _build(
            dockerfile,
            tmp_path,
            tag,
            {
                "BUILD_SOURCE": "workspace",
                "EXPECTED_BUILD_SOURCE": "workspace",
                "OMNI_HOME": "/x",
                "RUNTIME_VERSION": "0.38.3",
                "VCS_REF": "abc123def456",
                "BUILD_DATE": "2026-06-11T00:00:00Z",
            },
        )
        assert result.returncode == 0, f"build failed: {result.stderr}"
        assert _inspect_label(tag, "org.opencontainers.image.version") == "0.38.3"
        assert (
            _inspect_label(tag, "org.opencontainers.image.revision") == "abc123def456"
        )
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            _inspect_label(tag, "org.opencontainers.image.created"),
        )
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, check=False)


@pytest.mark.integration
def test_workspace_build_without_identity_args_fails_guard(
    skip_if_no_docker: None, tmp_path: Path
) -> None:
    """Workspace build without identity args → guard fails (blank identity refused)."""
    dockerfile = _write_proof_dockerfile(tmp_path)
    tag = f"omn12965-bad-{uuid.uuid4().hex[:8]}"
    try:
        result = _build(
            dockerfile,
            tmp_path,
            tag,
            {
                "BUILD_SOURCE": "workspace",
                "EXPECTED_BUILD_SOURCE": "workspace",
                "OMNI_HOME": "/x",
            },
        )
        assert result.returncode != 0, (
            "workspace build with blank identity must fail the guard (OMN-12965)"
        )
        assert "OMN-12965" in (result.stderr + result.stdout)
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, check=False)


@pytest.mark.integration
def test_release_build_allows_placeholder_default(
    skip_if_no_docker: None, tmp_path: Path
) -> None:
    """Release build without args keeps the placeholder default (no regression)."""
    dockerfile = _write_proof_dockerfile(tmp_path)
    tag = f"omn12965-rel-{uuid.uuid4().hex[:8]}"
    try:
        result = _build(dockerfile, tmp_path, tag, {"BUILD_SOURCE": "release"})
        assert result.returncode == 0, f"release build failed: {result.stderr}"
        assert _inspect_label(tag, "org.opencontainers.image.version") == "0.1.0"
    finally:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, check=False)
