# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the onex-api image build handler (OMN-23521).

A fake ``run`` records the argv it was called with, so the tests assert the
exact docker build line (both labels and the build arg), the cwd, and that no
argv contains k3s, kubectl or ctr. Refusals are asserted to make no run call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.handlers.handler_onex_api_image_build import (
    HandlerOnexApiImageBuild,
)
from omnibase_infra.models.model_onex_api_image_build_request import (
    ModelOnexApiImageBuildRequest,
)

pytestmark = pytest.mark.unit

_SHA = "a" * 40
_STAMP = "20260101T000000Z"
_CAPTURE_SHA = "b" * 40
_TREE = Path("work/source-tree")


def _request(**overrides: Any) -> ModelOnexApiImageBuildRequest:
    base: dict[str, Any] = {
        "sha": _SHA,
        "stamp": _STAMP,
        "capture_sha": _SHA,
        "runtime_image": "runtime:1",
        "capture_error": None,
        "source_tree": _TREE,
        "source_revision": _SHA,
        "image_repository": "onex-api",
        "dockerfile": "Dockerfile",
        "context": ".",
        "runtime_image_build_arg": "RUNTIME_IMAGE",
        "revision_label": "omni.revision",
        "source_label": "omni.source",
        "source_repo": "git@github.com:omni/onex-api.git",
        "timeout_seconds": 600,
    }
    base.update(overrides)
    return ModelOnexApiImageBuildRequest(**base)


def test_golden_build_records_the_exact_argv() -> None:
    calls: list[dict[str, Any]] = []

    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "timeout": timeout, "cwd": cwd, "check": check})
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    handler = HandlerOnexApiImageBuild(run=run)
    result = handler.handle(_request())
    assert result.built is True
    assert result.image == f"onex-api:{_SHA[:8]}-{_STAMP}"
    assert result.source_revision == _SHA
    assert result.reason.startswith("BUILT:")
    assert len(calls) == 1
    argv = calls[0]["argv"]
    assert argv == [
        "docker",
        "build",
        "-f",
        "Dockerfile",
        "-t",
        f"onex-api:{_SHA[:8]}-{_STAMP}",
        "--label",
        f"omni.revision={_SHA}",
        "--label",
        "omni.source=git@github.com:omni/onex-api.git",
        "--build-arg",
        "RUNTIME_IMAGE=runtime:1",
        ".",
    ]
    assert calls[0]["cwd"] == _TREE
    assert calls[0]["timeout"] == 600
    # The one call is the host daemon build: no cluster client, no import, no
    # apply script. ``check`` is off so the handler reads the exit code itself.
    assert calls[0]["check"] is False


def test_a_build_that_exits_nonzero_is_build_failed_with_its_tail() -> None:
    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=2, stdout="", stderr="step 4/9: no such file"
        )

    result = HandlerOnexApiImageBuild(run=run).handle(_request())
    assert result.built is False
    assert result.image == ""
    assert result.source_revision == _SHA
    assert (
        result.reason == "BUILD_FAILED: docker build exited 2: step 4/9: no such file"
    )


def test_a_build_that_cannot_start_keeps_the_source_revision() -> None:
    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, timeout)

    handler = HandlerOnexApiImageBuild(run=run)
    result = handler.handle(_request())
    assert result.built is False
    assert result.image == ""
    assert result.source_revision == _SHA
    assert result.reason.startswith("BUILD_FAILED:")


def test_capture_sha_mismatch_refuses_without_a_run_call() -> None:
    calls: list[Any] = []

    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    handler = HandlerOnexApiImageBuild(run=run)
    result = handler.handle(_request(capture_sha=_CAPTURE_SHA))
    assert result.built is False
    assert result.source_revision == _SHA
    assert result.reason.startswith("REFUSED_CAPTURE_SHA_MISMATCH: ")
    assert _CAPTURE_SHA[:12] in result.reason
    assert _SHA[:12] in result.reason
    assert calls == []


def test_capture_error_refuses_without_a_run_call() -> None:
    calls: list[Any] = []

    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    handler = HandlerOnexApiImageBuild(run=run)
    result = handler.handle(_request(capture_error="capture blew up"))
    assert result.built is False
    assert result.source_revision == _SHA
    assert result.reason.startswith("REFUSED_NO_RUNTIME_IMAGE: ")
    assert "capture blew up" in result.reason
    assert calls == []


def test_short_source_revision_refuses_without_a_run_call() -> None:
    calls: list[Any] = []

    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    handler = HandlerOnexApiImageBuild(run=run)
    result = handler.handle(_request(source_revision="abc"))
    assert result.built is False
    assert result.source_revision is None
    assert result.reason.startswith("REFUSED_NO_SOURCE_REVISION: ")
    assert calls == []
