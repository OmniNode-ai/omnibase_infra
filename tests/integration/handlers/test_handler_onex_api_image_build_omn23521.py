# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler and real-subprocess integration for the onex-api image build (OMN-23521).

Loads node_onex_api_image_build_effect's contract from disk, resolves its routed
operation and models to the declared classes, then drives the deploy agent's one
entry, ``build_onex_api_image``, through a real subprocess standing in for the host
``docker build``: the golden chain sees the build run in the archived tree with the
exact argv, the error chain sees a non-zero exit come back as ``BUILD_FAILED`` with
its stderr tail and its lineage. No Docker daemon, broker or deploy agent is touched.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_onex_api_image_build import (
    HandlerOnexApiImageBuild,
    build_onex_api_image,
)
from omnibase_infra.models.model_onex_api_image_build_request import (
    ModelOnexApiImageBuildRequest,
)
from omnibase_infra.models.model_onex_api_image_build_result import (
    ModelOnexApiImageBuildResult,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_onex_api_image_build_effect/contract.yaml"
)
_SHA = "a" * 40
_REVISION = "c" * 40
_STAMP = "20261110T120000Z"

#: Stands in for ``docker``: records its argv and cwd in the tree it ran in, and
#: exits with the code the test asks for, writing a stderr line when non-zero.
_STAND_IN = (
    "import json, os, pathlib, sys\n"
    "code = int(os.environ['STAND_IN_EXIT'])\n"
    "pathlib.Path('ran.json').write_text(json.dumps("
    "{'argv': sys.argv[1:], 'cwd': os.getcwd()}))\n"
    "if code: sys.stderr.write('step 4/9: COPY failed: no such file')\n"
    "sys.exit(code)\n"
)


def _runner(exit_code: int) -> Any:
    def run(
        argv: list[str], *, timeout: int, cwd: Path, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "docker"
        assert check is False
        return subprocess.run(
            [sys.executable, "-c", _STAND_IN, *argv[1:]],
            cwd=cwd,
            env={"STAND_IN_EXIT": str(exit_code), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return run


def _request(tree: Path) -> ModelOnexApiImageBuildRequest:
    return ModelOnexApiImageBuildRequest(
        sha=_SHA,
        stamp=_STAMP,
        capture_sha=_SHA,
        runtime_image=f"runtime:{_STAMP}-{_SHA[:8]}",
        capture_error=None,
        source_tree=tree,
        source_revision=_REVISION,
        image_repository="onex-api-image",
        dockerfile="docker/onex-api/Dockerfile",
        context="docker/onex-api",
        runtime_image_build_arg="RUNTIME_IMAGE",
        revision_label="org.opencontainers.image.revision",
        source_label="ai.omninode.image.source-repo",
        source_repo="overlay-source",
        timeout_seconds=60,
    )


def test_contract_routes_the_operation_to_the_handler_and_models() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    route = contract["handler_routing"]["handlers"][0]
    module = importlib.import_module(route["handler"]["module"])
    assert getattr(module, route["handler"]["name"]) is HandlerOnexApiImageBuild
    for key, cls in (
        ("input_model", ModelOnexApiImageBuildRequest),
        ("output_model", ModelOnexApiImageBuildResult),
    ):
        declared = importlib.import_module(contract[key]["module"])
        assert getattr(declared, contract[key]["name"]) is cls


def test_golden_chain_builds_in_the_archived_tree(tmp_path: Path) -> None:
    result = build_onex_api_image(_runner(0), _request(tmp_path))
    assert result.built is True
    assert result.image == f"onex-api-image:{_REVISION[:8]}-{_STAMP}"
    assert result.source_revision == _REVISION
    ran = json.loads((tmp_path / "ran.json").read_text())
    assert Path(ran["cwd"]) == tmp_path
    assert ran["argv"] == [
        "build",
        "-f",
        "docker/onex-api/Dockerfile",
        "-t",
        result.image,
        "--label",
        f"org.opencontainers.image.revision={_REVISION}",
        "--label",
        "ai.omninode.image.source-repo=overlay-source",
        "--build-arg",
        f"RUNTIME_IMAGE=runtime:{_STAMP}-{_SHA[:8]}",
        "docker/onex-api",
    ]


def test_error_chain_reports_the_failed_build_with_its_lineage(tmp_path: Path) -> None:
    result = build_onex_api_image(_runner(3), _request(tmp_path))
    assert result.built is False
    assert result.image == ""
    assert result.source_revision == _REVISION
    assert result.reason == (
        "BUILD_FAILED: docker build exited 3: step 4/9: COPY failed: no such file"
    )
