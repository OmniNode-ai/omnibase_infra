# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler and real-interpreter integration for the onex-api import check (OMN-23039).

Loads node_onex_api_image_import_check_effect's contract from disk, resolves its routed
operation to the declared handler class, then runs the handler's own check script in a
real Python subprocess over a fixture application tree, standing in for the candidate
image's interpreter. A missing omnimarket module rolls the pin back; a present one keeps
it. No Docker daemon, broker or deploy agent is touched.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_onex_api_image_import_check import (
    IMPORT_CHECK_SCRIPT,
    HandlerOnexApiImageImportCheck,
)
from omnibase_infra.models.model_onex_api_image_import_request import (
    ModelOnexApiImageImportRequest,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_onex_api_image_import_check_effect/contract.yaml"
)
_OLD = "old-pin"
_NEW = "new-pin"


def _app(tmp_path: Path, import_line: str) -> tuple[Path, Path]:
    site = tmp_path / "site"
    (site / "omnimarket" / "routing").mkdir(parents=True)
    (site / "omnimarket" / "__init__.py").write_text("")
    (site / "omnimarket" / "routing" / "__init__.py").write_text("")
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text(import_line + "\n")
    return site, app


def _runner(site: Path, app: Path) -> Any:
    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "docker"
        env = {**os.environ, "PYTHONPATH": str(site), "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run(
            [sys.executable, "-c", IMPORT_CHECK_SCRIPT],
            cwd=app,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return run


def _request(tmp_path: Path) -> ModelOnexApiImageImportRequest:
    env_file = tmp_path / "env"
    env_file.write_text(f"A=1\nONEX_API_IMAGE={_NEW}\nB=2\n")
    return ModelOnexApiImageImportRequest(
        image=_NEW, env_file=env_file, previous_pin=_OLD
    )


def test_contract_routes_the_operation_to_the_handler() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    route = contract["handler_routing"]["handlers"][0]
    module = importlib.import_module(route["handler"]["module"])
    assert getattr(module, route["handler"]["name"]) is HandlerOnexApiImageImportCheck


def test_a_moved_module_rolls_the_pin_back(tmp_path: Path) -> None:
    site, app = _app(tmp_path, "from omnimarket.routing.gone import X")
    request = _request(tmp_path)
    verdict = HandlerOnexApiImageImportCheck(run=_runner(site, app)).handle(request)
    assert not verdict.importable
    assert verdict.rolled_back
    assert verdict.reason.startswith("IMPORT_CHECK_FAILED_ROLLED_BACK")
    assert request.env_file.read_text() == f"A=1\nONEX_API_IMAGE={_OLD}\nB=2\n"


def test_a_present_module_keeps_the_advanced_pin(tmp_path: Path) -> None:
    site, app = _app(tmp_path, "from omnimarket.routing import something")
    request = _request(tmp_path)
    verdict = HandlerOnexApiImageImportCheck(run=_runner(site, app)).handle(request)
    assert verdict.importable
    assert request.env_file.read_text() == f"A=1\nONEX_API_IMAGE={_NEW}\nB=2\n"
