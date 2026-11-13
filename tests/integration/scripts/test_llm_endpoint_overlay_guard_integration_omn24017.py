# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy guard resolves the embedding address from a real overlay file (OMN-24017).

Runs ``scripts/check_llm_endpoint_env_contract.py`` as a subprocess against the
repository contract, an overlay root on disk and a settings-store file, so the
overlay lookup, the contract slot and the store reader are exercised together.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GUARD = _REPO_ROOT / "scripts" / "check_llm_endpoint_env_contract.py"
_ADDRESS = "http://192.0.2.10:8100"


def _run(tmp_path: Path, store_line: str, *, with_overlay: bool) -> int:
    root = tmp_path / "overlays"
    (root / "llm_endpoints").mkdir(parents=True)
    if with_overlay:
        (root / "llm_endpoints" / "overlay.yaml").write_text(
            f"schema_version: 1\nendpoint_urls:\n  embeddings-201: {_ADDRESS}\n"
        )
    store = tmp_path / "store.env"
    store.write_text(store_line + "\n")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"ONEX_SKILL_OVERLAY_ROOTS", "DEPLOY_AGENT_OVERLAY_HOME_CLONE"}
    }
    env["ONEX_SKILL_OVERLAY_ROOTS"] = str(root)
    return subprocess.run(
        [sys.executable, str(_GUARD), "--env-file", str(store)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    ).returncode


def test_store_naming_overlay_address_passes(tmp_path: Path) -> None:
    assert _run(tmp_path, f"LLM_EMBEDDING_URL={_ADDRESS}", with_overlay=True) == 0


def test_store_naming_other_address_is_refused(tmp_path: Path) -> None:
    other = "http://192.0.2.10:8002"
    assert _run(tmp_path, f"LLM_EMBEDDING_URL={other}", with_overlay=True) == 1


def test_missing_overlay_is_refused(tmp_path: Path) -> None:
    assert _run(tmp_path, f"LLM_EMBEDDING_URL={_ADDRESS}", with_overlay=False) == 1
