# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise store resolution through the environment adapter and disk renderer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.enums.enum_bifrost_lane_overlay_source import (
    EnumBifrostLaneOverlaySource,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
    render_bifrost_delegation_contract_with_source,
)
from omnibase_infra.secret_stores.adapter_env_secret_store import AdapterEnvSecretStore
from tests.helpers.lane_overlay_fixtures import lab_backend, lab_overlay, write_overlay

pytestmark = pytest.mark.integration

_STORE_KEY = "delegation.bifrost.lane_overlay.dev"
_FILE_ENDPOINT = "http://file.example.invalid:8000/v1/chat/completions"
_STORE_ENDPOINT = "http://store.example.invalid:8000/v1/chat/completions"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "base.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "provider": "local",
                        "model_name": "base-model",
                        "tier": "local",
                    }
                ],
                "routing_rules": [
                    {"task_class": "code_generation", "backend_ids": ["local-coder"]}
                ],
                "default_backends": ["local-coder"],
            }
        ),
        encoding="utf-8",
    )
    overlay = write_overlay(
        tmp_path / "lane.bifrost.yaml",
        lab_overlay(
            lane="dev",
            backends=[lab_backend("local-coder", endpoint_url=_FILE_ENDPOINT)],
        ),
    )
    return source, overlay


async def test_env_store_render_and_row_removal_restore_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read the real adapter from an active loop, then read back each render."""
    source, overlay = _write_inputs(tmp_path)
    baseline = tmp_path / "file-only.yaml"
    render_bifrost_delegation_contract(
        source_path=source, overlay_path=overlay, target_path=baseline, environ={}
    )
    file_bytes = baseline.read_bytes()
    store = AdapterEnvSecretStore()
    monkeypatch.setenv(
        _STORE_KEY,
        yaml.safe_dump(
            lab_overlay(
                lane="dev",
                backends=[
                    lab_backend(
                        "local-coder",
                        endpoint_url=_STORE_ENDPOINT,
                        served_model_id="store-model",
                    )
                ],
            )
        ),
    )
    target = tmp_path / "rendered.yaml"
    with_row = render_bifrost_delegation_contract_with_source(
        source_path=source,
        overlay_path=overlay,
        target_path=target,
        overlay_store=store,
        environ={"ONEX_ENVIRONMENT": "dev"},
    )
    assert with_row.path == target
    assert with_row.overlay_source is EnumBifrostLaneOverlaySource.STORE
    assert with_row.overlay_source_ref == _STORE_KEY
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert rendered["backends"][0]["endpoint_url"] == _STORE_ENDPOINT
    assert rendered["backends"][0]["model_name"] == "store-model"
    assert target.read_bytes() != file_bytes

    monkeypatch.delenv(_STORE_KEY)
    after_delete = render_bifrost_delegation_contract_with_source(
        source_path=source,
        overlay_path=overlay,
        target_path=target,
        overlay_store=store,
        environ={"ONEX_ENVIRONMENT": "dev"},
    )
    assert after_delete.path == target
    assert after_delete.overlay_source is EnumBifrostLaneOverlaySource.FILE
    assert after_delete.overlay_source_ref == str(overlay)
    assert (
        hashlib.sha256(target.read_bytes()).digest()
        == hashlib.sha256(file_bytes).digest()
    )


@pytest.mark.parametrize("row", ["", "backends: [unclosed", "[]", "lane: dev"])
def test_invalid_env_store_row_preserves_existing_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    source, overlay = _write_inputs(tmp_path)
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=source, overlay_path=overlay, target_path=target, environ={}
    )
    previous = target.read_bytes()
    monkeypatch.setenv(_STORE_KEY, row)
    with pytest.raises(ProtocolConfigurationError, match=_STORE_KEY):
        render_bifrost_delegation_contract_with_source(
            source_path=source,
            overlay_path=overlay,
            target_path=target,
            overlay_store=AdapterEnvSecretStore(),
            lane="dev",
            environ={},
        )
    assert target.read_bytes() == previous
    assert not target.with_suffix(".yaml.tmp").exists()
