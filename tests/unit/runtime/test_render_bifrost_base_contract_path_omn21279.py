# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy owns the Bifrost base contract path (OMN-21279)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime import render_bifrost_delegation_contract as renderer
from tests.helpers.lane_overlay_fixtures import write_overlay
from tests.unit.runtime.test_render_bifrost_delegation_contract import (
    _lab_overlay,
    _write_base_contract,
)

pytestmark = pytest.mark.unit


def test_unset_base_path_fails_without_a_default() -> None:
    with pytest.raises(
        ProtocolConfigurationError, match="BIFROST_BASE_CONTRACT_PATH"
    ) as exc:
        renderer._resolve_canonical_source_path({})
    assert "no default" in str(exc.value)
    assert "deploy" in str(exc.value)


@pytest.mark.parametrize("blank", ["", " \t\n"])
def test_blank_base_path_fails(blank: str) -> None:
    with pytest.raises(ProtocolConfigurationError, match="BIFROST_BASE_CONTRACT_PATH"):
        renderer._resolve_canonical_source_path({"BIFROST_BASE_CONTRACT_PATH": blank})


@pytest.mark.parametrize("directory", [False, True])
def test_base_path_must_be_an_existing_file(tmp_path: Path, directory: bool) -> None:
    path = tmp_path if directory else tmp_path / "missing.yaml"
    with pytest.raises(ProtocolConfigurationError) as exc:
        renderer._resolve_canonical_source_path(
            {"BIFROST_BASE_CONTRACT_PATH": str(path)}
        )
    assert str(path) in str(exc.value)


def test_existing_base_file_is_returned(tmp_path: Path) -> None:
    path = tmp_path / "base.yaml"
    path.write_text("backends: []\n", encoding="utf-8")
    assert (
        renderer._resolve_canonical_source_path(
            {"BIFROST_BASE_CONTRACT_PATH": str(path)}
        )
        == path
    )


def test_disabled_render_does_not_resolve_the_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_resolution(env: object) -> Path:
        pytest.fail("disabled render must not resolve the base contract")

    monkeypatch.setattr(
        renderer, "_resolve_canonical_source_path", unexpected_resolution
    )
    outcome = renderer.render_bifrost_delegation_contract_with_source(
        environ={"BIFROST_CONTRACT_PATH": ""}
    )
    assert outcome.path is None
    assert outcome.overlay_source is None


def test_disabled_main_needs_no_base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIFROST_CONTRACT_PATH", "")
    monkeypatch.delenv("BIFROST_BASE_CONTRACT_PATH", raising=False)
    assert renderer.main() == 0


def test_renderer_does_not_load_an_omnimarket_resource() -> None:
    assert 'files("omnimarket")' not in Path(renderer.__file__).read_text(
        encoding="utf-8"
    )


def test_render_uses_the_deploy_provided_base_path(tmp_path: Path) -> None:
    source = tmp_path / "base.yaml"
    _write_base_contract(source)
    overlay = write_overlay(tmp_path / "overlay.yaml", _lab_overlay())
    target = tmp_path / "rendered.yaml"
    outcome = renderer.render_bifrost_delegation_contract_with_source(
        environ={
            "BIFROST_BASE_CONTRACT_PATH": str(source),
            "BIFROST_LANE_OVERLAY_PATH": str(overlay),
            "BIFROST_CONTRACT_PATH": str(target),
        }
    )
    assert outcome.path == target
    by_id = {
        backend["backend_id"]: backend
        for backend in yaml.safe_load(target.read_text(encoding="utf-8"))["backends"]
    }
    for binding in _lab_overlay()["backends"]:
        rendered = by_id[binding["backend_id"]]
        assert rendered["endpoint_url"] == binding["endpoint_url"]
        assert rendered["model_name"] == binding["served_model_id"]
