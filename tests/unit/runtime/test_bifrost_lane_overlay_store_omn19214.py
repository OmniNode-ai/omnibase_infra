# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19214: a lane overlay is resolvable from the config store, with explicit precedence.

The mounted overlay file was the only delivery path for a lane overlay, so each
lab registration needed a PR and a runtime restart. The renderer now resolves
the SAME ``bifrost_lane_overlay.v3`` document from a store row too:

* a store row for the lane wins over the mounted file (explicit precedence);
* an absent row falls back to the file, and the render records which source it
  used, so the fallback is never silent (rule 8);
* a row that is present but invalid fails the render naming the store key and is
  never dropped in favour of the file;
* deleting the row restores the file-based render byte for byte.
"""

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
    lane_overlay_store_key,
    render_bifrost_delegation_contract,
    render_bifrost_delegation_contract_with_source,
)
from omnibase_infra.test_utils.fake_secret_store import FakeSecretStore
from tests.helpers.lane_overlay_fixtures import lab_backend, lab_overlay, write_overlay

_LANE = "dev"
_STORE_KEY = "delegation.bifrost.lane_overlay.dev"
_FILE_ENDPOINT = "http://file-host.example.invalid:8000/v1/chat/completions"
_STORE_ENDPOINT = "http://store-host.example.invalid:8000/v1/chat/completions"
_ADDED_ENDPOINT = "http://added-host.example.invalid:8000/v1/chat/completions"


def _base_contract(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "provider": "local",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "model_name": "served-model",
                        "tier": "local",
                        "timeout_ms": 300_000,
                    },
                ],
                "routing_rules": [
                    {"task_class": "code_generation", "backend_ids": ["local-coder"]}
                ],
                "default_backends": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _overlay(endpoint: str, *, served: str = "served-model") -> dict[str, object]:
    return lab_overlay(
        lane=_LANE,
        backends=[
            lab_backend("local-coder", endpoint_url=endpoint, served_model_id=served)
        ],
    )


def _store_with(overlay: dict[str, object]) -> FakeSecretStore:
    return FakeSecretStore({_STORE_KEY: yaml.safe_dump(overlay, sort_keys=False)})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coder_endpoint(path: Path) -> str:
    rendered = yaml.safe_load(path.read_text(encoding="utf-8"))
    coder = next(b for b in rendered["backends"] if b["backend_id"] == "local-coder")
    return str(coder["endpoint_url"])


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return _base_contract(tmp_path / "base.yaml")


@pytest.fixture
def file_overlay(tmp_path: Path) -> Path:
    return write_overlay(tmp_path / "dev.bifrost.yaml", _overlay(_FILE_ENDPOINT))


def test_store_key_is_per_lane() -> None:
    assert lane_overlay_store_key("dev") == _STORE_KEY
    with pytest.raises(ProtocolConfigurationError):
        lane_overlay_store_key("  ")


def test_store_row_wins_over_the_mounted_file_and_the_source_is_recorded(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    outcome = render_bifrost_delegation_contract_with_source(
        source_path=base,
        overlay_path=file_overlay,
        target_path=tmp_path / "rendered.yaml",
        environ={},
        verify_endpoints=False,
        overlay_store=_store_with(_overlay(_STORE_ENDPOINT)),
        lane=_LANE,
    )
    assert outcome.path is not None
    assert _coder_endpoint(outcome.path) == _STORE_ENDPOINT
    assert outcome.overlay_source is EnumBifrostLaneOverlaySource.STORE
    assert outcome.overlay_source_ref == _STORE_KEY


def test_store_row_registers_a_rung_with_no_file_at_all(
    base: Path, tmp_path: Path
) -> None:
    outcome = render_bifrost_delegation_contract_with_source(
        source_path=base,
        target_path=tmp_path / "rendered.yaml",
        environ={},
        verify_endpoints=False,
        overlay_store=_store_with(_overlay(_STORE_ENDPOINT)),
        lane=_LANE,
    )
    assert outcome.path is not None
    assert _coder_endpoint(outcome.path) == _STORE_ENDPOINT


def test_absent_row_falls_back_to_the_file_and_records_the_file(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    outcome = render_bifrost_delegation_contract_with_source(
        source_path=base,
        overlay_path=file_overlay,
        target_path=tmp_path / "rendered.yaml",
        environ={},
        verify_endpoints=False,
        overlay_store=FakeSecretStore(),
        lane=_LANE,
    )
    assert outcome.path is not None
    assert _coder_endpoint(outcome.path) == _FILE_ENDPOINT
    assert outcome.overlay_source is EnumBifrostLaneOverlaySource.FILE
    assert outcome.overlay_source_ref == str(file_overlay)


@pytest.mark.asyncio
async def test_removing_the_row_restores_the_file_render_by_sha256(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    committed = render_bifrost_delegation_contract(
        source_path=base,
        overlay_path=file_overlay,
        target_path=tmp_path / "committed.yaml",
        environ={},
        verify_endpoints=False,
    )
    assert committed is not None
    store = _store_with(_overlay(_STORE_ENDPOINT))
    with_row = render_bifrost_delegation_contract_with_source(
        source_path=base,
        overlay_path=file_overlay,
        target_path=tmp_path / "with-row.yaml",
        environ={},
        verify_endpoints=False,
        overlay_store=store,
        lane=_LANE,
    )
    assert with_row.path is not None
    assert _sha(with_row.path) != _sha(committed)

    assert await store.delete_secret(_STORE_KEY) is True
    after_delete = render_bifrost_delegation_contract_with_source(
        source_path=base,
        overlay_path=file_overlay,
        target_path=tmp_path / "after-delete.yaml",
        environ={},
        verify_endpoints=False,
        overlay_store=store,
        lane=_LANE,
    )
    assert after_delete.path is not None
    assert _sha(after_delete.path) == _sha(committed)


def test_invalid_row_missing_its_credential_declaration_fails_naming_the_key(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    # An ADDED backend must declare provider, tier AND credential; this row
    # declares the first two and drops the credential.
    overlay = lab_overlay(
        lane=_LANE,
        backends=[
            lab_backend("local-coder", endpoint_url=_STORE_ENDPOINT),
            {
                **lab_backend("local-second-host", endpoint_url=_ADDED_ENDPOINT),
                "provider": "local",
                "tier": "local",
            },
        ],
    )
    with pytest.raises(ProtocolConfigurationError, match=_STORE_KEY) as excinfo:
        render_bifrost_delegation_contract_with_source(
            source_path=base,
            overlay_path=file_overlay,
            target_path=tmp_path / "rendered.yaml",
            environ={},
            verify_endpoints=False,
            overlay_store=_store_with(overlay),
            lane=_LANE,
        )
    assert "credential" in str(excinfo.value)
    # Never dropped in favour of the file: nothing was rendered.
    assert not (tmp_path / "rendered.yaml").exists()


def test_malformed_yaml_row_fails_naming_the_key_and_does_not_fall_back(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    with pytest.raises(ProtocolConfigurationError, match=_STORE_KEY):
        render_bifrost_delegation_contract_with_source(
            source_path=base,
            overlay_path=file_overlay,
            target_path=tmp_path / "rendered.yaml",
            environ={},
            verify_endpoints=False,
            overlay_store=FakeSecretStore({_STORE_KEY: "backends: [unclosed"}),
            lane=_LANE,
        )
    assert not (tmp_path / "rendered.yaml").exists()


def test_row_for_another_lane_fails_naming_the_key(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    other = lab_overlay(
        lane="other",
        backends=[lab_backend("local-coder", endpoint_url=_STORE_ENDPOINT)],
    )
    with pytest.raises(ProtocolConfigurationError, match=_STORE_KEY) as excinfo:
        render_bifrost_delegation_contract_with_source(
            source_path=base,
            overlay_path=file_overlay,
            target_path=tmp_path / "rendered.yaml",
            environ={},
            verify_endpoints=False,
            overlay_store=_store_with(other),
            lane=_LANE,
        )
    assert "other" in str(excinfo.value)


def test_store_read_failure_fails_naming_the_key_and_does_not_fall_back(
    base: Path, file_overlay: Path, tmp_path: Path
) -> None:
    class _BrokenStore(FakeSecretStore):
        async def get_secret(self, key: str) -> str | None:
            raise ConnectionError("store unreachable")

    with pytest.raises(ProtocolConfigurationError, match=_STORE_KEY):
        render_bifrost_delegation_contract_with_source(
            source_path=base,
            overlay_path=file_overlay,
            target_path=tmp_path / "rendered.yaml",
            environ={},
            verify_endpoints=False,
            overlay_store=_BrokenStore(),
            lane=_LANE,
        )
    assert not (tmp_path / "rendered.yaml").exists()


def test_store_without_a_lane_is_refused(base: Path, tmp_path: Path) -> None:
    with pytest.raises(ProtocolConfigurationError, match="lane"):
        render_bifrost_delegation_contract_with_source(
            source_path=base,
            target_path=tmp_path / "rendered.yaml",
            environ={},
            verify_endpoints=False,
            overlay_store=FakeSecretStore(),
        )


def test_lane_defaults_to_the_environment_lane(base: Path, tmp_path: Path) -> None:
    outcome = render_bifrost_delegation_contract_with_source(
        source_path=base,
        target_path=tmp_path / "rendered.yaml",
        environ={"ONEX_ENVIRONMENT": _LANE},
        verify_endpoints=False,
        overlay_store=_store_with(_overlay(_STORE_ENDPOINT)),
    )
    assert outcome.overlay_source_ref == _STORE_KEY
