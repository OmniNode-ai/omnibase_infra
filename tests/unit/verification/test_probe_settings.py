# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Probe timeouts come from the verification contract, not from code (OMN-21623)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.verification.model_verification_probe_settings import (
    ModelVerificationProbeSettings,
    load_probe_settings,
)
from omnibase_infra.verification.probes.probe_publication import (
    _rpk_watermark_fallback,
)

pytestmark = pytest.mark.unit

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "verification"
    / "contract.yaml"
)


def _declared_block() -> dict[str, object]:
    return yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["config"][
        "probe_settings"
    ]


def test_settings_equal_the_contract_block() -> None:
    settings = load_probe_settings()
    assert settings.rpk_command_timeout_seconds == 10.0
    assert settings.rpk_discovery_timeout_seconds == 5.0


def test_missing_timeout_fails_to_load(tmp_path: Path) -> None:
    block = _declared_block()
    del block["rpk_discovery_timeout_seconds"]
    contract = tmp_path / "contract.yaml"
    contract.write_text(yaml.safe_dump({"config": {"probe_settings": block}}))
    with pytest.raises(ValidationError):
        ModelVerificationProbeSettings.from_contract(contract)


def test_contract_without_block_fails_to_load(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(yaml.safe_dump({"name": "x"}))
    with pytest.raises(ProtocolConfigurationError):
        ModelVerificationProbeSettings.from_contract(contract)


def test_rpk_call_uses_the_declared_timeout() -> None:
    completed = MagicMock(returncode=0, stdout='{"partitions": []}', stderr="")
    with patch(
        "omnibase_infra.models.verification.model_verification_probe_settings.subprocess.run",
        return_value=completed,
    ) as run:
        _rpk_watermark_fallback("topic")
    assert (
        run.call_args.kwargs["timeout"]
        == load_probe_settings().rpk_command_timeout_seconds
    )


def test_discovery_call_uses_the_discovery_timeout() -> None:
    completed = MagicMock(returncode=0, stdout="[]", stderr="")
    with patch(
        "omnibase_infra.models.verification.model_verification_probe_settings.subprocess.run",
        return_value=completed,
    ) as run:
        load_probe_settings().run_rpk_discovery(["group", "list"])
    assert (
        run.call_args.kwargs["timeout"]
        == load_probe_settings().rpk_discovery_timeout_seconds
    )
    assert "env" not in run.call_args.kwargs


@pytest.mark.parametrize(
    "field", ["rpk_command_timeout_seconds", "rpk_discovery_timeout_seconds"]
)
@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_timeout_is_refused(field: str, value: int) -> None:
    block = _declared_block()
    block[field] = value
    with pytest.raises(ValidationError):
        ModelVerificationProbeSettings.model_validate(block)
