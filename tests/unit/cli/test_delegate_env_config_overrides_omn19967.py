# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` names every delegation config key an env var overrides (OMN-19967).

``BIFROST_CONTRACT_PATH``, ``BIFROST_OVERLAY_PATH`` and
``DELEGATION_ROUTING_TIERS_PATH`` from a sourced env file silently redirect the
routing a delegation takes. Before this change only the capture log recorded
``config_provenance ... source=contract_overlay_env``. AC1: with one of the
keys exported the CLI prints a line naming the key and its source and the
receipt carries the provenance; with none exported neither appears.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli.cli_delegate import _write_local_run_files
from omnibase_infra.cli.delegate_env_config_overrides import (
    DELEGATION_ENV_CONFIG_KEYS,
    env_config_overrides,
    format_override_line,
)
from omnibase_infra.cli.model_delegate_run_addressing import ModelDelegateRunAddressing
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.enums.enum_task_type_resolution import EnumTaskTypeResolution
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.unit

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)
_IN_PROCESS = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.IN_PROCESS, bus="inmemory"
)


def _receipt() -> ModelSkillResult[dict[str, object]]:
    return ModelSkillResult(
        skill_name="node_delegate_skill_orchestrator",
        node_name="node_delegate_skill_orchestrator",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=0,
        duration_ms=1200,
        result={
            "status": "completed",
            "task_type": "summarization",
            "model_name": "Qwen3.8-27B",
            "provider": "local",
            "response": "OK",
            "attempts": [
                {
                    "tier": "local",
                    "backend_id": "local-coder",
                    "model_id": "Qwen3.8-27B",
                    "quality_gate_passed": True,
                    "quality_score": 1.0,
                    "cost_usd": 0.0,
                    "failure_class": None,
                    "error_message": "",
                    "acceptance_decision": "accept",
                    "acceptance_reason": "quality_bar_met",
                    "substituted_from_backend_id": None,
                }
            ],
            "terminal_failure_cause": None,
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _written_receipt(tmp_path: Path, env: dict[str, str]) -> dict[str, object]:
    receipt = _receipt()
    _write_local_run_files(
        receipt=receipt,
        state_root=tmp_path,
        prompt="Reply with exactly: OK",
        task_type="summarization",
        task_type_resolution=EnumTaskTypeResolution.EXPLICIT.value,
        addressing=_IN_PROCESS,
        config_overrides=env_config_overrides(env),
    )
    run_dir = tmp_path / "runs" / str(receipt.run_id)
    return json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))


def test_the_three_keys_are_the_ones_the_ticket_names() -> None:
    assert set(DELEGATION_ENV_CONFIG_KEYS) == {
        "BIFROST_CONTRACT_PATH",
        "BIFROST_OVERLAY_PATH",
        "DELEGATION_ROUTING_TIERS_PATH",
    }


def test_no_key_exported_yields_no_overrides() -> None:
    assert env_config_overrides({"PATH": "/usr/bin"}) == ()


def test_a_blank_key_is_not_an_override() -> None:
    """Matches the resolver: a blank value falls through to the packaged default."""
    assert env_config_overrides({"BIFROST_OVERLAY_PATH": "   "}) == ()


def test_one_exported_key_yields_one_override_naming_key_and_source() -> None:
    overrides = env_config_overrides({"DELEGATION_ROUTING_TIERS_PATH": "/x/tiers.yaml"})
    assert len(overrides) == 1
    line = format_override_line(overrides[0])
    assert "DELEGATION_ROUTING_TIERS_PATH" in line
    assert "contract_overlay_env" in line
    assert "/x/tiers.yaml" in line


def test_overrides_are_ordered_by_the_declared_key_order() -> None:
    overrides = env_config_overrides(
        {
            "DELEGATION_ROUTING_TIERS_PATH": "/c",
            "BIFROST_OVERLAY_PATH": "/b",
            "BIFROST_CONTRACT_PATH": "/a",
        }
    )
    assert [o.config_key for o in overrides] == list(DELEGATION_ENV_CONFIG_KEYS)


def test_receipt_carries_provenance_when_a_key_is_exported(tmp_path: Path) -> None:
    written = _written_receipt(
        tmp_path, {"DELEGATION_ROUTING_TIERS_PATH": "/x/tiers.yaml"}
    )
    assert written["config_overrides"] == [
        {
            "config_key": "DELEGATION_ROUTING_TIERS_PATH",
            "source": "contract_overlay_env",
            "resolved_path": "/x/tiers.yaml",
        }
    ]


def test_receipt_has_no_provenance_field_when_none_exported(tmp_path: Path) -> None:
    """The positive control: without it the field above is unfalsifiable."""
    written = _written_receipt(tmp_path, {})
    assert "config_overrides" not in written
