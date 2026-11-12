# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22714 (T8 part 2): auto-wiring gives a folded profile's contract its handover.

``wire_from_manifest`` builds one ``HandlerContractSubscribePreflight`` per
contract. The handover it is given is the contract's own declaration when it has
one, else the fold-wide one of the folded profile that owns it
(``ONEX_RUNTIME_FOLD_HANDOVERS``), else none. The preflight is replaced by a
recorder; the manifest path, the profile list and the environment are real.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.handlers import handler_contract_subscribe_preflight as preflight
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_request import (
    ModelContractSubscribePreflightRequest,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_result import (
    ModelContractSubscribePreflightResult,
)
from tests.unit.runtime.auto_wiring.test_consumer_group_handover_omn21288 import (
    DECLARATION,
    ENVIRONMENT,
    _contract,
    _HandoverSemanticFake,
    _wire,
)

pytestmark = pytest.mark.unit

FOLD = {
    "effects": {
        "from_instance_id": "runtime-effects",
        "from_auto_offset_reset": "latest",
        "handover_rule": "last_ran",
    }
}


class _RecordingPreflight:
    """Stands in for the preflight; records the handover each contract is given."""

    given: list[ModelConsumerGroupHandover | None] = []

    @classmethod
    def from_contract(
        cls, contract: Any, *, environ: Mapping[str, str], **_: Any
    ) -> _RecordingPreflight:
        cls.given.append(preflight.load_contract_handover(contract, environ))
        return cls()

    async def handle(
        self, request: ModelContractSubscribePreflightRequest
    ) -> ModelContractSubscribePreflightResult:
        return ModelContractSubscribePreflightResult(
            topic=request.topic, max_in_flight_records=1, handover=None
        )


async def _given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profiles: tuple[str, ...],
    declaration: str,
    fold: dict[str, Any] | None,
    folded: str = "effects",
) -> ModelConsumerGroupHandover | None:
    _RecordingPreflight.given = []
    monkeypatch.setattr(
        preflight, "HandlerContractSubscribePreflight", _RecordingPreflight
    )
    monkeypatch.setenv("ONEX_RUNTIME_FOLDED_PROFILES", folded)
    if fold is None:
        monkeypatch.delenv(preflight.FOLD_HANDOVERS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(preflight.FOLD_HANDOVERS_ENV_VAR, json.dumps(fold))
    contract = _contract(tmp_path, declaration=declaration).model_copy(
        update={"runtime_profiles": profiles}
    )
    bus = _HandoverSemanticFake(environment=ENVIRONMENT)
    await bus.start()
    await _wire(contract, bus)
    await bus.stop_loops()
    (handover,) = _RecordingPreflight.given
    return handover


@pytest.mark.asyncio
async def test_a_contract_the_folded_profile_owns_is_wired_with_the_fold_handover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handover = await _given(
        tmp_path, monkeypatch, profiles=("effects",), declaration="", fold=FOLD
    )

    assert handover is not None
    assert handover.from_instance_id == "runtime-effects"
    assert handover.handover_rule == "last_ran"


@pytest.mark.asyncio
async def test_a_contracts_own_declaration_wins_over_the_fold_handover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handover = await _given(
        tmp_path,
        monkeypatch,
        profiles=("effects",),
        declaration=DECLARATION,
        fold=FOLD,
    )

    assert handover is not None
    assert handover.from_instance_id is None
    assert handover.from_node == "live-events-writer"


@pytest.mark.asyncio
async def test_a_contract_of_another_profile_is_wired_with_no_handover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handover = await _given(
        tmp_path, monkeypatch, profiles=("main",), declaration="", fold=FOLD
    )

    assert handover is None


@pytest.mark.asyncio
async def test_error_an_unreadable_fold_declaration_fails_the_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RecordingPreflight.given = []
    monkeypatch.setattr(
        preflight, "HandlerContractSubscribePreflight", _RecordingPreflight
    )
    monkeypatch.setenv("ONEX_RUNTIME_FOLDED_PROFILES", "effects")
    monkeypatch.setenv(preflight.FOLD_HANDOVERS_ENV_VAR, "{not json")
    contract = _contract(tmp_path, declaration="").model_copy(
        update={"runtime_profiles": ("effects",)}
    )
    bus = _HandoverSemanticFake(environment=ENVIRONMENT)
    await bus.start()

    with pytest.raises(ValueError, match="not valid JSON"):
        await _wire(contract, bus)

    assert _RecordingPreflight.given == []
    assert bus._group_members == {}
