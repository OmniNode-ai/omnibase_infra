# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""RED->GREEN canonical def-B flip proof for HandlerCheckpointValidate (OMN-14820).

Companion to ``test_handler_checkpoint_validate.py``. Proves the OMN-14355
Class-B hand-flip (OMN-14781 path) of ``node_checkpoint_validate_compute`` to
canonical definition B, mirroring the merged siblings
``node_pr_state_projection_compute`` (OMN-14826) and
``node_ledger_projection_compute`` (OMN-14823).

* **SHAPE (RED->GREEN / RED-vs-EXISTS-but-WRONG).** The auto-wiring dispatch
  entrypoint ``handle`` is now the canonical definition-B shape — a single
  ``BaseModel``-typed positional ``request: ModelCheckpointValidateInput``
  resolvable by the runtime's own def-B input-model resolver
  (``_resolve_def_b_input_model_type``) and NOT an envelope-shaped signature.
  Pre-flip the class had no ``handle`` at all (the canonical-shape ratchet
  classified it ``op_method`` — only ``execute``/``validate`` operation
  methods), so ``test_handle_is_canonical_def_b_typed_entrypoint`` is RED there;
  it is ALSO RED for an EXISTS-but-WRONG ``handle(self, envelope: dict)`` shape,
  because the resolver returns ``None`` for a non-adaptable parameter.

* **EQUIVALENCE.** The pure ``validate`` logic is preserved byte-identical
  base_ref<->HEAD (the ``.handflip.json`` proof the ratchet re-derives from git).
  Over a deterministic corpus of valid checkpoints, the def-B ``handle`` output
  is asserted byte-equal to BOTH the pure ``validate`` result AND the legacy
  ``execute`` ProtocolHandler path — behavior preserved by the flip. The corpus
  is the exact input set bound (by ``input_hash``) into the adequacy receipt and
  the hand-flip proof under ``scripts/ci/adequacy_receipts/``.

  Note: ``validate``'s error branches (absolute path, invalid SHA, phase
  mismatch, attempt < 1) are pre-empted by ``ModelCheckpoint``'s own field/model
  validators, so a validly-constructed ``ModelCheckpointValidateInput`` — the
  only thing the canonical def-B adapter path can hand the handler — never
  reaches them. Those branches stay covered by the legacy envelope-dict suite
  in ``test_handler_checkpoint_validate.py`` and are waived (not re-driven) in
  the adequacy receipt.

Related: OMN-14355 (canonical-shape ratchet), OMN-14781 (hand-flip proof path),
OMN-14809 (verify_flip_bundle seam gate), OMN-2143 (original node).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

pytestmark = [pytest.mark.unit]

from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.models.checkpoint.model_checkpoint import ModelCheckpoint
from omnibase_infra.models.checkpoint.model_phase_payload_implement import (
    ModelPhasePayloadImplement,
)
from omnibase_infra.models.checkpoint.model_phase_payload_local_review import (
    ModelPhasePayloadLocalReview,
)
from omnibase_infra.nodes.node_checkpoint_validate_compute.handlers import (
    handler_checkpoint_validate as handler_module,
)
from omnibase_infra.nodes.node_checkpoint_validate_compute.handlers.handler_checkpoint_validate import (
    HandlerCheckpointValidate,
)
from omnibase_infra.nodes.node_checkpoint_validate_compute.models.model_checkpoint_validate_input import (
    ModelCheckpointValidateInput,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _handler_accepts_event_envelope,
    _resolve_def_b_input_model_type,
)

# Deterministic identity/time so the corpus (and therefore the flip's
# selected_input_hashes) is reproducible across runs and machines.
_RUN_ID = UUID(int=0x2143C0DE2143C0DE2143C0DE2143C0DE)
_CORR = UUID(int=0x14820DEF14820DEF14820DEF14820DEF)
_TS = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_FUTURE_TS = datetime(2099, 1, 1, 0, 0, 0, tzinfo=UTC)


def _checkpoint(**overrides: object) -> ModelCheckpoint:
    """Construct a VALID checkpoint (passes every ModelCheckpoint validator)."""
    fields: dict[str, object] = {
        "run_id": _RUN_ID,
        "ticket_id": "OMN-2143",
        "phase": EnumCheckpointPhase.IMPLEMENT,
        "timestamp_utc": _TS,
        "repo_commit_map": {"omnibase_infra": "abc1234"},
        "artifact_paths": ("src/foo.py",),
        "attempt_number": 1,
        "schema_version": "1.0.0",
        "phase_payload": ModelPhasePayloadImplement(
            branch_name="feature-branch",
            commit_sha="abc1234",
        ),
    }
    fields.update(overrides)
    return ModelCheckpoint(**fields)  # type: ignore[arg-type]


def _input(cp: ModelCheckpoint) -> ModelCheckpointValidateInput:
    return ModelCheckpointValidateInput(checkpoint=cp, correlation_id=_CORR)


def build_corpus() -> list[ModelCheckpointValidateInput]:
    """Deterministic valid-checkpoint corpus exercising the reachable validate() branches.

    Covers: clean IMPLEMENT (schema match, single repo/path, past timestamp),
    schema-version-mismatch warning, multi-repo/multi-path loop iteration, a
    distinct (LOCAL_REVIEW) phase with a matching payload, and the future-
    timestamp warning. Shared by the parity tests and the flip's adequacy
    receipt so the selected input set is identical (verify_flip_bundle binds the
    two field-by-field)."""
    return [
        _input(_checkpoint()),
        _input(_checkpoint(schema_version="2.0.0")),
        _input(
            _checkpoint(
                repo_commit_map={
                    "omnibase_infra": "abc1234",
                    "omnibase_core": "def5678",
                },
                artifact_paths=("src/a.py", "src/b.py"),
            )
        ),
        _input(
            _checkpoint(
                phase=EnumCheckpointPhase.LOCAL_REVIEW,
                phase_payload=ModelPhasePayloadLocalReview(
                    iteration_count=2,
                    last_clean_sha="abc1234",
                ),
            )
        ),
        _input(_checkpoint(timestamp_utc=_FUTURE_TS)),
    ]


_CORPUS = build_corpus()
_CORPUS_IDS = [
    "C1_implement_clean",
    "C2_schema_warning",
    "C3_multi_repo_paths",
    "C4_local_review_phase",
    "C5_future_timestamp",
]
# Per-case (is_valid, has_schema_warning, has_future_warning) expectations.
_EXPECTED = [
    (True, False, False),
    (True, True, False),
    (True, False, False),
    (True, False, False),
    (True, False, True),
]


@pytest.fixture
def handler() -> HandlerCheckpointValidate:
    return HandlerCheckpointValidate(MagicMock())


# =============================================================================
# SHAPE — the canonical definition-B flip (RED on the pre-flip tree)
# =============================================================================


class TestCanonicalDefBShape:
    """The dispatch entrypoint is canonical definition B (OMN-14355)."""

    def test_handle_is_canonical_def_b_typed_entrypoint(self) -> None:
        """RED->GREEN: the runtime's def-B resolver recovers the input model.

        Pre-flip there is no ``handle`` (``op_method``); an EXISTS-but-WRONG
        ``handle(self, envelope: dict)`` would make the resolver return ``None``.
        Post-flip ``handle(self, request: ModelCheckpointValidateInput)`` -> the
        resolver recovers the concrete input model.
        """
        resolved = _resolve_def_b_input_model_type(HandlerCheckpointValidate.handle)
        assert resolved is ModelCheckpointValidateInput

    def test_handle_is_not_envelope_shaped(self) -> None:
        """A definition-B core takes the domain model, never a transport envelope."""
        assert (
            _handler_accepts_event_envelope(HandlerCheckpointValidate.handle) is False
        )

    def test_handler_core_has_no_event_envelope_reference(self) -> None:
        """C-core: the resolved handler module must not reference the envelope type."""
        source = Path(handler_module.__file__).read_text(encoding="utf-8")
        assert "ModelEventEnvelope" not in source


# =============================================================================
# EQUIVALENCE — the def-B handle preserves the pure validate() / legacy execute()
# =============================================================================


class TestDispatchEquivalence:
    """The def-B ``handle`` output equals the pure validate and legacy execute."""

    @pytest.mark.parametrize("request_model", _CORPUS, ids=_CORPUS_IDS)
    def test_defb_handle_output_equals_pure_validate(
        self,
        handler: HandlerCheckpointValidate,
        request_model: ModelCheckpointValidateInput,
    ) -> None:
        output = asyncio.run(handler.handle(request_model))
        expected = handler.validate(
            request_model.checkpoint, request_model.correlation_id
        )
        assert output.result == expected

    @pytest.mark.parametrize("request_model", _CORPUS, ids=_CORPUS_IDS)
    def test_defb_handle_output_equals_legacy_execute(
        self,
        handler: HandlerCheckpointValidate,
        request_model: ModelCheckpointValidateInput,
    ) -> None:
        """The flip preserves behavior between the new and legacy entrypoints."""
        handle_out = asyncio.run(handler.handle(request_model))
        execute_out = asyncio.run(
            handler.execute(
                {
                    "checkpoint": request_model.checkpoint,
                    "correlation_id": request_model.correlation_id,
                }
            )
        )
        assert handle_out.result == execute_out.result

    @pytest.mark.parametrize(
        ("request_model", "expected"),
        list(zip(_CORPUS, _EXPECTED, strict=True)),
        ids=_CORPUS_IDS,
    )
    def test_defb_handle_result_semantics(
        self,
        handler: HandlerCheckpointValidate,
        request_model: ModelCheckpointValidateInput,
        expected: tuple[bool, bool, bool],
    ) -> None:
        """Validity + warning content matches the per-case expectation."""
        expected_valid, has_schema_warning, has_future_warning = expected
        output = asyncio.run(handler.handle(request_model)).result

        assert output.is_valid is expected_valid
        assert output.errors == ()
        assert output.correlation_id == _CORR
        assert (
            any("Schema version mismatch" in w for w in output.warnings)
            is has_schema_warning
        )
        assert any("future" in w for w in output.warnings) is has_future_warning


__all__ = ["TestCanonicalDefBShape", "TestDispatchEquivalence", "build_corpus"]
