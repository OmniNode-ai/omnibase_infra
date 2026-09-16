# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18414 — the gate-probe conjunct end to end, over the declared grammar.

The unit suite covers the pure decision. This one drives the conjunct the
closer actually runs — ``HandlerEvidenceAutocloseSweep._gate_probe_verdict`` —
so the contract, the resolver, the handler wiring and the GitHub read are
exercised together. Only the HTTP boundary is a stub; every layer above it is
the shipped code.

It also carries the cross-repo half of the contract's guarantee, from this
side: the transcription omniclaude's admission guard reads must be byte-equal
to the declaration here. omniclaude pins the same equality from its side. Two
assertions rather than one is deliberate — the failure this ticket closes was a
grammar that drifted while each repository's own suite stayed green.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.gate_binding import load_gate_binding_grammar
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)

pytestmark = pytest.mark.integration

_CANONICAL_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "contracts"
    / "gate_binding_grammar.json"
)
_GUARD_TRANSCRIPTION = Path("plugins/onex/hooks/config/gate_binding_grammar.json")


def _handler(
    *, workflow_runs: list[dict[str, object]]
) -> HandlerEvidenceAutocloseSweep:
    """The shipped handler with only the `gh` boundary replaced."""

    async def run_gh(args: list[str], timeout: float) -> tuple[Any, str]:
        assert args[0] == "gh", "the conjunct must read through the gh boundary"
        return {"workflow_runs": workflow_runs}, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=None,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=run_gh,
        run_dod_verify_command=None,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
class TestTheConjunctOverEveryDeclaredForm:
    """Every form the admission guard mandates reaches the next stage."""

    _GREEN = [{"id": 1, "conclusion": "success"}]

    @pytest.mark.parametrize(
        "description",
        [
            "Gate: C7",
            "Gate: INV-103",
            "Gate: OMN-16729 AC-5",
            "Gate: live-gate defect: workspace-reconcile-status",
            'Gate: <issue id="x" href="https://linear.app/omninode/issue/OMN-18031">'
            "OMN-18031</issue> AC-6",
        ],
    )
    async def test_a_traceability_binding_produces_no_hold(
        self, description: str
    ) -> None:
        """Each of these was `skipped_gate_probe_red` before OMN-18414.

        Live shapes: OMN-18387 and OMN-18406 carry the first, OMN-18403 the
        fourth, OMN-18365 the fifth. All four were held on the
        2026-09-15T21:34Z dry-run tick for no other reason.
        """
        handler = _handler(workflow_runs=[])
        ref, conclusion, hold = await handler._gate_probe_verdict(
            description=description, gh_timeout_seconds=5.0
        )
        assert hold == "", f"a traceability binding must not hold: {hold}"
        assert (ref, conclusion) == ("", "")

    async def test_a_green_workflow_binding_probes_and_releases(self) -> None:
        handler = _handler(workflow_runs=self._GREEN)
        ref, conclusion, hold = await handler._gate_probe_verdict(
            description="Gate: OmniNode-ai/omnibase_infra chain-canary.yml",
            gh_timeout_seconds=5.0,
        )
        assert ref == "OmniNode-ai/omnibase_infra chain-canary.yml"
        assert conclusion == "success"
        assert hold == ""

    async def test_a_red_workflow_binding_still_holds(self) -> None:
        """The OMN-16106 behaviour, unchanged.

        chain-canary was `failure` on every run of 2026-09-06, including run
        34061981317 fired two minutes AFTER the flip it failed to prevent.
        """
        handler = _handler(workflow_runs=[{"id": 34061981317, "conclusion": "failure"}])
        _ref, conclusion, hold = await handler._gate_probe_verdict(
            description="Gate: OmniNode-ai/omnibase_infra chain-canary.yml",
            gh_timeout_seconds=5.0,
        )
        assert conclusion == "failure"
        assert hold, "a red proof pointer must still hold the flip"
        assert "34061981317" in hold

    async def test_an_unreadable_binding_holds_and_names_the_accepted_forms(
        self,
    ) -> None:
        handler = _handler(workflow_runs=self._GREEN)
        _ref, conclusion, hold = await handler._gate_probe_verdict(
            description="Gate: chain-canary.yml", gh_timeout_seconds=5.0
        )
        assert conclusion == "", "an unreadable binding reports no conclusion"
        assert hold
        assert "C7" in hold and "live-gate defect" in hold

    async def test_a_description_with_no_binding_is_not_probed_at_all(self) -> None:
        """Positive control for the holds above, and a read that must not happen."""
        handler = _handler(workflow_runs=self._GREEN)
        assert await handler._gate_probe_verdict(
            description="a body that declares nothing", gh_timeout_seconds=5.0
        ) == ("", "", "")


class TestTheGuardTranscriptionMatchesTheDeclaration:
    """The cross-repo half, asserted from this side as well as omniclaude's."""

    @staticmethod
    def _guard_copy() -> Path | None:
        """Locate omniclaude's transcription, or report that we cannot.

        Resolved fail-fast from ``OMNI_HOME`` with no default (CLAUDE.md rule
        8). A CI runner for this repository has no omniclaude clone, so an
        unreachable transcription is a SKIP WITH A STATED REASON and never a
        silent pass — omniclaude's own suite is the enforcing copy.
        """
        omni_home = os.environ.get("OMNI_HOME", "")
        roots = [Path(omni_home) / "omniclaude"] if omni_home else []
        roots.append(Path(__file__).resolve().parents[4] / "omniclaude")
        for root in roots:
            candidate = root / _GUARD_TRANSCRIPTION
            if candidate.is_file():
                return candidate
        return None

    def test_the_two_copies_are_byte_identical(self) -> None:
        guard = self._guard_copy()
        if guard is None:
            pytest.skip(
                "no omniclaude clone carrying "
                f"{_GUARD_TRANSCRIPTION} is reachable from OMNI_HOME or the "
                "sibling layout, so the transcription cannot be compared here. "
                "omniclaude's own drift test is the enforcing copy."
            )
        canonical = _CANONICAL_CONTRACT.read_bytes()
        transcription = guard.read_bytes()
        assert (
            hashlib.sha256(transcription).hexdigest()
            == hashlib.sha256(canonical).hexdigest()
        ), (
            f"{guard} has drifted from {_CANONICAL_CONTRACT}. One grammar, two "
            "readers: a transcription that diverges reopens exactly the split "
            "OMN-18414 closed."
        )

    def test_the_declaration_parses_and_declares_one_proof_pointer(self) -> None:
        """Positive control: the file the test above compares is a real grammar."""
        raw = json.loads(_CANONICAL_CONTRACT.read_text(encoding="utf-8"))
        assert raw["forms"], "the contract declares no forms"
        grammar = load_gate_binding_grammar()
        probing = [f.id for f in grammar.forms if f.probe.value != "none"]
        assert probing == ["workflow_run"]
