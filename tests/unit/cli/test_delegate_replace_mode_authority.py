# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A replace-mode bar that no answer can meet is refused at the flag (OMN-19557).

MEASURED 2026-09-25. A Codex lane drafted a PR body with

    onex delegate ... --task-type document --criteria concise
        --criteria task_completed --criteria plain_text_only
        --criteria-mode replace-task-class

and run ``7d83a5df-8824-4817-a70d-e5997b1a7af3`` climbed six rungs (three
local, two free frontier, one metered) in 116 s. Every rung scored 1.0 against
a 0.8 bar and every rung was refused with the same reason:

    TASK_MISMATCH: no deterministic acceptance or judge adequacy authority;
    schema/length/no-refusal/marker checks are reject-only

That is the quality gate working as designed (OMN-13370: a structural or
marker check may reject an answer but never promote one). ``replace-task-class``
drops the task class's own adequacy authority (``semantic_adequacy`` for
``document``), and ``concise``, ``task_completed`` and ``plain_text_only`` are
all reject-only, so the bar that remained had nothing that could accept. The
verdict depends only on the rule set, never on the answer, so it was decided
before the first rung ran. The same command from a Claude lane reproduced it
(run ``82129fb6-5e2e-46e6-bf8f-d3011a0c00df``, six rungs, one metered call),
and the same criteria in the default extend mode completed on the first free
local rung (run ``1f82da14-fc9c-47a7-9595-c66ab27076d5``).

The lane reported the refusal as a Codex-specific "acceptance-authority gate".
It was not: nothing in the path keys on the caller. It was a request the CLI
accepted although its outcome was already fixed, and whose refusal named a
reducer rule instead of the flag the caller had to change.

The predicate is resolved from omnimarket's own quality gate, the same module
that refuses the run, rather than from a copy of its reject-only sets here, for
the reason ``load_supported_criteria`` gives: one source, not a table that
drifts. With no omnimarket the criteria pass through, exactly as the
vocabulary check does.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

pytestmark = pytest.mark.unit

#: The reject-only criteria in the measured run, and one criterion that the
#: gate's deterministic band does treat as acceptance authority. Only the
#: stub below consults these; the real predicate is omnimarket's.
_REJECT_ONLY = frozenset({"concise", "task_completed", "plain_text_only"})
_AUTHORITY = frozenset({"compiles_without_errors"})


def _stub_predicate(criteria: tuple[str, ...]) -> bool:
    return any(item in _AUTHORITY for item in criteria)


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    predicate: Callable[[tuple[str, ...]], bool] | None,
) -> None:
    from omnibase_infra.cli import cli_delegate

    monkeypatch.setattr(
        cli_delegate, "load_criteria_adequacy_authority", lambda: predicate
    )
    monkeypatch.setattr(
        cli_delegate,
        "load_supported_criteria",
        lambda: _REJECT_ONLY | _AUTHORITY,
    )


class TestAReplaceModeBarWithNoAuthorityIsRefused:
    def test_the_measured_request_is_refused_before_a_run_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.cli import cli_delegate

        _patch(monkeypatch, _stub_predicate)
        with pytest.raises(ValueError) as excinfo:
            cli_delegate._validate_criteria_mode(
                ("concise", "task_completed", "plain_text_only"),
                "replace-task-class",
            )
        message = str(excinfo.value)
        assert "--criteria-mode replace-task-class" in message
        assert "no adequacy authority" in message
        assert "concise, task_completed, plain_text_only" in message
        # The caller's next move: which flag to drop, and what else would work.
        assert "Drop --criteria-mode" in message
        assert "compiles_without_errors" in message, (
            "the refusal must name the criteria that CAN accept an answer, "
            "derived from the same predicate"
        )

    def test_replace_mode_with_no_criteria_at_all_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty replaced bar falls to the legacy checks, which never accept."""
        from omnibase_infra.cli import cli_delegate

        _patch(monkeypatch, _stub_predicate)
        with pytest.raises(ValueError, match="no adequacy authority"):
            cli_delegate._validate_criteria_mode((), "replace-task-class")


class TestPositiveControls:
    def test_replace_mode_with_an_authority_criterion_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.cli import cli_delegate

        _patch(monkeypatch, _stub_predicate)
        assert (
            cli_delegate._validate_criteria_mode(
                ("concise", "compiles_without_errors"), "replace-task-class"
            )
            == "replace-task-class"
        )

    @pytest.mark.parametrize("mode", [None, "extend-task-class"])
    def test_extend_mode_keeps_the_class_authority_and_is_never_checked(
        self, monkeypatch: pytest.MonkeyPatch, mode: str | None
    ) -> None:
        """Extend mode keeps the task class's own authority, so it is not ours to judge."""
        from omnibase_infra.cli import cli_delegate

        def _must_not_be_called() -> None:
            raise AssertionError("extend mode must not consult the predicate")

        monkeypatch.setattr(
            cli_delegate, "load_criteria_adequacy_authority", _must_not_be_called
        )
        assert cli_delegate._validate_criteria_mode(("concise",), mode) == mode

    def test_without_omnimarket_replace_mode_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No gate to ask means no verdict; dispatch fails for its own reason."""
        from omnibase_infra.cli import cli_delegate

        _patch(monkeypatch, None)
        assert (
            cli_delegate._validate_criteria_mode(("concise",), "replace-task-class")
            == "replace-task-class"
        )

    def test_the_real_resolver_returns_none_without_omnimarket(self) -> None:
        """This suite runs without omnimarket (OMN-15620 venv purity).

        With omnimarket present the real predicate is exercised instead, so
        the assertion is evidence in either environment.
        """
        import importlib.util

        from omnibase_infra.cli import cli_delegate

        predicate = cli_delegate.load_criteria_adequacy_authority()
        if importlib.util.find_spec("omnimarket") is None:
            assert predicate is None
        else:
            assert predicate is not None
            assert predicate(("concise", "task_completed", "plain_text_only")) is False
            assert predicate(("final_artifact_only",)) is True
