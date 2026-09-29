# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` offers what the registry declares, and nothing of its own (OMN-19407).

Operator ruling, 2026-09-24: "Everything should be fetched from the registry
for the CLI no fucking bespoke anything". These tests pin the READ PATH, not a
vocabulary. Every stand-in below declares names that exist in no production
contract, so a CLI that still carried a copied list would fail them, and each
test changes what the registry declares and checks that the CLI follows.

The stand-ins replace only ``contract_registry._entry_points``, the single
seam where installed metadata is read (see ``tests/helpers/cli_registry_stand_in``).
"""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import get_args

import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate, contract_registry
from omnibase_infra.cli.cli_delegate import DELEGATE_NODE_NAME, delegate_command
from tests.helpers.cli_registry_stand_in import (
    StandInExecutionBudget,
    StandInTaskClassAuthority,
    install_stand_in_registry,
)
from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
    ModelStandInDelegateRequest,
)

pytestmark = pytest.mark.unit

#: The real request-model read, captured before the directory conftest swaps it
#: for a shortcut, so these tests go through the node contract as the CLI does.
_REAL_REQUEST_MODEL_READ = cli_delegate._delegate_request_model

_BUDGET = StandInExecutionBudget(
    task_class_timeout_ceiling_seconds=30, terminal_delivery_margin_seconds=5
)


def _authority(**overrides: object) -> StandInTaskClassAuthority:
    fields: dict[str, object] = {
        "public": frozenset({"probe_public_alpha", "probe_public_beta"}),
        "internal": frozenset({"probe_internal_gamma"}),
        "unroutable": {"probe_unroutable_delta": "no tier offers probe-capability"},
        "fallback": "probe_public_alpha",
        "budgets": dict.fromkeys(
            ("probe_public_alpha", "probe_public_beta", "probe_internal_gamma"),
            _BUDGET,
        ),
    }
    fields.update(overrides)
    return StandInTaskClassAuthority.model_validate(fields)


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(delegate_command, args)


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def registry_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advertise the stand-in delegate node and read its model through its contract."""
    install_stand_in_registry(
        monkeypatch, _authority(), delegate_node_name=DELEGATE_NODE_NAME
    )
    monkeypatch.setattr(
        cli_delegate, "_delegate_request_model", _REAL_REQUEST_MODEL_READ
    )


class TestTaskClassesComeFromTheAuthority:
    """AC1: ``--task-type`` help and validation are the task-class contract's."""

    def test_help_lists_exactly_the_authoritys_classes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_stand_in_registry(monkeypatch, _authority())
        help_text = _flat(_invoke(["--help"]).output)
        assert (
            "Public, selectable from a prompt: probe_public_alpha, probe_public_beta."
            in help_text
        )
        assert "Internal, by explicit name only: probe_internal_gamma." in help_text
        assert (
            "refused with the contract's reason: probe_unroutable_delta." in help_text
        )
        assert "selection_fallback (probe_public_alpha)" in help_text

    def test_help_follows_the_registry_at_invocation_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing is frozen at import: a changed contract changes the next --help."""
        install_stand_in_registry(monkeypatch, _authority())
        first = _flat(_invoke(["--help"]).output)
        install_stand_in_registry(
            monkeypatch,
            _authority(
                public=frozenset({"probe_public_epsilon"}),
                fallback="probe_public_epsilon",
            ),
        )
        second = _flat(_invoke(["--help"]).output)
        assert "probe_public_alpha" in first
        assert "probe_public_alpha" not in second
        assert "probe_public_epsilon" in second

    def test_a_class_the_authority_does_not_declare_is_refused_by_the_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        install_stand_in_registry(monkeypatch, _authority())
        result = _invoke(["hello", "--task-type", "document"])
        assert result.exit_code == 2
        assert "'document' is not one of" in result.output
        for name in ("probe_public_alpha", "probe_internal_gamma"):
            assert name in result.output

    def test_an_internal_class_is_admitted_by_explicit_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-13966's behaviour, now the authority's decision rather than a CLI list."""
        authority = _authority()
        install_stand_in_registry(monkeypatch, authority)
        resolved = authority.resolve_task_type("x", explicit="probe_internal_gamma")
        assert resolved.task_type == "probe_internal_gamma"
        choice = next(p for p in delegate_command.params if p.name == "task_type").type
        assert "probe_internal_gamma" in choice.choices  # type: ignore[attr-defined]

    def test_an_unroutable_class_is_refused_in_the_contracts_words(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        install_stand_in_registry(monkeypatch, _authority())
        result = _invoke(
            [
                "hello",
                "--task-type",
                "probe_unroutable_delta",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "no tier offers probe-capability" in result.output


class TestRequestVocabulariesComeFromTheDelegateContract:
    """AC4: ``--source`` and ``--criteria-mode`` are the input model's Literals."""

    def test_source_choices_are_the_input_models(self, registry_node: None) -> None:
        declared = get_args(
            ModelStandInDelegateRequest.model_fields["source"].annotation
        )
        choice = next(p for p in delegate_command.params if p.name == "source").type
        assert tuple(choice.choices) == declared  # type: ignore[attr-defined]
        assert "stand-in-source" in _flat(_invoke(["--help"]).output)

    def test_criteria_mode_choices_are_the_input_models_spelled_as_flags(
        self, registry_node: None
    ) -> None:
        declared = get_args(
            ModelStandInDelegateRequest.model_fields["quality_contract_mode"].annotation
        )
        choice = next(
            p for p in delegate_command.params if p.name == "criteria_mode"
        ).type
        assert tuple(choice.choices) == tuple(  # type: ignore[attr-defined]
            value.replace("_", "-") for value in declared
        )

    def test_a_source_the_model_does_not_declare_is_refused(
        self, registry_node: None
    ) -> None:
        result = _invoke(["hello", "--source", "codex"])
        assert result.exit_code == 2
        assert "'codex' is not one of" in result.output
        assert "stand-in-source" in result.output


class TestAnUnresolvableRegistryIsARefusalByName:
    """AC2: no fallback list; the missing entry is named."""

    def test_unresolvable_task_class_authority_entry_refuses_help(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(contract_registry, "_entry_points", lambda _group: ())
        result = _invoke(["--help"])
        assert result.exit_code != 0
        assert "task_class_authority" in result.output
        assert "onex.contracts" in result.output

    def test_unresolvable_task_class_authority_loader_refuses_a_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        install_stand_in_registry(monkeypatch, None)
        result = _invoke(["hello", "--state-root", str(tmp_path)])
        assert result.exit_code != 0
        assert "task_class_authority" in result.output

    def test_unresolvable_delegate_node_refuses_the_source_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_stand_in_registry(monkeypatch, _authority())
        monkeypatch.setattr(
            cli_delegate, "_delegate_request_model", _REAL_REQUEST_MODEL_READ
        )
        real = contract_registry._entry_points

        def _without_delegate_node(group: str) -> tuple[EntryPoint, ...]:
            return tuple(e for e in real(group) if e.name != DELEGATE_NODE_NAME)

        monkeypatch.setattr(contract_registry, "_entry_points", _without_delegate_node)
        result = _invoke(["hello", "--source", "claude-code"])
        assert result.exit_code != 0
        assert DELEGATE_NODE_NAME in result.output
        assert "onex.nodes" in result.output

    def test_positive_control_the_same_run_resolves_with_the_entry_present(
        self, registry_node: None
    ) -> None:
        """Without this, a CLI that refused everything would pass the tests above."""
        result = _invoke(["--help"])
        assert result.exit_code == 0, result.output
