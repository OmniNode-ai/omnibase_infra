# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The CLI registry-vocabulary guard refuses copies and passes reads (OMN-19407).

Negative controls (must fail), positive controls (must pass) and the real CLI
package (must pass), so a guard that refused everything or nothing would fail
this module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "validation"
    / "check_cli_registry_vocabulary.py"
)


def _guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_cli_registry_vocabulary", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "cli_probe.py"
    path.write_text(source, encoding="utf-8")
    findings: list[str] = _guard().check_file(path)
    return findings


class TestNegativeControlsAreRefused:
    """The three shapes the operator ruling bans, as they looked before OMN-19407."""

    def test_a_choices_tuple(self, tmp_path: Path) -> None:
        findings = _scan(
            tmp_path,
            'TASK_TYPE_CHOICES = ("code_generation", "document", "research")\n',
        )
        assert len(findings) == 1
        assert "TASK_TYPE_CHOICES" in findings[0]

    def test_an_annotated_choices_tuple(self, tmp_path: Path) -> None:
        findings = _scan(
            tmp_path,
            'DELEGATE_SOURCE_CHOICES: tuple[str, ...] = ("claude-code", "codex")\n',
        )
        assert len(findings) == 1

    def test_a_literal_click_choice(self, tmp_path: Path) -> None:
        findings = _scan(
            tmp_path,
            "import click\n"
            'opt = click.option("--m", type=click.Choice(["extend", "replace"]))\n',
        )
        assert len(findings) == 1
        assert "click.Choice" in findings[0]

    def test_a_string_literal_type(self, tmp_path: Path) -> None:
        findings = _scan(
            tmp_path,
            "from typing import Literal\n"
            'Source = Literal["claude-code", "codex", "external-client"]\n',
        )
        assert len(findings) == 1
        assert "Literal" in findings[0]

    def test_an_exemption_without_a_reason_does_not_exempt(
        self, tmp_path: Path
    ) -> None:
        findings = _scan(
            tmp_path,
            'X_CHOICES = ("a", "b")  # cli-own-vocabulary:\n',
        )
        assert len(findings) == 1


class TestPositiveControlsPass:
    def test_a_registry_choice(self, tmp_path: Path) -> None:
        assert not _scan(
            tmp_path,
            "from omnibase_infra.cli.contract_registry import ContractChoice\n"
            'opt = ContractChoice("sources", lambda: ())\n',
        )

    def test_a_choice_derived_from_an_enum(self, tmp_path: Path) -> None:
        assert not _scan(
            tmp_path,
            "import click\nfrom enum import Enum\n"
            "class E(Enum):\n    A = 'a'\n"
            "opt = click.Choice([e.value for e in E])\n",
        )

    def test_a_cli_owned_vocabulary_with_a_reason(self, tmp_path: Path) -> None:
        assert not _scan(
            tmp_path,
            "import click\n"
            "# cli-own-vocabulary: this command's own output modes\n"
            'opt = click.Choice(["default", "receipt"])\n',
        )

    def test_a_single_value_literal(self, tmp_path: Path) -> None:
        assert not _scan(
            tmp_path,
            'from typing import Literal\nKind = Literal["only"]\n',
        )


def test_the_onex_cli_package_is_clean() -> None:
    """The real tree: every vocabulary is read from the registry or marked owned."""
    guard = _guard()
    root = _SCRIPT.parents[2] / "src" / "omnibase_infra" / "cli"
    assert guard.main([str(root)]) == 0
