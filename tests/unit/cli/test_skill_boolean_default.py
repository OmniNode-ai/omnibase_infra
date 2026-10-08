# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17648: omitted CLI booleans leave the node's defaults in control."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_skill
from omnibase_infra.cli.enum_skill_arg_type import EnumSkillArgType
from omnibase_infra.cli.model_skill_arg_spec import ModelSkillArgSpec
from omnibase_infra.cli.model_skill_mapping import ModelSkillMapping

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("default", [None, False, True])
@pytest.mark.parametrize("supplied", [False, True])
def test_boolean_omission_preserves_declared_default(
    default: bool | None, supplied: bool
) -> None:
    mapping = ModelSkillMapping(
        skill_name="example",
        node_name="node_example",
        result_model="example.Result",
        args=(
            ModelSkillArgSpec(
                name="flag-only",
                payload_field="flag_only",
                arg_type=EnumSkillArgType.BOOLEAN,
                default=default,
            ),
        ),
    )
    payload = cli_skill._parse_skill_args(mapping, ("--flag-only",) if supplied else ())
    expected = (
        {"flag_only": True}
        if supplied
        else ({} if default is None else {"flag_only": default})
    )
    assert payload == expected


def test_omitted_boolean_preserves_static_payload() -> None:
    mapping = ModelSkillMapping(
        skill_name="example",
        node_name="node_example",
        result_model="example.Result",
        static_payload={"flag_only": True},
        args=(
            ModelSkillArgSpec(
                name="flag-only",
                payload_field="flag_only",
                arg_type=EnumSkillArgType.BOOLEAN,
            ),
        ),
    )
    assert cli_skill._parse_skill_args(mapping, ()) == {"flag_only": True}


@pytest.mark.parametrize("flags", [[], ["--flag-only", "--dry-run"]])
def test_linear_housekeeping_dispatch_preserves_node_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str]
) -> None:
    """Inspect the actual CLI dispatch payload without executing Linear writes."""
    captured: list[dict[str, object]] = []

    def capture_dispatch(**kwargs: object) -> int:
        assert kwargs["node_name"] == "node_linear_triage"
        input_path = kwargs["input_path"]
        assert isinstance(input_path, Path)
        captured.append(json.loads(input_path.read_text(encoding="utf-8")))
        return 0

    monkeypatch.setattr(cli_skill, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_skill, "_resolve_packaged_contract", lambda _: tmp_path / "contract.yaml"
    )
    monkeypatch.setattr(cli_skill, "run_receipt_mode", capture_dispatch)
    cli_skill.load_skill_registry.cache_clear()
    try:
        result = CliRunner().invoke(
            cli_skill.run_skill_by_name,
            ["linear_housekeeping", "--state-root", str(tmp_path), *flags],
        )
        assert result.exit_code == 0, result.output
        assert captured == ([{"flag_only": True, "dry_run": True}] if flags else [{}])
    finally:
        cli_skill.load_skill_registry.cache_clear()
