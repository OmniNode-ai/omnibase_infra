# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17648: omitted CLI booleans leave the node's defaults in control."""

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
