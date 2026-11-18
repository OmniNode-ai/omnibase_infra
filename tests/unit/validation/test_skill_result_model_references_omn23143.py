# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23143 AC3: the skill mapping names no market result model.

``skill_mapping.yaml`` once carried 54 ``result_model`` references into the
market package, 49 of them into private node packages. The receipt stamps the
handler's own result model at run time, so the mapping needs none of them. What
the mapping still names must resolve in core or infra.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.cli.cli_skill import load_skill_registry
from omnibase_infra.cli.model_skill_mapping import ModelSkillMapping

pytestmark = pytest.mark.unit

_MAPPING = Path(__file__).resolve().parents[3] / (
    "src/omnibase_infra/cli/skill_mapping.yaml"
)
_ALLOWED_ROOTS = frozenset({"omnibase_core", "omnibase_infra", "builtins"})


def test_skill_result_model_references_name_no_market_package() -> None:
    market_lines = [
        line
        for line in _MAPPING.read_text(encoding="utf-8").splitlines()
        if re.match(r"\s*result_model:\s*omnimarket\.", line)
    ]
    assert market_lines == []


def test_skill_result_model_references_resolve_to_core_or_infra() -> None:
    named = [s for s in load_skill_registry().skills if s.result_model is not None]
    assert named, "positive control: the registry still names core or infra models"
    for skill in named:
        assert skill.result_model is not None
        module_path, _, attr = skill.result_model.rpartition(".")
        assert module_path.split(".", 1)[0] in _ALLOWED_ROOTS, skill.skill_name
        module = importlib.import_module(module_path)
        assert hasattr(module, attr), f"{skill.skill_name}: {skill.result_model}"


def test_skill_result_model_references_refused_for_market_package() -> None:
    with pytest.raises(ValidationError, match="match pattern"):
        ModelSkillMapping(
            skill_name="probe",
            node_name="node_probe",
            result_model="omnimarket.nodes.node_probe.models.ModelProbeResult",
        )
    # Control: a core model is accepted by the same validator.
    ok = ModelSkillMapping(
        skill_name="probe",
        node_name="node_probe",
        result_model="omnibase_core.models.claude_code_binding.ModelBindingInstallResult",
    )
    assert ok.result_model is not None
