# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23143 AC3: the shipped skill registry names no market result model.

Loads the registry the way the ``onex skill`` command does, from the packaged
``skill_mapping.yaml``, and checks every named result model resolves in core,
infra or builtins.
"""

from __future__ import annotations

import importlib

import pytest

from omnibase_infra.cli.cli_skill import load_skill_registry

pytestmark = pytest.mark.integration

_ALLOWED_ROOTS = frozenset({"omnibase_core", "omnibase_infra", "builtins"})


def test_shipped_skill_registry_names_no_market_result_model() -> None:
    skills = load_skill_registry().skills
    assert skills, "positive control: the registry loads skills"
    named = [s for s in skills if s.result_model is not None]
    assert named, "positive control: some skills still name a result model"
    for skill in named:
        assert skill.result_model is not None
        module_path, _, attr = skill.result_model.rpartition(".")
        assert module_path.split(".", 1)[0] in _ALLOWED_ROOTS, skill.skill_name
        assert hasattr(importlib.import_module(module_path), attr), skill.skill_name
