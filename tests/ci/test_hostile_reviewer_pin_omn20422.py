# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Hostile Reviewer runs the omniintelligence build that fails an unparseable vote (OMN-20422).

omniintelligence#1016 made an unparseable reviewer reply a failed vote instead
of a clean review and constrained the second voter's decoding to a findings
schema; omniintelligence#1014 renamed the second voter's registry key to
``local-studio-planner``. Nothing consumes either until the pin moves, and a
workflow that still passes the old key names a model the registry no longer has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "hostile-reviewer.yml"
PINS = ROOT / ".github" / "sibling-pins.yaml"
# The omniintelligence dev tip that carries both changes.
PARSE_FIX_SHA = "33a29cdfba6f14be6f94c3c77a05e29f345c692b"
RETIRED_KEY = "gpt-oss-review"
RETIRED_ENV = "LLM_GPT_OSS_REVIEW_URL"


def test_sibling_pin_is_the_parse_fix() -> None:
    pins = yaml.safe_load(PINS.read_text(encoding="utf-8"))["pins"]
    assert pins["omniintelligence"] == PARSE_FIX_SHA


def test_hostile_reviewer_clones_the_parse_fix() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert f"checkout --quiet {PARSE_FIX_SHA}" in text
    assert not re.search(r"51e1cb35", text)


def test_the_retired_key_and_env_var_are_gone() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert RETIRED_KEY not in text
    assert RETIRED_ENV not in text


def test_every_roster_site_names_the_renamed_key() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    flags = re.findall(r"^\s+--model local-studio-planner \\$", text, re.MULTILINE)
    assert len(flags) == 1
    assert 'REVIEW_MODEL_KEYS: "qwen3-review local-studio-planner"' in text
