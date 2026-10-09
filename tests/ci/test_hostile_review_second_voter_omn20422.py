# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20422: the Hostile Reviewer's second voter is a different lab model.

RULING 2026-10-09T04:15:56Z keeps Codex out of CI. The gpt-oss-review registry
key resolves its URL from ``LLM_GPT_OSS_REVIEW_URL``, so the workflow points it
at the second llama-server on the Studio (port 8131), which serves
Qwen3.6-35B-A3B, while qwen3-review stays Qwen3.8-27B on .201.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/hostile-reviewer.yml"
FIRST_VOTER_MODEL = "Qwen3.8-27B"


def _job_env() -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return cast("dict[str, str]", workflow["jobs"]["hostile-review"]["env"])


@pytest.mark.live_contact(
    "tests/ci/fixtures/hostile_reviewer_second_voter_omn20422.json"
)
def test_second_voter_endpoint_serves_a_different_model_than_the_first(
    recorded_response: dict[str, object],
) -> None:
    recording = cast("dict[str, dict[str, object]]", recorded_response["response"])
    served = [
        cast("str", entry["id"])
        for entry in cast("list[dict[str, object]]", recording["models"]["data"])
    ]
    assert served == ["qwen3.6-35b-a3b"]
    assert FIRST_VOTER_MODEL not in served
    assert recording["completion"]["finish_reason"] == "stop"


def test_review_and_preflight_share_one_second_voter_url_on_port_8131() -> None:
    url = _job_env()["LLM_GPT_OSS_REVIEW_URL"]
    assert urlparse(url).port == 8131


def test_the_quorum_keys_are_unchanged() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--model qwen3-review" in text
    assert "--model gpt-oss-review" in text
    assert 'REVIEW_MODEL_KEYS: "qwen3-review gpt-oss-review"' in text
