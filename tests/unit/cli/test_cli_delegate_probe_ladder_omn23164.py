# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --probe-ladder`` sends a committed ladder file (OMN-23164).

A synthetic probe names the rungs one run walks in a committed file. The file's
mapping rides in the request's ``probe_ladder`` field, and only together with
``--traffic-class synthetic``: anything else is refused before dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import _write_payload, delegate_command
from omnibase_infra.models.delegation.delegate_request_payload import (
    load_probe_ladder,
)

pytestmark = pytest.mark.unit

_LADDER_YAML = """\
rungs:
  - backend_id: stub-rung-1
    model_id: stub-model-1
    endpoint_ref: http://127.0.0.1:18081/v1/chat/completions
  - backend_id: stub-rung-2
    model_id: stub-model-2
    endpoint_ref: http://127.0.0.1:18082/v1/chat/completions
"""


def _ladder_file(
    tmp_path: Path, text: str = _LADDER_YAML, name: str = "ladder.yaml"
) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _payload(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    path = _write_payload(
        prompt="Reply with exactly: OK",
        task_type="summarization",
        source="claude-code",
        state_root=tmp_path,
        run_id=uuid4(),
        correlation_id=uuid4(),
        max_tokens=None,
        **kwargs,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_probe_ladder_flag_is_declared() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert "--probe-ladder" in declared


def test_probe_ladder_file_reaches_the_request(tmp_path: Path) -> None:
    ladder = load_probe_ladder(_ladder_file(tmp_path), traffic_class="synthetic")
    payload = _payload(
        tmp_path,
        traffic_class="synthetic",
        source_surface="doctor_escalation_check",
        probe_ladder=ladder,
    )
    rungs = payload["probe_ladder"]
    assert isinstance(rungs, dict)
    assert [rung["backend_id"] for rung in rungs["rungs"]] == [
        "stub-rung-1",
        "stub-rung-2",
    ]


def test_probe_ladder_json_file_is_read_like_yaml(tmp_path: Path) -> None:
    path = _ladder_file(tmp_path, '{"rungs": []}', "ladder.json")
    assert load_probe_ladder(path, traffic_class="synthetic") == {"rungs": []}


def test_probe_ladder_absent_writes_no_key(tmp_path: Path) -> None:
    assert load_probe_ladder(None, traffic_class="synthetic") is None
    assert "probe_ladder" not in _payload(tmp_path)


def test_probe_ladder_must_be_a_mapping(tmp_path: Path) -> None:
    path = _ladder_file(tmp_path, "- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_probe_ladder(path, traffic_class="synthetic")


def test_probe_ladder_needs_synthetic_in_the_loader(tmp_path: Path) -> None:
    path = _ladder_file(tmp_path)
    for traffic_class in (None, "organic"):
        with pytest.raises(ValueError, match="synthetic"):
            load_probe_ladder(path, traffic_class=traffic_class)


@pytest.mark.parametrize("flags", [[], ["--traffic-class", "organic"]])
def test_probe_ladder_needs_synthetic_before_dispatch(
    tmp_path: Path, flags: list[str]
) -> None:
    ladder = _ladder_file(tmp_path)
    state_root = tmp_path / "state"
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly: OK",
            "--probe-ladder",
            str(ladder),
            *flags,
            "--state-root",
            str(state_root),
        ],
    )
    assert result.exit_code == 2
    assert "--traffic-class synthetic" in result.output
    assert not (state_root / "tmp").exists()
