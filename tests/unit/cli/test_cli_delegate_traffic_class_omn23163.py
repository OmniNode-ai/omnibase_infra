# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --traffic-class/--source-surface`` stamp provenance (OMN-23163).

A probe or experiment marks its delegation as not a user's run with a traffic
class and a source surface. Both ride in the request's typed ``provenance``
field, whose ``source`` is the request's own ``source``. Synthetic traffic with
no source surface is refused by the core provenance model before dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import _write_payload, delegate_command

pytestmark = pytest.mark.unit


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


def test_traffic_class_flags_are_declared() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert {"--traffic-class", "--source-surface"} <= declared


def test_traffic_class_and_surface_reach_the_request_provenance(
    tmp_path: Path,
) -> None:
    payload = _payload(
        tmp_path, traffic_class="synthetic", source_surface="probe-ladder"
    )
    assert payload["provenance"] == {
        "source": "claude-code",
        "traffic_class": "synthetic",
        "source_surface": "probe-ladder",
    }


def test_traffic_class_alone_organic_needs_no_surface(tmp_path: Path) -> None:
    payload = _payload(tmp_path, traffic_class="organic")
    assert payload["provenance"] == {
        "source": "claude-code",
        "traffic_class": "organic",
    }


def test_no_traffic_flags_write_no_provenance(tmp_path: Path) -> None:
    assert "provenance" not in _payload(tmp_path)


def test_synthetic_without_surface_is_refused_in_the_payload(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="source_surface"):
        _payload(tmp_path, traffic_class="synthetic")


def test_synthetic_without_surface_is_refused_before_dispatch(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly: OK",
            "--traffic-class",
            "synthetic",
            "--state-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "source_surface" in result.output
    assert not (tmp_path / "tmp").exists()


def test_unknown_traffic_class_is_refused_by_the_flag() -> None:
    result = CliRunner().invoke(
        delegate_command, ["x", "--traffic-class", "bogus", "--source-surface", "s"]
    )
    assert result.exit_code == 2
    assert "bogus" in result.output
