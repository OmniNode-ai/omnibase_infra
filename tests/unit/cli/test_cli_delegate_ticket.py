# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --ticket`` names the ticket a delegation works (OMN-19514).

A delegation run is joined to its ticket, and through the ticket to the DoD
verdict that judged it, only when the request says which ticket it served. The
ticket rides in the request's ``metadata`` map under ``ticket_id``, because
every released request consumer already accepts that map: a new declared
request field would be refused by the deployed consumer until a release
carried it.

Resolution, in order: the explicit flag; otherwise the ticket segment of an
``omni_worktrees/<TICKET>/`` working directory; otherwise nothing. A malformed
flag is a usage error raised before anything is published, and nothing is ever
guessed.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.cli.cli_delegate import (
    _write_payload,
    delegate_command,
    resolve_delegate_ticket,
)

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


def test_the_command_declares_a_ticket_option() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert "--ticket" in declared


def test_the_ticket_is_written_into_the_request_metadata(tmp_path: Path) -> None:
    payload = _payload(tmp_path, ticket_id="OMN-19514")
    assert payload["metadata"] == {"ticket_id": "OMN-19514"}
    assert "ticket_id" not in payload


def test_no_ticket_writes_no_metadata_key(tmp_path: Path) -> None:
    assert "metadata" not in _payload(tmp_path)


def test_the_metadata_is_a_flat_string_map(tmp_path: Path) -> None:
    """The released request model types ``metadata`` as ``dict[str, str]``."""
    metadata = _payload(tmp_path, ticket_id="OMN-19514")["metadata"]
    assert isinstance(metadata, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())


def test_an_explicit_ticket_wins_over_the_working_directory() -> None:
    cwd = Path("/work/omni_worktrees/OMN-1/omnimarket")
    assert resolve_delegate_ticket("OMN-19514", cwd=cwd) == ("OMN-19514", "explicit")


def test_surrounding_whitespace_is_normalised() -> None:
    assert resolve_delegate_ticket(" OMN-19514\n", cwd=Path("/work/elsewhere")) == (
        "OMN-19514",
        "explicit",
    )


@pytest.mark.parametrize("value", ["omn-19514", "OMN-0", "19514", "OMN 19514", "  "])
def test_a_malformed_ticket_is_refused_by_name(value: str) -> None:
    with pytest.raises(ValueError, match="--ticket"):
        resolve_delegate_ticket(value, cwd=Path("/work/elsewhere"))


def test_a_ticket_worktree_names_its_ticket() -> None:
    cwd = Path("/work/omni_worktrees/OMN-19514/omnimarket/src/omnimarket")
    assert resolve_delegate_ticket(None, cwd=cwd) == ("OMN-19514", "worktree path")


def test_a_directory_that_is_not_a_ticket_worktree_names_nothing() -> None:
    for cwd in (
        Path("/work/omni_home"),
        Path("/work/omni_worktrees/scratch/omnimarket"),
        Path("/work/omni_worktrees"),
    ):
        assert resolve_delegate_ticket(None, cwd=cwd) == (None, "none")


def test_the_flag_refuses_a_malformed_ticket_before_anything_runs() -> None:
    from click.testing import CliRunner

    result = CliRunner().invoke(
        delegate_command, ["Reply with exactly: OK", "--ticket", "omn-1"]
    )
    assert result.exit_code == 2
    assert "--ticket" in result.output
