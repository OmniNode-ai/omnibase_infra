# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` names the lane and session that issued it (OMN-19860).

The lab ``delegation_events`` projection recorded model, provider and outcome
but never who asked, so per-lane delegation use could not be queried from the
event stream. The request now says who asked:

* the caller's ledger lane rides in the request ``metadata`` map under
  ``caller_lane`` (every released request consumer accepts that map);
* the caller's session rides in the request's declared ``session_id`` field,
  and only as a UUID, because the terminal projection types it as one.

Lane resolution, most explicit first: the ``--caller-lane`` flag; then the lane
environment variables the gh shim and the PR-ownership guard already read, in
their order; then the lane registered for the ``omni_worktrees/<ticket>/<dir>``
worktree the command runs from; otherwise none. A malformed flag is a usage
error; a malformed environment value is skipped and named, never guessed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.cli.cli_delegate import _write_payload, delegate_command
from omnibase_infra.cli.delegate_caller import resolve_delegate_caller
from omnibase_infra.cli.model_delegate_caller import ModelDelegateCaller

pytestmark = pytest.mark.unit

_LANE = "delegation-fix-delegation-events-no-caller-lane-83"
_SESSION = "15501d1a-5cfc-422b-a688-0ace6bfe0f21"
_ELSEWHERE = Path("/work/omni_home")


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


def _register(root: Path, worktree: Path, lane: str, session: str) -> None:
    """Write a record the way omniclaude's lane_identity.register does."""
    key = hashlib.sha256(str(worktree.resolve()).encode("utf-8")).hexdigest()[:32]
    record = root / "lane_identity" / f"{key}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "lane": lane,
                "session_id": session,
                "ticket": "OMN-19860",
                "worktree": str(worktree.resolve()),
                "registered_at": "2026-09-27T11:40:04Z",
            }
        ),
        encoding="utf-8",
    )


def test_the_command_declares_a_caller_lane_option() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert "--caller-lane" in declared


def test_lane_and_session_are_written_into_the_request(tmp_path: Path) -> None:
    payload = _payload(
        tmp_path,
        ticket_id="OMN-19860",
        caller=ModelDelegateCaller(
            lane=_LANE, lane_source="explicit", session_id=_SESSION, session_source="x"
        ),
    )
    assert payload["metadata"] == {"ticket_id": "OMN-19860", "caller_lane": _LANE}
    assert payload["session_id"] == _SESSION
    assert "caller_lane" not in payload


def test_an_unattributed_caller_writes_no_new_key(tmp_path: Path) -> None:
    payload = _payload(tmp_path, caller=ModelDelegateCaller.unattributed())
    assert "metadata" not in payload
    assert "session_id" not in payload


def test_the_metadata_stays_a_flat_string_map(tmp_path: Path) -> None:
    """The released request model types ``metadata`` as ``dict[str, str]``."""
    metadata = _payload(
        tmp_path,
        caller=ModelDelegateCaller(
            lane=_LANE, lane_source="explicit", session_id=None, session_source="none"
        ),
    )["metadata"]
    assert isinstance(metadata, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())


def test_the_explicit_flag_wins_over_the_environment() -> None:
    caller = resolve_delegate_caller(
        _LANE, cwd=_ELSEWHERE, environ={"ONEX_LANE_ID": "other-lane"}
    )
    assert (caller.lane, caller.lane_source) == (_LANE, "explicit")


@pytest.mark.parametrize(
    "value", ["two words", "lane|pipe", "", "  ", "-leading", "a" * 129]
)
def test_a_malformed_flag_is_refused_by_name(value: str) -> None:
    with pytest.raises(ValueError, match="--caller-lane"):
        resolve_delegate_caller(value, cwd=_ELSEWHERE, environ={})


@pytest.mark.parametrize(
    "name",
    [
        "ONEX_LANE",
        "ONEX_LANE_ID",
        "ONEX_AGENT_NAME",
        "CLAUDE_AGENT_NAME",
        "CLAUDE_SUBAGENT_NAME",
    ],
)
def test_each_lane_environment_variable_names_the_lane(name: str) -> None:
    caller = resolve_delegate_caller(None, cwd=_ELSEWHERE, environ={name: _LANE})
    assert (caller.lane, caller.lane_source) == (_LANE, f"env {name}")


def test_the_environment_order_matches_the_gh_shim() -> None:
    caller = resolve_delegate_caller(
        None,
        cwd=_ELSEWHERE,
        environ={"CLAUDE_SUBAGENT_NAME": "sub", "ONEX_LANE_ID": "declared"},
    )
    assert (caller.lane, caller.lane_source) == ("declared", "env ONEX_LANE_ID")


def test_a_malformed_environment_value_is_skipped_and_named() -> None:
    caller = resolve_delegate_caller(
        None,
        cwd=_ELSEWHERE,
        environ={"ONEX_LANE_ID": "not a lane", "CLAUDE_SUBAGENT_NAME": _LANE},
    )
    assert caller.lane == _LANE
    assert caller.lane_source == "env CLAUDE_SUBAGENT_NAME; skipped ONEX_LANE_ID"


def test_the_registered_worktree_lane_names_the_lane(tmp_path: Path) -> None:
    worktree = tmp_path / "omni_worktrees" / "OMN-19860" / "omnimarket"
    (worktree / "src").mkdir(parents=True)
    registry = tmp_path / "state"
    _register(registry, worktree, _LANE, "0123456789abcdef0123456789abcdef")
    caller = resolve_delegate_caller(
        None,
        cwd=worktree / "src",
        environ={"ONEX_LANE_REGISTRY_ROOT": str(registry)},
    )
    assert (caller.lane, caller.lane_source) == (_LANE, "registered worktree")
    # The registry's session is used when the harness names none, as a UUID.
    assert caller.session_id == "01234567-89ab-cdef-0123-456789abcdef"
    assert caller.session_source == "registered worktree"


def test_the_registry_root_falls_back_to_the_workspace_state_dir(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "omni_worktrees" / "OMN-19860" / "omnibase_infra"
    worktree.mkdir(parents=True)
    _register(tmp_path / ".onex_state", worktree, _LANE, _SESSION)
    caller = resolve_delegate_caller(
        None, cwd=worktree, environ={"OMNI_HOME": str(tmp_path)}
    )
    assert caller.lane == _LANE


def test_an_unregistered_worktree_or_other_directory_names_no_lane(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "omni_worktrees" / "OMN-1" / "omnimarket"
    worktree.mkdir(parents=True)
    for cwd in (worktree, tmp_path):
        caller = resolve_delegate_caller(
            None, cwd=cwd, environ={"ONEX_LANE_REGISTRY_ROOT": str(tmp_path)}
        )
        assert (caller.lane, caller.lane_source) == (None, "none")


def test_no_registry_root_names_no_lane_and_never_raises() -> None:
    caller = resolve_delegate_caller(
        None, cwd=Path("/work/omni_worktrees/OMN-1/omnimarket"), environ={}
    )
    assert caller == ModelDelegateCaller.unattributed()


def test_the_harness_session_is_the_session() -> None:
    caller = resolve_delegate_caller(
        None, cwd=_ELSEWHERE, environ={"CLAUDE_CODE_SESSION_ID": _SESSION.upper()}
    )
    assert caller.session_id == _SESSION
    assert caller.session_source == "env CLAUDE_CODE_SESSION_ID"


def test_a_session_that_is_not_a_uuid_is_not_sent() -> None:
    """The terminal projection types session_id as a UUID; free text is dropped."""
    caller = resolve_delegate_caller(
        None, cwd=_ELSEWHERE, environ={"CLAUDE_CODE_SESSION_ID": "session-label"}
    )
    assert caller.session_id is None
    assert caller.session_source == "none; skipped CLAUDE_CODE_SESSION_ID"


def test_describe_names_both_halves_and_how_they_were_chosen() -> None:
    caller = ModelDelegateCaller(
        lane=_LANE, lane_source="explicit", session_id=None, session_source="none"
    )
    assert caller.describe() == f"caller: lane={_LANE} (explicit) session=none (none)"


def test_the_flag_refuses_a_malformed_lane_before_anything_runs() -> None:
    from click.testing import CliRunner

    result = CliRunner().invoke(
        delegate_command, ["Reply with exactly: OK", "--caller-lane", "two words"]
    )
    assert result.exit_code == 2
    assert "--caller-lane" in result.output
