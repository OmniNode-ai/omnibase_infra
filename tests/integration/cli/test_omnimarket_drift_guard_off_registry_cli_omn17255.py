# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for off-registry drift reporting (OMN-17255).

The unit module beside this one drives the resolver and the guard directly.
This one goes through the real ``click`` command, because the defect it covers
was only ever observable from a command line: on 2026-09-18, row L2 of the
local-path ground truth, ``env -u OMNI_HOME onex delegate "say ok"`` exited 0
with **zero** drift lines and **zero** skip lines on stderr. Nothing about the
guard function in isolation shows that, and nothing about it shows whether the
verdict survives the CLI's own stream handling to reach a customer's terminal.

Both halves of the ticket are asserted here against the real command:

* the structured verdict line reaches stderr, so the check is no longer silent;
* the command is NOT blocked by it. Per the 2026-09-18 operator ruling (goal
  row L2) the off-registry branch reports and never refuses, and the proof is
  that execution reaches a LATER, unrelated failure -- an unresolvable node --
  rather than stopping at the guard.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from omnibase_infra.cli import omnimarket_drift_guard as guard
from omnibase_infra.cli.cli_node import run_node_by_name

pytestmark = pytest.mark.integration

#: A workspace root with no ``omnimarket`` clone under it: exactly the customer
#: shape, and the shape a CI runner and a fresh machine also have.
_UNRESOLVABLE_NODE = "node_that_does_not_exist_omn17255"

#: The live 2026-09-18 pairing on this host's shared plugin CLI venv: the
#: installed omnimarket declares a floor the installed omnibase-infra is below.
_DRIFTED_ENV: dict[str, tuple[str, tuple[str, ...]]] = {
    "omnimarket": ("0.4.121", ("omnibase-infra<0.39.0,>=0.38.31",)),
    "omnibase-infra": ("0.38.30", ()),
}


def _run_guarded_command(state_root: Path, omni_home: Path) -> click.ClickException:
    """Invoke the real command and return the exception it ended on.

    Not a helper that swallows failures: the command is EXPECTED to fail, on an
    unresolvable node, and WHICH failure it is is the assertion.
    """
    with pytest.raises(click.ClickException) as exc_info:
        run_node_by_name.callback(
            _UNRESOLVABLE_NODE,
            contract_path=None,
            input_path=None,
            state_root=state_root,
            backend=(),
            timeout=1,
            verbose=False,
            output_mode="default",
            emit_socket=None,
            omni_home=omni_home,
            allow_omnimarket_drift=False,
        )
    return exc_info.value


def _verdict_lines(captured: str) -> list[str]:
    return [line for line in captured.splitlines() if line.startswith("drift_guard:")]


def test_cli_emits_the_verdict_line_when_no_canonical_clone_resolves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The L2 measurement, inverted into an assertion.

    ``tmp_path`` holds no ``omnimarket`` clone, so the guard takes the
    off-registry branch through the real command. Exactly one verdict line must
    reach stderr. The verdict itself depends on what is installed in the venv
    running the test, so it is asserted to be one of the three rather than
    pinned -- a specific value here would be a claim about the test host, not
    about the CLI.
    """
    state_root = tmp_path / "state"
    state_root.mkdir()

    error = _run_guarded_command(state_root, tmp_path)

    lines = _verdict_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert "mode=off-registry" in lines[0]
    assert any(
        f"verdict={verdict.value}" in lines[0]
        for verdict in guard.EnumOffRegistryVerdict
    ), lines[0]
    # Reached a later failure, so the guard let the command through.
    assert str(error).startswith("Unknown node")
    assert _UNRESOLVABLE_NODE in str(error)


def test_cli_is_not_blocked_by_an_off_registry_drift_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The goal-row invariant, through the real command.

    The installed environment is bound so the verdict is deterministically
    DRIFTED; everything else -- option parsing, the guard call site, the emit
    path, the stream -- is the real thing. The command must still get past the
    guard and fail on the node it could not resolve.
    """

    def _metadata(name: str) -> tuple[str, tuple[str, ...]] | None:
        return _DRIFTED_ENV.get(guard.canonical_distribution_name(name))

    monkeypatch.setattr(guard, "installed_distribution_metadata", _metadata)
    state_root = tmp_path / "state"
    state_root.mkdir()

    error = _run_guarded_command(state_root, tmp_path)

    lines = _verdict_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert "verdict=DRIFTED" in lines[0]
    assert "pin=omnibase-infra" in lines[0]
    assert "installed=0.38.30" in lines[0]
    # The whole point: a DRIFTED verdict off-registry does not stop the run.
    # Asserted on the failure's own opening, not on a substring search of it:
    # the unknown-node error enumerates every known node and several of those
    # names contain "drift", so a naive search matches the wrong thing.
    assert str(error).startswith("Unknown node")
    assert _UNRESOLVABLE_NODE in str(error)
    assert "OFF-REGISTRY DRIFT" not in str(error)
