# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18073: refuse an inherited environment carrying un-decoded ANSI-C quoting.

The deploy agent runs as a systemd user unit whose ``EnvironmentFile=`` points
at the operator env store. systemd's env-file parser does not implement bash
ANSI-C ``$'...'`` quoting: it keeps the literal ``$'``/``'`` wrapper and drops
every backslash escape, so a ``\\n`` becomes the bare letter ``n``.
``deploy-runtime.sh`` bash-``source``s the same file and decodes it correctly,
which is why only the agent path produced broken containers.

``_compose_env()`` hands its result straight to ``docker compose``, whose
``${VAR:-}`` interpolation writes it into every container the deploy creates.
Before this guard the mangled value was passed through silently and surfaced
only as a ``pyjwt`` ``InvalidKeyError`` in a different process, hours later.

The RED case is the whole point: without ``_assert_no_undecoded_ansi_c_quoting``
in ``_compose_env``, ``test_compose_env_refuses_undecoded_value`` passes the
mangled value through and the assertion on the raised error fails.
"""

from __future__ import annotations

import pytest
from deploy_agent.executor import (
    UndecodedAnsiCQuotingError,
    _assert_no_undecoded_ansi_c_quoting,
    _compose_env,
    _undecoded_ansi_c_quoted_names,
)

pytestmark = pytest.mark.unit

# A synthetic, non-secret value in exactly the shape systemd produces from an
# ANSI-C-quoted PEM line: wrapper retained, newlines collapsed to bare "n".
_MANGLED = "$'-----BEGIN SYNTHETIC-----nAAAAnBBBBn-----END SYNTHETIC-----n'"
# The same synthetic value as bash `source` yields it: a real multi-line value.
_DECODED = "-----BEGIN SYNTHETIC-----\nAAAA\nBBBB\n-----END SYNTHETIC-----\n"


def test_detects_undecoded_value_by_name_only() -> None:
    names = _undecoded_ansi_c_quoted_names(
        {"SYNTH_CREDENTIAL": _MANGLED, "PATH": "/usr/bin"}
    )
    assert names == ["SYNTH_CREDENTIAL"]


def test_decoded_multiline_value_is_not_flagged() -> None:
    """The shape both bash and systemd decode identically must pass."""
    assert _undecoded_ansi_c_quoted_names({"SYNTH_CREDENTIAL": _DECODED}) == []


def test_ordinary_values_are_not_flagged() -> None:
    env = {
        "A": "plain",
        "B": "",
        "C": "'single quoted but not ansi-c'",
        "D": "$HOME",
        "E": "$'",  # too short to be a wrapper pair
    }
    assert _undecoded_ansi_c_quoted_names(env) == []


def test_error_names_the_variable_and_never_the_value() -> None:
    with pytest.raises(UndecodedAnsiCQuotingError) as excinfo:
        _assert_no_undecoded_ansi_c_quoting({"SYNTH_CREDENTIAL": _MANGLED})
    message = str(excinfo.value)
    assert "SYNTH_CREDENTIAL" in message
    # The value must never appear, in whole or in a distinctive part.
    assert _MANGLED not in message
    assert "BEGIN SYNTHETIC" not in message
    # The message must name the repair, not just the symptom.
    assert "EnvironmentFile" in message


def test_clean_environment_passes() -> None:
    _assert_no_undecoded_ansi_c_quoting({"A": "plain", "B": _DECODED})


def test_compose_env_refuses_undecoded_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED without the guard: _compose_env would return the mangled value."""
    monkeypatch.setenv("SYNTH_CREDENTIAL", _MANGLED)
    with pytest.raises(UndecodedAnsiCQuotingError) as excinfo:
        _compose_env()
    assert "SYNTH_CREDENTIAL" in str(excinfo.value)


def test_compose_env_allows_decoded_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the same variable in decoded form deploys normally."""
    monkeypatch.setenv("SYNTH_CREDENTIAL", _DECODED)
    env = _compose_env()
    assert env["SYNTH_CREDENTIAL"] == _DECODED
