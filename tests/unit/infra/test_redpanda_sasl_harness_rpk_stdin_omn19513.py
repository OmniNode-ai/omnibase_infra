# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513: the SASL test harness never puts a broker password on a command line.

``RedpandaSasl`` drives ``rpk`` inside a broker container with ``docker exec``. It can adopt
the declared broker of a CI lane, so its credential is a real one. A ``-X pass=<value>``
element in that argv sits on the host process table for the life of the call, and a client
that is killed leaves its in-container ``rpk`` behind with the value still on show
(OMN-17427). The harness hands the password over stdin instead: the one-shot shell reads it
into ``RPK_PASS`` and execs ``rpk``, with the user and mechanism as plain ``-e`` values.

These tests run no Docker: ``subprocess.run`` is replaced, and the recorded argv and stdin
are what the host would show and what the container would read.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from tests.integration.customer_path import redpanda_sasl_harness as harness

pytestmark = pytest.mark.unit

_PASSWORD = "unit-test-secret-value"
_OTHER_PASSWORD = "other-principal-secret"


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(
        self, argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), kwargs.get("input")))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    return rec


@pytest.fixture
def broker() -> harness.RedpandaSasl:
    return harness.RedpandaSasl(
        container="c-under-test",
        port=19092,
        username="harness-user",
        password=_PASSWORD,
        mechanism="SCRAM-SHA-256",
    )


def _assert_password_only_on_stdin(
    call: tuple[list[str], str | None], password: str, user: str
) -> None:
    argv, stdin = call
    joined = " ".join(argv)
    assert password not in joined
    assert "pass=" not in joined
    assert "--password" not in joined
    assert f"RPK_USER={user}" in argv
    assert any(item.startswith("RPK_SASL_MECHANISM=") for item in argv)
    assert "-i" in argv, "stdin must be open for the password line"
    assert stdin is not None and stdin.split("\n", 1)[0] == password
    assert "IFS= read -r RPK_PASS" in joined


def test_authenticated_rpk_carries_the_password_on_stdin(
    recorder: _Recorder, broker: harness.RedpandaSasl
) -> None:
    broker.rpk("topic", "list")
    assert len(recorder.calls) == 1
    _assert_password_only_on_stdin(recorder.calls[0], _PASSWORD, "harness-user")
    argv = recorder.calls[0][0]
    assert argv[argv.index("rpk") + 1 : argv.index("rpk") + 3] == ["topic", "list"]


def test_produce_sends_the_password_line_then_the_records(
    recorder: _Recorder, broker: harness.RedpandaSasl
) -> None:
    broker.produce("t", ["a", "b"])
    call = recorder.calls[0]
    _assert_password_only_on_stdin(call, _PASSWORD, "harness-user")
    assert call[1] == f"{_PASSWORD}\na\nb\n"


def test_rpk_as_uses_the_named_principals_password(
    recorder: _Recorder, broker: harness.RedpandaSasl
) -> None:
    principal = harness.RuntimePrincipal(
        username="other-user", password=_OTHER_PASSWORD
    )
    broker.rpk_as(principal, "topic", "list")
    call = recorder.calls[0]
    _assert_password_only_on_stdin(call, _OTHER_PASSWORD, "other-user")
    assert _PASSWORD not in " ".join(call[0])


def test_unauthenticated_control_still_sends_no_credential(
    recorder: _Recorder, broker: harness.RedpandaSasl
) -> None:
    """Positive control: the no-credential call is not turned into an authenticated one."""
    broker.rpk_unauthenticated("topic", "list")
    argv, stdin = recorder.calls[0]
    assert "RPK_USER=harness-user" not in argv
    assert not any(item.startswith("RPK_") for item in argv)
    assert stdin is None
