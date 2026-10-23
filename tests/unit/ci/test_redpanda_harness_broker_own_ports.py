# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The harness never publishes the broker on a port the broker binds itself.

Bus ACL Boundary (OMN-19928) went red on omnibase_infra#5289 when
``_free_port()`` drew 33145: ``docker run`` succeeded, then Redpanda exited
with "Address already in use" because its internal RPC listener holds 33145
inside the same container namespace the external listener binds.
"""

from __future__ import annotations

import socket

import pytest

from tests.integration.customer_path import redpanda_sasl_harness as harness

pytestmark = pytest.mark.unit


class _FakeSocket:
    def __init__(self, port: int) -> None:
        self._port = port

    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def bind(self, _addr: tuple[str, int]) -> None:
        return None

    def getsockname(self) -> tuple[str, int]:
        return ("127.0.0.1", self._port)


@pytest.mark.parametrize("own_port", sorted(harness._BROKER_OWN_PORTS))
def test_a_drawn_broker_own_port_is_redrawn(
    monkeypatch: pytest.MonkeyPatch, own_port: int
) -> None:
    draws = iter([own_port, 41000])
    monkeypatch.setattr(
        socket, "socket", lambda *_args, **_kwargs: _FakeSocket(next(draws))
    )

    assert harness._free_port() == 41000


def test_the_rpc_port_that_broke_the_job_is_reserved() -> None:
    assert 33145 in harness._BROKER_OWN_PORTS
    assert harness.INTERNAL_PORT in harness._BROKER_OWN_PORTS
