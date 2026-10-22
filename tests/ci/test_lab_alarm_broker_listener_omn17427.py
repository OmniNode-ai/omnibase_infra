# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab alarm raises when the dev broker's external listener stops answering (OMN-17427).

On 2026-10-20 the dev broker crash-looped from 15:19:18Z and its external
listener (:19092) stopped answering. The restart bound saw the broker at
15:25:57Z, but only because the container kept restarting: a broker that is
up with its external listener unbound or unreachable (the advertised-host
shape of OMN-15173) restarts nothing, and the consumer-lag condition reads a
broker that refuses as INDETERMINATE, never as an alarm. Every dev-lane client
outside the compose network (hook capture, ``onex scm``, the ledger writers,
the deploy agent) dials that listener, so it is a condition of its own.

Falsifiers (refused, timed out) and positive controls (answers, a real bound
socket) are paired, and a name that does not resolve is INDETERMINATE: the
alarm host failing to resolve the broker says nothing about the broker.
"""

from __future__ import annotations

import socket
import threading

import pytest

from scripts.lab_alarm import (
    EnumAlarmCondition,
    EnumConditionOutcome,
    evaluate_broker_listener,
    probe_tcp_listener,
)

pytestmark = [pytest.mark.unit]

BOOTSTRAP = "lab-host:19092"


class _RaisingProbe:
    """A ListenerProbe that records each dial and raises ``exc``."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls: list[tuple[str, int, float]] = []

    def __call__(self, host: str, port: int, *, timeout: float) -> None:
        self.calls.append((host, port, timeout))
        raise self.exc


def _probe_answering(host: str, port: int, *, timeout: float) -> None:
    return None


def test_a_refused_listener_is_an_alarm() -> None:
    report = evaluate_broker_listener(
        BOOTSTRAP, probe=_RaisingProbe(ConnectionRefusedError(111, "refused"))
    )
    assert report.condition is EnumAlarmCondition.BROKER_LISTENER
    assert report.outcome is EnumConditionOutcome.ALARM
    (alarm,) = report.alarms
    assert alarm.subject == f"broker-listener:{BOOTSTRAP}"
    assert BOOTSTRAP in alarm.detail
    assert "refused" in alarm.detail


def test_a_listener_that_times_out_is_an_alarm() -> None:
    report = evaluate_broker_listener(BOOTSTRAP, probe=_RaisingProbe(TimeoutError()))
    assert report.outcome is EnumConditionOutcome.ALARM
    assert report.alarms[0].subject == f"broker-listener:{BOOTSTRAP}"


def test_an_answering_listener_is_ok() -> None:
    report = evaluate_broker_listener(BOOTSTRAP, probe=_probe_answering)
    assert report.outcome is EnumConditionOutcome.OK
    assert report.alarms == ()
    assert BOOTSTRAP in report.evidence


def test_one_answering_bootstrap_address_is_enough() -> None:
    def probe(host: str, port: int, *, timeout: float) -> None:
        if host == "down":
            raise ConnectionRefusedError(111, "refused")

    report = evaluate_broker_listener("down:19092,up:19092", probe=probe)
    assert report.outcome is EnumConditionOutcome.OK


def test_an_unresolvable_name_is_indeterminate_not_ok() -> None:
    report = evaluate_broker_listener(
        BOOTSTRAP, probe=_RaisingProbe(socket.gaierror(-2, "Name or service not known"))
    )
    assert report.outcome is EnumConditionOutcome.INDETERMINATE
    assert report.alarms == ()


def test_a_malformed_bootstrap_is_indeterminate() -> None:
    report = evaluate_broker_listener("no-port-here", probe=_probe_answering)
    assert report.outcome is EnumConditionOutcome.INDETERMINATE


def test_the_probe_is_given_the_declared_host_port_and_a_bounded_timeout() -> None:
    probe = _RaisingProbe(ConnectionRefusedError(111, "refused"))
    evaluate_broker_listener(BOOTSTRAP, probe=probe, timeout_seconds=5.0)
    assert probe.calls == [("lab-host", 19092, 5.0)]


def test_the_real_probe_reads_an_api_versions_answer_as_answering() -> None:
    """Positive control for the real probe: a socket that answers ApiVersions is OK."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def answer() -> None:
            conn, _ = server.accept()
            with conn:
                request = conn.recv(64)
                correlation = request[8:12]
                conn.sendall((6).to_bytes(4, "big") + correlation + b"\x00\x00")

        worker = threading.Thread(target=answer)
        worker.start()
        report = evaluate_broker_listener(
            f"127.0.0.1:{port}", probe=probe_tcp_listener, timeout_seconds=5.0
        )
        worker.join(timeout=5)
    assert report.outcome is EnumConditionOutcome.OK, report.evidence


def test_the_real_probe_refuses_a_socket_that_accepts_but_never_answers_kafka() -> None:
    """Falsifier: a port proxy that accepts and closes is not a serving broker."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_and_close() -> None:
            conn, _ = server.accept()
            conn.close()

        worker = threading.Thread(target=accept_and_close)
        worker.start()
        report = evaluate_broker_listener(
            f"127.0.0.1:{port}", probe=probe_tcp_listener, timeout_seconds=5.0
        )
        worker.join(timeout=5)
    assert report.outcome is EnumConditionOutcome.ALARM


def test_the_real_probe_reads_a_closed_port_as_an_alarm() -> None:
    """Falsifier for the real probe: a port nothing listens on is refused."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]
    report = evaluate_broker_listener(f"127.0.0.1:{port}", probe=probe_tcp_listener)
    assert report.outcome is EnumConditionOutcome.ALARM
