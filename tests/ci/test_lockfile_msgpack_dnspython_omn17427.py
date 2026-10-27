# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: retain the msgpack and dnspython security updates in uv.lock.

The previous lock pinned msgpack 1.2.1 (GHSA-j586-36cw-2gc2) and dnspython
2.8.0 (GHSA-45c3-73f7-pv4m). Read every lock entry so a second resolution
cannot retain the old version. Exercise the installed replacements locally
without contacting a DNS server or any deployed lane.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import dns.message
import dns.rdatatype
import dns.rrset
import dns.version
import msgpack
import pytest
from packaging.version import Version

pytestmark = pytest.mark.unit

UV_LOCK = Path(__file__).resolve().parents[2] / "uv.lock"


@pytest.mark.parametrize(
    ("name", "floor", "advisory"),
    [
        ("msgpack", "1.2.4", "GHSA-j586-36cw-2gc2"),
        ("dnspython", "2.9.0", "GHSA-45c3-73f7-pv4m"),
    ],
    ids=["msgpack", "dnspython"],
)
def test_lockfile_security_floor(name: str, floor: str, advisory: str) -> None:
    with UV_LOCK.open("rb") as handle:
        lock = tomllib.load(handle)
    versions = [p["version"] for p in lock["package"] if p["name"] == name]
    assert versions, f"uv.lock has no {name} entry"
    vulnerable = [version for version in versions if Version(version) < Version(floor)]
    assert not vulnerable, f"{name} {vulnerable} is below {floor} ({advisory})"


def test_installed_msgpack_round_trip() -> None:
    assert Version(msgpack.__version__) >= Version("1.2.4")
    event = {"topic": "onex.evt.example.v1", "payload": b"\x00\xff", "ids": [1, 2]}
    packed = msgpack.packb(event, use_bin_type=True)
    assert msgpack.unpackb(packed, raw=False) == event
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(packed[:3])
    assert list(unpacker) == []
    unpacker.feed(packed[3:] + packed)
    assert list(unpacker) == [event, event]


def test_installed_dnspython_wire_round_trip() -> None:
    assert Version(dns.version.version) >= Version("2.9.0")
    query = dns.message.make_query("example.invalid.", dns.rdatatype.A)
    response = dns.message.make_response(query)
    response.answer.append(
        dns.rrset.from_text("example.invalid.", 60, "IN", "A", "192.0.2.1")
    )
    decoded = dns.message.from_wire(response.to_wire())
    assert query.is_response(decoded)
    assert decoded.question == query.question
    assert decoded.answer == response.answer
