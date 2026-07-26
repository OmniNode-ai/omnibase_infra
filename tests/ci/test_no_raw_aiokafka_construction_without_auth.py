# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard: every direct aiokafka client construction must carry MSK-IAM auth kwargs.

OMN-14155 found ~14 direct ``AIOKafkaConsumer(``/``AIOKafkaProducer(``/
``AIOKafkaAdminClient(`` construction sites that bypassed
``omnibase_infra.event_bus.kafka_auth.build_aiokafka_auth_kwargs`` /
``build_aiokafka_auth_kwargs_from_env`` and passed only ``bootstrap_servers=``
(implicit PLAINTEXT). Several of those sites are docker-catalog-deployed cloud
runtime containers — against an MSK_IAM-secured broker, a construction site
without auth kwargs fails the SASL handshake outright.

``build_aiokafka_auth_kwargs_from_env()`` returns ``{}`` under today's
PLAINTEXT config, so wiring it in everywhere is a no-op until the broker is
cut over to ``KAFKA_SASL_MECHANISM=AWS_MSK_IAM`` — there is no reason for a
new call site to skip it.

This is an AST scan (not a text grep) so it does not false-positive on
docstring examples (e.g. ``>>> producer = AIOKafkaProducer(...)`` in a
module docstring) or on files that only reference the client class for
typing without constructing it.

Ticket: OMN-14155
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "omnibase_infra"

_GUARDED_CALL_NAMES: frozenset[str] = frozenset(
    {"AIOKafkaConsumer", "AIOKafkaProducer", "AIOKafkaAdminClient"}
)

_AUTH_HELPER_NAMES: frozenset[str] = frozenset(
    {
        "build_aiokafka_auth_kwargs",
        "build_aiokafka_auth_kwargs_from_env",
        # service_pattern_b_broker.py reuses an injected EventBusKafka's own
        # auth-kwargs builder instead of calling kafka_auth.py directly —
        # still a legitimate route to the same auth kwargs.
        "_direct_terminal_auth_kwargs",
    }
)

# Files that construct an aiokafka client but are explicitly out of scope.
# Every entry here must carry a reason — this is an allowlist of known,
# reviewed exceptions, not a place to silence new findings.
_ALLOWLIST: dict[str, str] = {
    "cli/cli_kafka.py": (
        "Module docstring explicitly defers SASL/TLS support to V2; "
        "dev-machine CLI convenience tool, not a deployed runtime service."
    ),
}


def _iter_python_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _relative(path: Path) -> str:
    return str(path.relative_to(_SRC_ROOT))


def _file_constructs_guarded_client(tree: ast.AST) -> bool:
    """Return True if the AST contains a real (non-docstring) Call to a guarded client."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name in _GUARDED_CALL_NAMES:
            return True
    return False


def _file_references_auth_helper(tree: ast.AST, source: str) -> bool:
    """Return True if the file calls or imports one of the known auth-kwargs helpers."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name in _AUTH_HELPER_NAMES:
                return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _AUTH_HELPER_NAMES:
                    return True
    # Fallback substring check covers `**build_aiokafka_auth_kwargs_from_env()`
    # spreads that AST-walk already catches as Call nodes above; kept as a
    # belt-and-suspenders check for any call shape this walk might miss.
    return any(helper in source for helper in _AUTH_HELPER_NAMES)


def test_no_raw_aiokafka_construction_without_auth_kwargs() -> None:
    """Every direct aiokafka client construction site must thread auth kwargs.

    Fails if a file constructs ``AIOKafkaConsumer``/``AIOKafkaProducer``/
    ``AIOKafkaAdminClient`` directly without also referencing one of the
    approved auth-kwargs helpers, unless the file is in ``_ALLOWLIST``.
    """
    violations: list[str] = []

    for path in _iter_python_files():
        rel = _relative(path)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue

        if not _file_constructs_guarded_client(tree):
            continue

        if rel in _ALLOWLIST:
            continue

        if not _file_references_auth_helper(tree, source):
            violations.append(rel)

    assert not violations, (
        "Direct aiokafka client construction without MSK-IAM auth kwargs "
        f"(OMN-14155 regression guard): {violations}. Either spread "
        "`**build_aiokafka_auth_kwargs_from_env()` (from "
        "omnibase_infra.event_bus.kafka_auth) into the client constructor, "
        "or add the file to `_ALLOWLIST` in this test with a documented reason."
    )


def test_allowlist_entries_still_construct_a_guarded_client() -> None:
    """Anti-permanence: an allowlist entry that no longer applies must be removed."""
    stale: list[str] = []
    for rel in _ALLOWLIST:
        path = _SRC_ROOT / rel
        if not path.exists():
            stale.append(f"{rel} (file no longer exists)")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _file_constructs_guarded_client(tree):
            stale.append(f"{rel} (no longer constructs a guarded aiokafka client)")

    assert not stale, f"Stale OMN-14155 allowlist entries, remove them: {stale}"
