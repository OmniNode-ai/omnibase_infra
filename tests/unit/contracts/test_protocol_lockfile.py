# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot tests for protocol lockfile integrity.

These tests verify that the protocol definitions in ``omnibase_infra.protocols``
match the committed lockfile at ``contracts/runtime/runtime_protocol.lock.json``.

If a test fails, it means a protocol signature has changed. This is intentional
protection against accidental breaking changes to protocol interfaces.

To update the lockfile after an intentional protocol change::

    uv run python -c "
    from omnibase_infra.runtime.protocol_lockfile import write_lockfile
    write_lockfile()
    "

Then commit the updated lockfile alongside the protocol changes.

Related:
    - OMN-335: Add Protocol Lockfile Snapshot Tests
    - omnibase_infra.runtime.protocol_lockfile: Lockfile generator

.. versionadded:: 0.11.0
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.runtime.protocol_lockfile import (
    generate_lockfile,
    load_lockfile,
)

# Resolve repo root from test file location
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOCKFILE_PATH = _REPO_ROOT / "contracts" / "runtime" / "runtime_protocol.lock.json"


@pytest.mark.unit
class TestProtocolLockfileExists:
    """Verify the lockfile exists and is valid JSON."""

    def test_lockfile_exists(self) -> None:
        """The lockfile must exist in the repository."""
        assert _LOCKFILE_PATH.exists(), (
            f"Protocol lockfile not found at {_LOCKFILE_PATH}. "
            "Generate it with: "
            'uv run python -c "from omnibase_infra.runtime.protocol_lockfile '
            'import write_lockfile; write_lockfile()"'
        )

    def test_lockfile_is_valid_json(self) -> None:
        """The lockfile must be valid JSON."""
        if not _LOCKFILE_PATH.exists():
            pytest.skip("Lockfile does not exist")
        content = _LOCKFILE_PATH.read_text(encoding="utf-8")
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_lockfile_has_schema_version(self) -> None:
        """The lockfile must include a schema version."""
        lockfile = load_lockfile(_REPO_ROOT)
        assert "schema_version" in lockfile
        assert lockfile["schema_version"] == "1.0.0"

    def test_lockfile_has_package_version(self) -> None:
        """The lockfile must include the package version (envelope schema version)."""
        lockfile = load_lockfile(_REPO_ROOT)
        assert "package_version" in lockfile
        assert isinstance(lockfile["package_version"], str)
        # Verify it looks like a semver
        parts = lockfile["package_version"].split(".")
        assert len(parts) >= 2, (
            f"Package version '{lockfile['package_version']}' does not look like semver"
        )


@pytest.mark.unit
class TestProtocolLockfileSnapshot:
    """Snapshot comparison: lockfile vs current protocol definitions.

    These are the core tests. If any of them fail, a protocol interface has
    changed and the lockfile needs to be regenerated.
    """

    def test_protocol_count_matches(self) -> None:
        """Number of protocols in lockfile must match current definitions."""
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()
        assert lockfile["protocol_count"] == current["protocol_count"], (
            f"Protocol count mismatch: lockfile has {lockfile['protocol_count']}, "
            f"current has {current['protocol_count']}. "
            "A protocol was added or removed. Regenerate the lockfile."
        )

    def test_protocol_names_match(self) -> None:
        """Protocol names in lockfile must match current definitions."""
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()
        lockfile_names = set(lockfile["protocols"].keys())
        current_names = set(current["protocols"].keys())

        added = current_names - lockfile_names
        removed = lockfile_names - current_names

        assert not added, (
            f"New protocols not in lockfile: {sorted(added)}. Regenerate the lockfile."
        )
        assert not removed, (
            f"Protocols in lockfile but not in code: {sorted(removed)}. "
            "Regenerate the lockfile."
        )

    def test_method_counts_match(self) -> None:
        """Method counts per protocol must match between lockfile and current."""
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()

        mismatches: list[str] = []
        for name in lockfile["protocols"]:
            if name not in current["protocols"]:
                continue
            lock_count = lockfile["protocols"][name]["method_count"]
            curr_count = current["protocols"][name]["method_count"]
            if lock_count != curr_count:
                mismatches.append(
                    f"  {name}: lockfile={lock_count}, current={curr_count}"
                )

        assert not mismatches, (
            "Method count mismatches detected (handler protocol versions changed):\n"
            + "\n".join(mismatches)
            + "\nRegenerate the lockfile."
        )

    def test_method_names_match(self) -> None:
        """Method names per protocol must match between lockfile and current."""
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()

        mismatches: list[str] = []
        for name in lockfile["protocols"]:
            if name not in current["protocols"]:
                continue
            lock_methods = set(lockfile["protocols"][name]["methods"].keys())
            curr_methods = set(current["protocols"][name]["methods"].keys())

            added = curr_methods - lock_methods
            removed = lock_methods - curr_methods

            if added:
                mismatches.append(f"  {name}: added methods {sorted(added)}")
            if removed:
                mismatches.append(f"  {name}: removed methods {sorted(removed)}")

        assert not mismatches, (
            "Method name mismatches detected:\n"
            + "\n".join(mismatches)
            + "\nRegenerate the lockfile."
        )

    def test_method_signatures_match(self) -> None:
        """Full method signatures must match between lockfile and current.

        This is the most granular check: it compares parameter names, types,
        kinds (positional, keyword-only, etc.), defaults, and return types.
        """
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()

        mismatches: list[str] = []
        for proto_name in lockfile["protocols"]:
            if proto_name not in current["protocols"]:
                continue
            lock_methods = lockfile["protocols"][proto_name]["methods"]
            curr_methods = current["protocols"][proto_name]["methods"]

            for method_name in lock_methods:
                if method_name not in curr_methods:
                    continue

                lock_sig = lock_methods[method_name]
                curr_sig = curr_methods[method_name]

                # Compare parameters
                if lock_sig["parameters"] != curr_sig["parameters"]:
                    mismatches.append(
                        f"  {proto_name}.{method_name}: parameter signature changed\n"
                        f"    lockfile: {json.dumps(lock_sig['parameters'], indent=6)}\n"
                        f"    current:  {json.dumps(curr_sig['parameters'], indent=6)}"
                    )

                # Compare return annotation
                if lock_sig["return_annotation"] != curr_sig["return_annotation"]:
                    mismatches.append(
                        f"  {proto_name}.{method_name}: return type changed\n"
                        f"    lockfile: {lock_sig['return_annotation']}\n"
                        f"    current:  {curr_sig['return_annotation']}"
                    )

        assert not mismatches, (
            "Method signature mismatches detected:\n"
            + "\n".join(mismatches)
            + "\nRegenerate the lockfile if these changes are intentional."
        )

    def test_runtime_checkable_status_matches(self) -> None:
        """runtime_checkable status must match between lockfile and current."""
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()

        mismatches: list[str] = []
        for name in lockfile["protocols"]:
            if name not in current["protocols"]:
                continue
            lock_rc = lockfile["protocols"][name]["runtime_checkable"]
            curr_rc = current["protocols"][name]["runtime_checkable"]
            if lock_rc != curr_rc:
                mismatches.append(f"  {name}: lockfile={lock_rc}, current={curr_rc}")

        assert not mismatches, (
            "runtime_checkable status mismatches detected:\n"
            + "\n".join(mismatches)
            + "\nRegenerate the lockfile."
        )

    def test_full_lockfile_matches(self) -> None:
        """Complete lockfile content must match current protocol definitions.

        This is the ultimate snapshot test. It compares the entire protocol
        section of the lockfile against the freshly generated version.
        The schema_version and package_version are excluded since the package
        version may legitimately change without protocol changes.
        """
        lockfile = load_lockfile(_REPO_ROOT)
        current = generate_lockfile()

        # Compare only the protocols section (not metadata)
        assert lockfile["protocols"] == current["protocols"], (
            "Protocol lockfile is out of sync with current protocol definitions.\n"
            "This means a protocol interface has changed.\n\n"
            "If the change is intentional, regenerate the lockfile:\n"
            '  uv run python -c "from omnibase_infra.runtime.protocol_lockfile '
            'import write_lockfile; write_lockfile()"\n\n'
            "Then commit the updated lockfile alongside your protocol changes."
        )


@pytest.mark.unit
class TestProtocolLockfileGenerator:
    """Tests for the lockfile generator itself."""

    def test_generate_lockfile_returns_dict(self) -> None:
        """generate_lockfile must return a dict."""
        result = generate_lockfile()
        assert isinstance(result, dict)

    def test_generate_lockfile_has_required_keys(self) -> None:
        """Generated lockfile must have all required top-level keys."""
        result = generate_lockfile()
        required_keys = {
            "schema_version",
            "package_version",
            "protocol_count",
            "protocols",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_generate_lockfile_protocols_are_sorted(self) -> None:
        """Protocol entries must be sorted alphabetically."""
        result = generate_lockfile()
        proto_names = list(result["protocols"].keys())
        assert proto_names == sorted(proto_names), (
            "Protocols are not sorted alphabetically"
        )

    def test_generate_lockfile_methods_are_sorted(self) -> None:
        """Method entries within each protocol must be sorted alphabetically."""
        result = generate_lockfile()
        for proto_name, proto_data in result["protocols"].items():
            method_names = list(proto_data["methods"].keys())
            assert method_names == sorted(method_names), (
                f"Methods in {proto_name} are not sorted alphabetically"
            )

    def test_generate_lockfile_is_json_serializable(self) -> None:
        """Generated lockfile must be JSON-serializable."""
        result = generate_lockfile()
        serialized = json.dumps(result, indent=2, sort_keys=False)
        roundtripped = json.loads(serialized)
        assert roundtripped == result

    def test_generate_lockfile_is_deterministic(self) -> None:
        """Calling generate_lockfile twice must produce identical output."""
        first = generate_lockfile()
        second = generate_lockfile()
        assert first == second, "generate_lockfile() is not deterministic"

    def test_all_protocols_have_at_least_one_method(self) -> None:
        """Every protocol in the lockfile must have at least one method."""
        result = generate_lockfile()
        empty_protocols: list[str] = []
        for name, data in result["protocols"].items():
            if data["method_count"] == 0:
                empty_protocols.append(name)

        assert not empty_protocols, f"Protocols with no methods: {empty_protocols}"

    def test_protocol_count_matches_entries(self) -> None:
        """protocol_count must match the number of protocol entries."""
        result = generate_lockfile()
        assert result["protocol_count"] == len(result["protocols"]), (
            f"protocol_count ({result['protocol_count']}) does not match "
            f"number of protocol entries ({len(result['protocols'])})"
        )
