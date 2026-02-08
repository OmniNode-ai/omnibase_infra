# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelPluginDiscoveryReport and PluginDiscoveryEntry.

This test module provides comprehensive coverage for the plugin discovery
report dataclasses, which provide structured diagnostics for plugin loading.

Tests cover:
    - PluginDiscoveryEntry construction and defaults
    - ModelPluginDiscoveryReport construction and defaults
    - ``rejected`` property filtering
    - ``has_errors`` property detection logic
    - Edge cases (empty entries, all-accepted, all-rejected)
    - Typical usage patterns

.. versionadded:: 0.8.0
    Created as part of OMN-2012.
"""

from __future__ import annotations

from omnibase_infra.runtime.models.model_plugin_discovery_report import (
    ModelPluginDiscoveryReport,
    PluginDiscoveryEntry,
)

# ---------------------------------------------------------------------------
# PluginDiscoveryEntry
# ---------------------------------------------------------------------------


class TestPluginDiscoveryEntryConstruction:
    """Tests for PluginDiscoveryEntry construction and field defaults."""

    def test_construct_accepted_entry(self) -> None:
        """Verify accepted entry with all fields."""
        entry = PluginDiscoveryEntry(
            entry_point_name="my_plugin",
            module_path="myapp.plugins.my_plugin",
            status="accepted",
            plugin_id="my_plugin",
        )

        assert entry.entry_point_name == "my_plugin"
        assert entry.module_path == "myapp.plugins.my_plugin"
        assert entry.status == "accepted"
        assert entry.reason == ""
        assert entry.plugin_id == "my_plugin"

    def test_construct_rejected_entry(self) -> None:
        """Verify rejected entry with reason and no plugin_id."""
        entry = PluginDiscoveryEntry(
            entry_point_name="bad_plugin",
            module_path="myapp.plugins.bad",
            status="import_error",
            reason="ModuleNotFoundError: No module named 'myapp.plugins.bad'",
        )

        assert entry.status == "import_error"
        assert "ModuleNotFoundError" in entry.reason
        assert entry.plugin_id is None

    def test_defaults_reason_empty_string(self) -> None:
        """Verify reason defaults to empty string."""
        entry = PluginDiscoveryEntry(
            entry_point_name="x",
            module_path="m",
            status="accepted",
        )

        assert entry.reason == ""

    def test_defaults_plugin_id_none(self) -> None:
        """Verify plugin_id defaults to None."""
        entry = PluginDiscoveryEntry(
            entry_point_name="x",
            module_path="m",
            status="namespace_rejected",
            reason="blocked",
        )

        assert entry.plugin_id is None

    def test_all_valid_statuses(self) -> None:
        """Verify all documented status values can be set."""
        valid_statuses = [
            "accepted",
            "namespace_rejected",
            "import_error",
            "instantiation_error",
            "protocol_invalid",
            "duplicate_skipped",
        ]

        for status in valid_statuses:
            entry = PluginDiscoveryEntry(
                entry_point_name="test",
                module_path="test.module",
                status=status,
            )
            assert entry.status == status

    def test_duplicate_skipped_has_plugin_id(self) -> None:
        """Verify duplicate_skipped entries carry the original plugin_id."""
        entry = PluginDiscoveryEntry(
            entry_point_name="dup_plugin",
            module_path="pkg.dup",
            status="duplicate_skipped",
            reason="Already registered by pkg.original",
            plugin_id="shared_id",
        )

        assert entry.status == "duplicate_skipped"
        assert entry.plugin_id == "shared_id"
        assert entry.reason != ""


# ---------------------------------------------------------------------------
# ModelPluginDiscoveryReport — construction
# ---------------------------------------------------------------------------


class TestModelPluginDiscoveryReportConstruction:
    """Tests for ModelPluginDiscoveryReport construction and defaults."""

    def test_construct_minimal(self) -> None:
        """Verify construction with only required fields."""
        report = ModelPluginDiscoveryReport(
            group="omnibase_infra.projectors",
            discovered_count=0,
        )

        assert report.group == "omnibase_infra.projectors"
        assert report.discovered_count == 0
        assert report.accepted == []
        assert report.entries == []

    def test_construct_full(self) -> None:
        """Verify construction with all fields populated."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="good",
                module_path="pkg.good",
                status="accepted",
                plugin_id="good",
            ),
            PluginDiscoveryEntry(
                entry_point_name="bad",
                module_path="pkg.bad",
                status="import_error",
                reason="No module named 'pkg.bad'",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="my.plugins",
            discovered_count=2,
            accepted=["good"],
            entries=entries,
        )

        assert report.group == "my.plugins"
        assert report.discovered_count == 2
        assert report.accepted == ["good"]
        assert len(report.entries) == 2

    def test_accepted_preserves_order(self) -> None:
        """Verify accepted list preserves registration order."""
        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=3,
            accepted=["alpha", "bravo", "charlie"],
        )

        assert report.accepted == ["alpha", "bravo", "charlie"]


# ---------------------------------------------------------------------------
# ModelPluginDiscoveryReport — rejected property
# ---------------------------------------------------------------------------


class TestModelPluginDiscoveryReportRejected:
    """Tests for the ``rejected`` property."""

    def test_rejected_filters_non_accepted(self) -> None:
        """Verify rejected returns only non-accepted entries."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="good",
                module_path="pkg.good",
                status="accepted",
                plugin_id="good",
            ),
            PluginDiscoveryEntry(
                entry_point_name="blocked",
                module_path="pkg.blocked",
                status="namespace_rejected",
                reason="not in allowlist",
            ),
            PluginDiscoveryEntry(
                entry_point_name="broken",
                module_path="pkg.broken",
                status="import_error",
                reason="SyntaxError",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=3,
            accepted=["good"],
            entries=entries,
        )

        rejected = report.rejected
        assert len(rejected) == 2
        assert rejected[0].entry_point_name == "blocked"
        assert rejected[1].entry_point_name == "broken"

    def test_rejected_empty_when_all_accepted(self) -> None:
        """Verify rejected is empty when all entries are accepted."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="a",
                module_path="m.a",
                status="accepted",
                plugin_id="a",
            ),
            PluginDiscoveryEntry(
                entry_point_name="b",
                module_path="m.b",
                status="accepted",
                plugin_id="b",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=2,
            accepted=["a", "b"],
            entries=entries,
        )

        assert report.rejected == []

    def test_rejected_all_when_none_accepted(self) -> None:
        """Verify rejected returns all entries when none are accepted."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="x",
                module_path="m.x",
                status="namespace_rejected",
                reason="blocked",
            ),
            PluginDiscoveryEntry(
                entry_point_name="y",
                module_path="m.y",
                status="protocol_invalid",
                reason="missing method",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=2,
            entries=entries,
        )

        assert len(report.rejected) == 2

    def test_rejected_empty_when_no_entries(self) -> None:
        """Verify rejected is empty when entries list is empty."""
        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=0,
        )

        assert report.rejected == []

    def test_rejected_preserves_discovery_order(self) -> None:
        """Verify rejected entries maintain their original order."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="c_blocked",
                module_path="m.c",
                status="namespace_rejected",
                reason="blocked",
            ),
            PluginDiscoveryEntry(
                entry_point_name="a_good",
                module_path="m.a",
                status="accepted",
                plugin_id="a_good",
            ),
            PluginDiscoveryEntry(
                entry_point_name="b_error",
                module_path="m.b",
                status="import_error",
                reason="fail",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=3,
            accepted=["a_good"],
            entries=entries,
        )

        rejected = report.rejected
        assert [e.entry_point_name for e in rejected] == ["c_blocked", "b_error"]


# ---------------------------------------------------------------------------
# ModelPluginDiscoveryReport — has_errors property
# ---------------------------------------------------------------------------


class TestModelPluginDiscoveryReportHasErrors:
    """Tests for the ``has_errors`` property."""

    def test_has_errors_true_for_import_error(self) -> None:
        """Verify has_errors detects import_error."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="broken",
                module_path="pkg.broken",
                status="import_error",
                reason="No module",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=1,
            entries=entries,
        )

        assert report.has_errors is True

    def test_has_errors_true_for_instantiation_error(self) -> None:
        """Verify has_errors detects instantiation_error."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="crash",
                module_path="pkg.crash",
                status="instantiation_error",
                reason="TypeError in __init__",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=1,
            entries=entries,
        )

        assert report.has_errors is True

    def test_has_errors_false_for_policy_rejections(self) -> None:
        """Verify policy rejections are NOT treated as errors."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="ns_blocked",
                module_path="pkg.ns",
                status="namespace_rejected",
                reason="not in allowlist",
            ),
            PluginDiscoveryEntry(
                entry_point_name="proto_bad",
                module_path="pkg.proto",
                status="protocol_invalid",
                reason="missing handle()",
            ),
            PluginDiscoveryEntry(
                entry_point_name="dup",
                module_path="pkg.dup",
                status="duplicate_skipped",
                reason="already registered",
                plugin_id="dup",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=3,
            entries=entries,
        )

        assert report.has_errors is False

    def test_has_errors_false_for_all_accepted(self) -> None:
        """Verify has_errors is False when all entries are accepted."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="good",
                module_path="pkg.good",
                status="accepted",
                plugin_id="good",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=1,
            accepted=["good"],
            entries=entries,
        )

        assert report.has_errors is False

    def test_has_errors_false_for_empty_entries(self) -> None:
        """Verify has_errors is False when entries list is empty."""
        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=0,
        )

        assert report.has_errors is False

    def test_has_errors_true_mixed_with_accepted(self) -> None:
        """Verify has_errors is True even when some entries are accepted."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="good",
                module_path="pkg.good",
                status="accepted",
                plugin_id="good",
            ),
            PluginDiscoveryEntry(
                entry_point_name="broken",
                module_path="pkg.broken",
                status="import_error",
                reason="failed",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=2,
            accepted=["good"],
            entries=entries,
        )

        assert report.has_errors is True


# ---------------------------------------------------------------------------
# Usage patterns
# ---------------------------------------------------------------------------


class TestModelPluginDiscoveryReportUsagePatterns:
    """Tests demonstrating typical usage patterns."""

    def test_import_from_package_init(self) -> None:
        """Verify both classes are importable from the package __init__."""
        from omnibase_infra.runtime.models import (
            ModelPluginDiscoveryReport as Report,
        )
        from omnibase_infra.runtime.models import (
            PluginDiscoveryEntry as Entry,
        )

        entry = Entry(
            entry_point_name="test",
            module_path="test.mod",
            status="accepted",
            plugin_id="test",
        )
        report = Report(
            group="test.group",
            discovered_count=1,
            accepted=["test"],
            entries=[entry],
        )

        assert report.group == "test.group"

    def test_debugging_workflow(self) -> None:
        """Demonstrate the 10-second debugging workflow."""
        # Simulate a real discovery pass
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="registration_projector",
                module_path="omnibase_infra.projectors.registration",
                status="accepted",
                plugin_id="registration_projector",
            ),
            PluginDiscoveryEntry(
                entry_point_name="metrics_projector",
                module_path="omnibase_infra.projectors.metrics",
                status="namespace_rejected",
                reason="omnibase_infra.projectors.metrics not in allowed namespaces",
            ),
            PluginDiscoveryEntry(
                entry_point_name="custom_projector",
                module_path="thirdparty.custom",
                status="import_error",
                reason="ModuleNotFoundError: No module named 'thirdparty'",
            ),
        ]

        report = ModelPluginDiscoveryReport(
            group="omnibase_infra.projectors",
            discovered_count=3,
            accepted=["registration_projector"],
            entries=entries,
        )

        # Quick checks a developer would run
        assert report.discovered_count == 3
        assert len(report.accepted) == 1
        assert report.has_errors is True
        assert len(report.rejected) == 2

        # Find the actual error (not just a policy rejection)
        errors = [
            e
            for e in report.entries
            if e.status in ("import_error", "instantiation_error")
        ]
        assert len(errors) == 1
        assert errors[0].entry_point_name == "custom_projector"
        assert "thirdparty" in errors[0].reason

    def test_equality_of_entries(self) -> None:
        """Verify dataclass equality works for entries."""
        entry1 = PluginDiscoveryEntry(
            entry_point_name="x",
            module_path="m.x",
            status="accepted",
            plugin_id="x",
        )
        entry2 = PluginDiscoveryEntry(
            entry_point_name="x",
            module_path="m.x",
            status="accepted",
            plugin_id="x",
        )

        assert entry1 == entry2

    def test_equality_of_reports(self) -> None:
        """Verify dataclass equality works for reports."""
        entries = [
            PluginDiscoveryEntry(
                entry_point_name="x",
                module_path="m.x",
                status="accepted",
                plugin_id="x",
            ),
        ]

        report1 = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=1,
            accepted=["x"],
            entries=list(entries),
        )
        report2 = ModelPluginDiscoveryReport(
            group="g",
            discovered_count=1,
            accepted=["x"],
            entries=list(entries),
        )

        assert report1 == report2


__all__ = [
    "TestPluginDiscoveryEntryConstruction",
    "TestModelPluginDiscoveryReportConstruction",
    "TestModelPluginDiscoveryReportRejected",
    "TestModelPluginDiscoveryReportHasErrors",
    "TestModelPluginDiscoveryReportUsagePatterns",
]
