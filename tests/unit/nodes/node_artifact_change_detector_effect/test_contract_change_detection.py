# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerContractFileWatcher and HandlerManualTrigger."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

_CMD_ID = UUID("00000000-0000-0000-0000-000000000001")

from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_manual_trigger import (
    HandlerManualTrigger,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_manual_reconcile_command import (
    ModelManualReconcileCommand,
)

# ---------------------------------------------------------------------------
# HandlerManualTrigger tests (no I/O — pure mapping)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerManualTrigger:
    """Tests for HandlerManualTrigger.handle()."""

    def _make_command(self, **kwargs: object) -> ModelManualReconcileCommand:
        defaults: dict[str, object] = {
            "command_id": _CMD_ID,
            "source_repo": "omnibase_infra",
            "changed_files": [],
            "actor": "jonah",
            "reason": "Manual check after migration",
        }
        defaults.update(kwargs)
        return ModelManualReconcileCommand(**defaults)  # type: ignore[arg-type]

    def test_trigger_type_is_manual_plan_request(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command()
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.trigger_type == "manual_plan_request"

    def test_source_repo_preserved(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command(source_repo="omnibase_infra")
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.source_repo == "omnibase_infra"

    def test_empty_changed_files(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command(changed_files=[])
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.changed_files == []

    def test_changed_files_with_paths(self) -> None:
        files = ["src/omnibase_infra/nodes/foo/contract.yaml"]
        handler = HandlerManualTrigger()
        cmd = self._make_command(changed_files=files)
        trigger = handler.build_reconcile_trigger(cmd)
        assert list(trigger.changed_files) == files

    def test_reason_preserved(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command(reason="Post-migration check")
        trigger = handler.build_reconcile_trigger(cmd)
        assert "Post-migration check" in trigger.reason

    def test_default_reason_includes_command_id(self) -> None:
        handler = HandlerManualTrigger()
        specific_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        cmd = self._make_command(command_id=specific_id, reason="")
        trigger = handler.build_reconcile_trigger(cmd)
        assert str(specific_id) in trigger.reason

    def test_actor_preserved(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command(actor="dev-agent")
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.actor == "dev-agent"

    def test_source_ref_is_none(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command()
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.source_ref is None

    def test_ticket_ids_is_empty(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command()
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.ticket_ids == []

    def test_unique_trigger_id_per_call(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command()
        t1 = handler.build_reconcile_trigger(cmd)
        t2 = handler.build_reconcile_trigger(cmd)
        assert t1.trigger_id != t2.trigger_id

    def test_timestamp_is_set(self) -> None:
        handler = HandlerManualTrigger()
        cmd = self._make_command()
        trigger = handler.build_reconcile_trigger(cmd)
        assert trigger.timestamp is not None

    def test_handler_type_and_category(self) -> None:
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        handler = HandlerManualTrigger()
        assert handler.handler_type == EnumHandlerType.NODE_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


@pytest.mark.unit
class TestModelManualReconcileCommand:
    """Tests for ModelManualReconcileCommand validation."""

    def test_valid_minimal(self) -> None:
        cmd = ModelManualReconcileCommand(
            command_id=_CMD_ID,
            source_repo="omnibase_infra",
        )
        assert cmd.changed_files == []
        assert cmd.actor is None
        assert cmd.reason == ""

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ModelManualReconcileCommand(
                command_id=_CMD_ID,
                source_repo="omnibase_infra",
                bogus_field="nope",  # type: ignore[call-arg]
            )

    def test_model_is_frozen(self) -> None:
        cmd = ModelManualReconcileCommand(
            command_id=_CMD_ID,
            source_repo="omnibase_infra",
        )
        with pytest.raises(Exception):
            cmd.source_repo = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HandlerContractFileWatcher tests (filesystem I/O — use tmpdir)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerContractFileWatcher:
    """Tests for HandlerContractFileWatcher.

    Skipped if watchdog is not installed (optional dependency for this handler).
    """

    @pytest.fixture
    def tmp_nodes_dir(self, tmp_path: Path) -> Path:
        """Create a temporary nodes directory structure with contract.yaml files."""
        for node_name in ["node_foo", "node_bar"]:
            node_dir = tmp_path / node_name
            node_dir.mkdir()
            (node_dir / "contract.yaml").write_text(
                f"name: {node_name}\nnode_type: COMPUTE_GENERIC\n"
            )
        return tmp_path

    def _skip_if_no_watchdog(self) -> None:
        try:
            import watchdog
        except ImportError:
            pytest.skip("watchdog not installed")

    def test_import_error_without_watchdog(self, tmp_path: Path) -> None:
        """HandlerContractFileWatcher raises ImportError if watchdog is absent."""
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers import (
            handler_contract_file_watcher as mod,
        )

        original = mod._WATCHDOG_AVAILABLE
        try:
            mod._WATCHDOG_AVAILABLE = False
            with pytest.raises(ImportError, match="watchdog"):
                from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
                    HandlerContractFileWatcher,
                )

                HandlerContractFileWatcher(
                    watch_root=tmp_path,
                    source_repo="omnibase_infra",
                )
        finally:
            mod._WATCHDOG_AVAILABLE = original

    def test_watch_root_not_found_raises(self, tmp_path: Path) -> None:
        """start() raises FileNotFoundError if watch_root does not exist."""
        self._skip_if_no_watchdog()
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        handler = HandlerContractFileWatcher(
            watch_root=tmp_path / "nonexistent",
            source_repo="omnibase_infra",
        )

        async def run() -> None:
            with pytest.raises(FileNotFoundError):
                await handler.start()

        asyncio.run(run())

    def test_md5_hash_seeding(self, tmp_nodes_dir: Path) -> None:
        """Handler seeds MD5 hashes for all existing contract files on start."""
        self._skip_if_no_watchdog()
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        handler = HandlerContractFileWatcher(
            watch_root=tmp_nodes_dir,
            source_repo="omnibase_infra",
        )

        async def run() -> None:
            await handler.start()
            try:
                # Two contract.yaml files should be seeded
                assert len(handler._file_hashes) == 2
            finally:
                await handler.stop()

        asyncio.run(run())

    def test_no_trigger_on_unchanged_file(self, tmp_nodes_dir: Path) -> None:
        """Writing identical content to a contract file does not emit a trigger."""
        self._skip_if_no_watchdog()
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        contract_file = tmp_nodes_dir / "node_foo" / "contract.yaml"
        original_content = contract_file.read_text()

        handler = HandlerContractFileWatcher(
            watch_root=tmp_nodes_dir,
            source_repo="omnibase_infra",
            debounce_seconds=0.1,
        )

        async def run() -> None:
            await handler.start()
            try:
                # Overwrite with same content
                contract_file.write_text(original_content)
                # Manually add path to pending (simulating watchdog event)
                with handler._pending_lock:
                    handler._pending_paths.append(contract_file)
                # Run debounce
                await handler._debounce_and_process()
                # No trigger expected (hash unchanged)
                triggers = await handler.get_pending_triggers()
                assert triggers == []
            finally:
                await handler.stop()

        asyncio.run(run())

    def test_trigger_emitted_on_content_change(self, tmp_nodes_dir: Path) -> None:
        """Modifying a contract file's content emits a trigger."""
        self._skip_if_no_watchdog()
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        contract_file = tmp_nodes_dir / "node_foo" / "contract.yaml"

        handler = HandlerContractFileWatcher(
            watch_root=tmp_nodes_dir,
            source_repo="omnibase_infra",
            debounce_seconds=0.1,
        )

        async def run() -> None:
            await handler.start()
            try:
                # Modify content
                contract_file.write_text("name: node_foo\nnode_type: EFFECT_GENERIC\n")
                # Manually add to pending
                with handler._pending_lock:
                    handler._pending_paths.append(contract_file)
                # Run debounce processing
                await handler._debounce_and_process()
                # Should emit one trigger
                triggers = await handler.get_pending_triggers()
                assert len(triggers) == 1
                trigger = triggers[0]
                assert trigger.trigger_type == "contract_changed"
                assert trigger.source_repo == "omnibase_infra"
                assert len(trigger.changed_files) == 1
            finally:
                await handler.stop()

        asyncio.run(run())

    def test_changed_files_are_relative_to_watch_root(
        self, tmp_nodes_dir: Path
    ) -> None:
        """Changed file paths in triggers are relative to watch_root."""
        self._skip_if_no_watchdog()
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        contract_file = tmp_nodes_dir / "node_foo" / "contract.yaml"

        handler = HandlerContractFileWatcher(
            watch_root=tmp_nodes_dir,
            source_repo="omnibase_infra",
            debounce_seconds=0.1,
        )

        async def run() -> None:
            await handler.start()
            try:
                contract_file.write_text("name: node_foo\nnode_type: EFFECT_GENERIC\n")
                with handler._pending_lock:
                    handler._pending_paths.append(contract_file)
                await handler._debounce_and_process()
                triggers = await handler.get_pending_triggers()
                assert len(triggers) == 1
                # Path should be relative (not absolute)
                changed = triggers[0].changed_files[0]
                assert not Path(changed).is_absolute()
                assert "node_foo" in changed
            finally:
                await handler.stop()

        asyncio.run(run())

    def test_handler_type_and_category(self, tmp_nodes_dir: Path) -> None:
        self._skip_if_no_watchdog()
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        handler = HandlerContractFileWatcher(
            watch_root=tmp_nodes_dir,
            source_repo="omnibase_infra",
        )
        assert handler.handler_type == EnumHandlerType.NODE_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    def test_get_pending_triggers_empty_initially(self, tmp_nodes_dir: Path) -> None:
        """No triggers queued until a change is detected."""
        self._skip_if_no_watchdog()
        from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_contract_file_watcher import (
            HandlerContractFileWatcher,
        )

        handler = HandlerContractFileWatcher(
            watch_root=tmp_nodes_dir,
            source_repo="omnibase_infra",
        )

        async def run() -> None:
            await handler.start()
            try:
                triggers = await handler.get_pending_triggers()
                assert triggers == []
            finally:
                await handler.stop()

        asyncio.run(run())
