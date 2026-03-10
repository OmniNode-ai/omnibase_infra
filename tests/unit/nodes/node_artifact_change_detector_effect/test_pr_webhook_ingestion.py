# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerPRWebhookIngestion."""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_artifact_change_detector_effect.handlers.handler_pr_webhook_ingestion import (
    HandlerPRWebhookIngestion,
)
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_pr_webhook_event import (
    ModelPRWebhookEvent,
)


@pytest.mark.unit
class TestHandlerPRWebhookIngestion:
    """Tests for HandlerPRWebhookIngestion.handle()."""

    def _make_event(self, **kwargs: object) -> ModelPRWebhookEvent:
        defaults: dict[str, object] = {
            "action": "opened",
            "repo": "OmniNode-ai/omnibase_infra",
            "pr_number": 42,
            "head_ref": "feature/test-branch",
            "head_sha": "abc123def456",
            "changed_files": ["src/omnibase_infra/nodes/foo/contract.yaml"],
            "ticket_ids": ["OMN-1234"],
            "actor": "octocat",
            "merged": False,
        }
        defaults.update(kwargs)
        return ModelPRWebhookEvent(**defaults)  # type: ignore[arg-type]

    def test_opened_maps_to_pr_opened(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(action="opened")
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.trigger_type == "pr_opened"

    def test_synchronize_maps_to_pr_updated(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(action="synchronize")
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.trigger_type == "pr_updated"

    def test_reopened_maps_to_pr_updated(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(action="reopened")
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.trigger_type == "pr_updated"

    def test_edited_maps_to_pr_updated(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(action="edited")
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.trigger_type == "pr_updated"

    def test_closed_merged_maps_to_pr_merged(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(action="closed", merged=True)
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.trigger_type == "pr_merged"

    def test_closed_not_merged_maps_to_pr_updated(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(action="closed", merged=False)
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.trigger_type == "pr_updated"

    def test_trigger_preserves_source_repo(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(repo="OmniNode-ai/omnibase_infra")
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.source_repo == "OmniNode-ai/omnibase_infra"

    def test_trigger_source_ref_is_pr_ref(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(pr_number=99)
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.source_ref == "refs/pull/99/head"

    def test_trigger_preserves_changed_files(self) -> None:
        files = [
            "src/omnibase_infra/nodes/foo/contract.yaml",
            "src/omnibase_infra/nodes/bar/contract.yaml",
        ]
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(changed_files=files)
        trigger = handler.ingest_pr_webhook_event(event)
        assert list(trigger.changed_files) == files

    def test_trigger_preserves_ticket_ids(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(ticket_ids=["OMN-1234", "OMN-5678"])
        trigger = handler.ingest_pr_webhook_event(event)
        assert list(trigger.ticket_ids) == ["OMN-1234", "OMN-5678"]

    def test_trigger_preserves_actor(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(actor="dev-bot")
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.actor == "dev-bot"

    def test_trigger_has_unique_id_per_call(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event()
        t1 = handler.ingest_pr_webhook_event(event)
        t2 = handler.ingest_pr_webhook_event(event)
        assert t1.trigger_id != t2.trigger_id

    def test_trigger_has_timestamp(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event()
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.timestamp is not None

    def test_empty_changed_files_passes_through(self) -> None:
        handler = HandlerPRWebhookIngestion()
        event = self._make_event(changed_files=[])
        trigger = handler.ingest_pr_webhook_event(event)
        assert trigger.changed_files == []

    def test_handler_type_and_category(self) -> None:
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        handler = HandlerPRWebhookIngestion()
        assert handler.handler_type == EnumHandlerType.NODE_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


@pytest.mark.unit
class TestModelPRWebhookEvent:
    """Tests for ModelPRWebhookEvent validation."""

    def test_valid_event_minimal(self) -> None:
        event = ModelPRWebhookEvent(
            action="opened",
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            head_ref="main",
            head_sha="abc123",
        )
        assert event.pr_number == 1
        assert event.changed_files == []
        assert event.ticket_ids == []
        assert event.actor is None
        assert event.merged is False

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(Exception):
            ModelPRWebhookEvent(
                action="invalid_action",  # type: ignore[arg-type]
                repo="OmniNode-ai/omnibase_infra",
                pr_number=1,
                head_ref="main",
                head_sha="abc123",
            )

    def test_pr_number_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ModelPRWebhookEvent(
                action="opened",
                repo="OmniNode-ai/omnibase_infra",
                pr_number=0,  # ge=1 constraint
                head_ref="main",
                head_sha="abc123",
            )

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ModelPRWebhookEvent(
                action="opened",
                repo="OmniNode-ai/omnibase_infra",
                pr_number=1,
                head_ref="main",
                head_sha="abc123",
                bogus_field="nope",  # type: ignore[call-arg]
            )

    def test_model_is_frozen(self) -> None:
        event = ModelPRWebhookEvent(
            action="opened",
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            head_ref="main",
            head_sha="abc123",
        )
        with pytest.raises(Exception):
            event.pr_number = 99  # type: ignore[misc]
