# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Redacted structural manifest for one sim-only archive selection."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.model_sim_archive_selected_record import (
    CanonicalSha256,
    ModelSimArchiveSelectedRecord,
)


class ModelSimArchiveSelectionManifest(BaseModel):
    """Caller-constructible selection claim, intentionally not provenance authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_lane: Literal["sim-202"]
    correlation_sha256: CanonicalSha256
    owner_binding_sha256: CanonicalSha256
    source_scan_sha256: CanonicalSha256
    head_envelope_sha256: CanonicalSha256
    topology_sha256: CanonicalSha256
    selected_records: tuple[ModelSimArchiveSelectedRecord, ...] = Field(min_length=1)
    withheld_envelope_sha256: tuple[CanonicalSha256, ...] = ()

    @model_validator(mode="after")
    def _validate_internal_graph(self) -> Self:
        record_keys = {
            (record.source_topic, record.source_partition, record.source_offset)
            for record in self.selected_records
        }
        if len(record_keys) != len(self.selected_records):
            raise ValueError("selection repeats a broker source coordinate")
        envelopes = {record.envelope_sha256 for record in self.selected_records}
        if len(envelopes) != len(self.selected_records):
            raise ValueError("selection repeats an envelope identity")
        if self.head_envelope_sha256 not in envelopes:
            raise ValueError("selection head is not selected")
        if self.head_envelope_sha256 in self.withheld_envelope_sha256:
            raise ValueError("selection head cannot be withheld")
        if len(set(self.withheld_envelope_sha256)) != len(
            self.withheld_envelope_sha256
        ):
            raise ValueError("withheld envelope identity repeats")
        if envelopes.intersection(self.withheld_envelope_sha256):
            raise ValueError("selected and withheld envelope identities overlap")
        parents = {
            record.parent_envelope_sha256
            for record in self.selected_records
            if record.parent_envelope_sha256 is not None
        }
        if not parents.issubset(envelopes):
            raise ValueError("selection has a parent outside the selected graph")
        head = next(
            record
            for record in self.selected_records
            if record.envelope_sha256 == self.head_envelope_sha256
        )
        if head.parent_envelope_sha256 is not None:
            raise ValueError("selection head records a parent")
        return self


__all__ = ["ModelSimArchiveSelectionManifest"]
