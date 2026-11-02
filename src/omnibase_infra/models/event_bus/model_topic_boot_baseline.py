# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The recorded count of topic specs a runtime boot provisions (OMN-21288).

Every dev runtime process builds its provisioning list from contracts and
creates the whole list at boot (``TopicProvisioner._build_topic_specs``), so a
contract that declares one more topic grows what every boot creates. The count
a lane is held to is DATA: ``config/boot_topic_specs.yaml`` carries it with the
time it was read and where it came from, this model types that record, and the
check compares the live provisioning list with it. Lowering the baseline after
a fold is a data edit, never a code edit.

Two figures are recorded because two lists exist:

* ``specs_per_boot`` is what the running dev image logged
  (``topic provisioning (contract-first) - total: N``). It includes every
  installed package and marketplace skill manifest, none of which a unit test
  environment holds. It is the figure T28's after-reading is compared with.
* ``repo_specs`` is the same provisioning code run over this repository's own
  contracts and standalone manifests, which is the list a unit test can build.
  It is the figure the CI check holds the code to.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_READ_AT_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


class ModelTopicBootBaseline(BaseModel):
    """One lane's recorded topic-spec count per boot, with its provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    specs_per_boot: int = Field(ge=0)
    read_at: str = Field(pattern=_READ_AT_PATTERN)
    source: str = Field(min_length=1)
    repo_specs: int = Field(ge=0)
    repo_specs_measured_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _repo_list_is_part_of_the_boot_list(self) -> ModelTopicBootBaseline:
        if self.repo_specs > self.specs_per_boot:
            raise ValueError(
                f"repo_specs ({self.repo_specs}) cannot exceed specs_per_boot "
                f"({self.specs_per_boot}): the repository's own list is a part "
                "of what the image provisions"
            )
        return self

    @classmethod
    def from_yaml_file(cls, path: Path, *, lane: str) -> ModelTopicBootBaseline:
        """Load the record for ``lane`` from a baseline data file."""
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        lanes = document.get("lanes") if isinstance(document, dict) else None
        if not isinstance(lanes, dict) or lane not in lanes:
            raise ValueError(f"{path} declares no topic baseline for lane {lane!r}")
        return cls.model_validate(
            {
                "schema_version": str(document.get("schema_version", "")),
                "lane": lane,
                **lanes[lane],
            }
        )

    def repo_specs_over(self, spec_count: int) -> int:
        """Return how many specs ``spec_count`` is above the repo baseline."""
        return max(0, spec_count - self.repo_specs)


__all__: list[str] = ["ModelTopicBootBaseline"]
