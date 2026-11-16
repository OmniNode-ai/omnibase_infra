# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-merge runtime pool driver's judge overlay (OMN-23715).

``node_lab_proof_judge_compute``'s command is read from an overlay root, not
from the frozen workspace-config pool overlay (``ModelPreprPoolOverlay``): the
command names a judge node that lives in the overlays repository, so it moves
with that repository instead of the workspace's pool config. The first root
holding ``<root>/prepr_runtime_pool/overlay.yaml`` wins; a root holding no such
file is skipped, so no overlay anywhere is an empty command and the
``changed_path_live`` check it judges stays absent. An overlay that exists but
does not validate is a loud refusal, never a quiet skip.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

OVERLAY_ID = "prepr_runtime_pool"


class ModelPreprPoolJudgeOverlay(BaseModel):
    """One judge overlay file, validated whole."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    #: the argv of node_lab_proof_judge_compute, environment-expanded before its argv split
    lab_proof_judge_command: str = Field(min_length=1)

    @classmethod
    def command(cls, environ: Mapping[str, str], config_root: Path) -> str:
        """The judge command from the first overlay root holding one, or empty.

        The roots, in order: the ``ONEX_SKILL_OVERLAY_ROOTS`` entries, then the
        overlays clone beside the workspace config root.
        """
        roots = [
            Path(entry)
            for entry in environ.get("ONEX_SKILL_OVERLAY_ROOTS", "").split(os.pathsep)
            if entry
        ]
        roots.append(config_root.parent / "overlays" / "overlays")
        for root in roots:
            path = root / OVERLAY_ID / "overlay.yaml"
            if not path.is_file():
                continue
            return cls.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            ).lab_proof_judge_command
        return ""
