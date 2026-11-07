# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Node: NodeReleaseNotesApplyEffect - write a release block where the repo's docs policy accepts it.

OMN-17427: the release train wrote ``CHANGELOG.md`` for every repository, so every
omnibase_core release PR was born red once core deleted that file under the
README-only documentation policy. The train's cut job now runs this node, which
reads the target repository's own ``.readme-only-docs.yaml`` and writes the block
where that policy accepts it.
"""

from omnibase_infra.nodes.node_release_notes_apply_effect.node import (
    NodeReleaseNotesApplyEffect,
)

__all__ = ["NodeReleaseNotesApplyEffect"]
