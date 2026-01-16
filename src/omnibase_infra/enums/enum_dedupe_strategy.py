# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Deduplication strategy enum for corpus capture.

Defines how duplicate executions are detected and handled.

.. versionadded:: 0.5.0
    Added for CorpusCapture (OMN-1203)
"""

from enum import Enum


class EnumDedupeStrategy(str, Enum):
    """
    Deduplication strategies for corpus capture.

    When capturing executions, duplicates can be detected and filtered
    using different strategies based on what constitutes a "duplicate".

    .. versionadded:: 0.5.0
        Added for CorpusCapture (OMN-1203)
    """

    NONE = "none"
    """No deduplication. All executions are captured."""

    INPUT_HASH = "input_hash"
    """
    Deduplicate based on input hash.

    Two executions with the same handler and input data are considered
    duplicates. This is useful for capturing unique test cases without
    redundant variations.
    """

    FULL_MANIFEST_HASH = "full_manifest_hash"
    """
    Deduplicate based on full manifest hash.

    Only exact duplicates (same handler, input, output, timing, etc.)
    are filtered. This preserves executions that had different outcomes
    even with the same input.
    """
