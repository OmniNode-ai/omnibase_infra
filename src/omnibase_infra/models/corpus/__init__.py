# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Corpus capture models.

.. versionadded:: 0.5.0
    Added for CorpusCapture (OMN-1203)
"""

from omnibase_infra.models.corpus.model_capture_config import ModelCaptureConfig
from omnibase_infra.models.corpus.model_capture_result import ModelCaptureResult

__all__ = [
    "ModelCaptureConfig",
    "ModelCaptureResult",
]
