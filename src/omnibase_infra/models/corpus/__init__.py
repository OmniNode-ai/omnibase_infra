# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
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
