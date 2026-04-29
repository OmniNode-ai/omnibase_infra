# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Models for the waste detection compute node."""

from omnibase_infra.nodes.node_waste_detection_compute.models.model_waste_call import (
    ModelWasteCall,
)
from omnibase_infra.nodes.node_waste_detection_compute.models.model_waste_detection_input import (
    ModelWasteDetectionInput,
)
from omnibase_infra.nodes.node_waste_detection_compute.models.model_waste_finding import (
    ModelWasteFinding,
    WasteSeverity,
)

__all__ = [
    "ModelWasteCall",
    "ModelWasteDetectionInput",
    "ModelWasteFinding",
    "WasteSeverity",
]
