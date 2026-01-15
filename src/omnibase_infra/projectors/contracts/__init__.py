# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Projector Contract Definitions.

This package contains declarative YAML contracts for projectors in the
omnibase_infra layer. Each contract defines the complete specification
for a projector including schema, behavior, and event mappings.

Available Contracts:
    - registration_projector.yaml: Registration domain projector contract
      (matches legacy ProjectorRegistration implementation)

Contract Structure:
    Contracts follow the ModelProjectorContract schema from omnibase_core
    with optional extensions for partial update definitions.

    Core fields (validated by ModelProjectorContract):
        - projector_kind: Type of projector (materialized_view)
        - projector_id: Unique identifier
        - name: Human-readable name
        - version: Contract version string
        - aggregate_type: Domain aggregate type
        - consumed_events: List of event names to consume
        - projection_schema: Database schema definition
        - behavior: Projection behavior configuration

    Extension fields (for runtime implementations):
        - partial_updates: Specialized update operations for subsets of columns

Example:
    Load and validate a projector contract::

        from pathlib import Path
        import yaml
        from omnibase_core.models.projectors import ModelProjectorContract

        contract_path = Path(__file__).parent / "registration_projector.yaml"
        with open(contract_path) as f:
            data = yaml.safe_load(f)

        # Core fields validate against ModelProjectorContract
        # partial_updates is an extension handled by runtime

Related Tickets:
    - OMN-1170: Create registration_projector.yaml contract
    - OMN-1166: Projector contract models

.. versionadded:: 0.5.0
"""

from pathlib import Path

# Contract file paths for programmatic access
CONTRACTS_DIR = Path(__file__).parent

REGISTRATION_PROJECTOR_CONTRACT = CONTRACTS_DIR / "registration_projector.yaml"

__all__ = [
    "CONTRACTS_DIR",
    "REGISTRATION_PROJECTOR_CONTRACT",
]
