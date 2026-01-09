# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pydantic models for compute plugins.

This module exports the core data models used by compute plugins:
- ModelPluginInputData: Base input data model for plugins
- ModelPluginContext: Execution context model for plugins
- ModelPluginOutputData: Base output data model for plugins

These models replace TypedDict definitions to conform to ONEX coding standards.
"""

from omnibase_infra.plugins.models.model_plugin_context import ModelPluginContext
from omnibase_infra.plugins.models.model_plugin_input_data import ModelPluginInputData
from omnibase_infra.plugins.models.model_plugin_output_data import ModelPluginOutputData

__all__ = [
    "ModelPluginContext",
    "ModelPluginInputData",
    "ModelPluginOutputData",
]
