# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Config overlay sources: one per deployment, never layered (OMN-19747).

The local-home half of model-config plan task B4, keyed by scope as the runtime
lane overlays plan section 3.5 requires. The store half is not in this package
yet; :func:`select_config_overlay_source` refuses a deployment that selects it.
"""
