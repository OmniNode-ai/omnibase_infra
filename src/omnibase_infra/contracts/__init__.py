# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Packaged contracts and the contract readers (OMN-26359).

A regular package, so the readers import from a wheel that is itself on
``sys.path`` (zipimport does not resolve an implicit namespace directory).
"""
