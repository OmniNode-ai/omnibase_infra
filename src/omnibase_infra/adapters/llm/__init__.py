# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""LLM protocol adapter implementations.

This package provides concrete implementations of the SPI LLM protocols
(ProtocolLLMProvider, ProtocolModelRouter, ProtocolLLMToolProvider) that
wrap the existing infra-layer handlers (HandlerLlmOpenaiCompatible,
HandlerLlmOllama) behind the protocol interfaces, enabling container-based
dependency injection for LLM services.

The package re-exports nothing: import each adapter from its own module.
Importing a submodule runs this ``__init__`` first, so an eager export list
made every such import (the Coding Plan predicate, from the endpoint health
config's validator) load every adapter and, through the similarity adapter, the
Qdrant client, inside the running event loop. The composed boot check measured
that as a 1.66 s loop hold.

Related Tickets:
    - OMN-2319: Implement SPI LLM protocol adapters
"""
