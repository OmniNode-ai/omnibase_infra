#!/usr/bin/env python3
"""Check configuration values for NodeLLMEffect."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.llm_effect.v1_0_0 import NodeLLMEffect

# Load credentials
load_dotenv("/Volumes/PRO-G40/Code/omniclaude/.env")

print("Environment variables:")
print(
    f"  ZAI_API_KEY: {'SET' if os.getenv('ZAI_API_KEY') else 'NOT SET'} (length: {len(os.getenv('ZAI_API_KEY', ''))})"
)
print(f"  ZAI_ENDPOINT: {os.getenv('ZAI_ENDPOINT', 'NOT SET')}")

# Initialize node
container = ModelContainer(value={}, container_type="config")
node = NodeLLMEffect(container)

print("\nNode configuration:")
print(f"  zai_base_url: {node.config.zai_base_url}")
print(f"  zai_api_key: [REDACTED] (length: {len(node.config.zai_api_key)})")
print(f"  circuit_breaker_threshold: {node.config.circuit_breaker_threshold}")
print(f"  max_retry_attempts: {node.config.max_retry_attempts}")
print(f"  http_timeout_seconds: {node.config.http_timeout_seconds}")

print("\nTier models:")
for tier, model in node.tier_models.items():
    print(f"  {tier.value}: {model}")

print("\nContext windows:")
for tier, window in node.context_windows.items():
    print(f"  {tier.value}: {window} tokens")

print("\nCost per 1M tokens:")
print(
    f"  CLOUD_FAST input: ${node.cost_per_1m_input.get(node.tier_models.__class__.__dict__['CLOUD_FAST'], 'N/A')}"
)
print(
    f"  CLOUD_FAST output: ${node.cost_per_1m_output.get(node.tier_models.__class__.__dict__['CLOUD_FAST'], 'N/A')}"
)
