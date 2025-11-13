# ONEX Documentation - Local Reference

This directory contains local copies of ONEX documentation from the Archon project for reference during omninode_bridge implementation.

## Files

- **ONEX_GUIDE.md** - Comprehensive ONEX implementation guide
  - 3-tier base class system (Minimal, Standard, Full)
  - Contract and subcontract composition
  - Directory structure and naming conventions
  - Best practices and anti-patterns

- **ONEX_QUICK_REFERENCE.md** - Quick reference for patterns and templates
  - 4-node architecture overview
  - Node type quick reference
  - Quick start templates
  - Decision trees
  - Common imports

- **SHARED_RESOURCE_VERSIONING.md** - Versioning strategy for shared resources
  - Independent versioning with shared/ directory
  - Major version only (v1, v2, v3)
  - Lazy promotion strategy
  - Gradual migration

## Source

These files are copied from the Archon repository `docs/onex/` as of 2025-10-02.

For the most up-to-date documentation, refer to the original location in the Archon project.

## Usage in omninode_bridge

Refer to these documents when:
- Implementing NodeBridgeOrchestrator and NodeBridgeReducer
- Understanding contract/subcontract composition
- Making architectural decisions
- Following ONEX naming conventions
- Structuring directories and files

## Primary Implementation Plan

**Use BRIDGE_NODE_IMPLEMENTATION_PLAN.md** in the project root for the focused implementation plan specific to omninode_bridge's core nodes.
