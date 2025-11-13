# OmniNode Bridge Planning

## Overview

Current planning documents, strategies, and validation summaries for active development efforts.

## Active Planning Documents

### 🔥 Current Active Projects

#### Code Generator Enhancement (NEW - Nov 2025)
- **[Mixin Enhancement Master Plan](./CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md)** ⭐ - Comprehensive 6-week plan to enhance code generator with omnibase_core mixin support
- **[Migration Tracking Dashboard](./MIGRATION_TRACKING.md)** - Real-time progress tracking for node migrations
- **[Node Migration Checklist Template](./NODE_MIGRATION_CHECKLIST_TEMPLATE.md)** - Step-by-step checklist for migrating nodes
- **[Quick Start Guide](../guides/MIXIN_ENHANCED_GENERATION_QUICKSTART.md)** 🚀 - Get started with mixin-enhanced generation in 10 minutes
- **[Mixin Quick Reference](../reference/MIXIN_QUICK_REFERENCE.md)** - Developer reference card for all 33 omnibase_core mixins

**Quick Links**:
- **Project Manager**: Start with [Master Plan](./CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md)
- **Developer (New Node)**: Start with [Quick Start Guide](../guides/MIXIN_ENHANCED_GENERATION_QUICKSTART.md)
- **Developer (Migration)**: Start with [Migration Checklist](./NODE_MIGRATION_CHECKLIST_TEMPLATE.md)

---

### Infrastructure & Environment
- **[Docker Environment](./DOCKER_ENVIRONMENT.md)** - Docker configuration and container strategy
- **[Environment Configuration](./ENVIRONMENT_CONFIGURATION.md)** - Environment variable management and configuration

### Integration & Messaging
- **[Kafka Topic Strategy](./KAFKA_TOPIC_STRATEGY.md)** - Event streaming topic design and partitioning

### System Design
- **[ONEX Generation System Plan](./ONEX_GENERATION_SYSTEM_PLAN.md)** - ONEX-compliant generation system architecture
- **[LLM Business Logic Generation Plan](./LLM_BUSINESS_LOGIC_GENERATION_PLAN.md)** - LLM-powered business logic generation strategy

### Bridge Node Implementation
- **[Bridge Node Implementation Plan](./BRIDGE_NODE_IMPLEMENTATION_PLAN.md)** - Bridge nodes (Orchestrator, Reducer, Registry) implementation
- **[Node Bridge Store Effect Implementation Plan](./NODE_BRIDGE_STORE_EFFECT_IMPLEMENTATION_PLAN.md)** - Store effect node implementation

### Quality & Validation
- **[CI/CD Pipeline Validation Summary](./CI_CD_PIPELINE_VALIDATION_SUMMARY.md)** - Continuous integration validation status
- **[EventBus Compliance Audit](./EVENTBUS_COMPLIANCE_AUDIT.md)** - Event bus compliance validation

## Document Lifecycle

Planning documents follow this lifecycle:

1. **Active Planning** (current location: `docs/planning/`)
   - Documents actively guiding current development
   - Regularly updated as implementation progresses

2. **Implementation Complete** → Move to relevant permanent location
   - Architecture plans → `docs/architecture/`
   - Deployment strategies → `docs/deployment/`
   - Developer guides → `docs/developers/`

3. **Historical Reference** → Archive to `to_remove/point_in_time_reports/`
   - Plans superseded by implementation
   - Point-in-time snapshots no longer current

## Related Documentation

- [Architecture](../architecture/)
- [Deployment](../deployment/)
- [Developers](../developers/)
- [Operations](../operations/)
