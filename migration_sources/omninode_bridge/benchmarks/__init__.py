"""Performance benchmarking suite for omninode_bridge.

This module contains comprehensive performance benchmarks for all components:
- Event infrastructure (event_log_insertion.py, dashboard_queries.py)
- CRUD operations (workflow_crud.py, bridge_state_crud.py)
- Kafka integration (kafka_producer.py)
- Orchestrator workflows (orchestrator_workflow.py)
- Reducer aggregation (reducer_aggregation.py)
- Load testing (load_test.py)

All benchmarks target the performance thresholds defined in:
- docs/ROADMAP.md
- migrations/EVENT_LOGS_DESIGN_RATIONALE.md
- migrations/BRIDGE_STATE_DESIGN_RATIONALE.md
"""
