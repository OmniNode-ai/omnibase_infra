#!/usr/bin/env python3

# This file has been refactored - all models moved to separate files following ONEX standards:
# - model_consul_service_projection.py
# - model_consul_health_projection.py  
# - model_consul_kv_projection.py
# - model_consul_topology_projection.py
#
# Each model now uses strongly typed components instead of generic Dict[str, Any]
# All models follow one model per file pattern as per ONEX compliance