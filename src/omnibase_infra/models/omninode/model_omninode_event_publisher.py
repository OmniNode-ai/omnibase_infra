"""OmniNode Event Publisher for ModelEventEnvelope integration."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from omnibase_core.core.core_error_codes import CoreErrorCode
from omnibase_core.core.errors.onex_error import OnexError

# Import ModelEventEnvelope from omnibase_core
from omnibase_core.model.core.model_event_envelope import ModelEventEnvelope
from omnibase_core.model.core.model_onex_event import ModelOnexEvent
from omnibase_core.model.core.model_route_spec import ModelRouteSpec

from .model_omninode_topic_spec import ModelOmniNodeTopicSpec


class ModelOmniNodeEventPublisher(BaseModel):
    """
    Publisher for OmniNode events using ModelEventEnvelope from omnibase_core.
    
    Wraps PostgreSQL adapter operations in proper ModelEventEnvelope structure
    for publishing to RedPanda topics following OmniNode topic namespace design.
    """
    
    node_id: str = Field(
        default="postgres_adapter_node", 
        description="Node identifier for envelope source"
    )
    
    def create_postgres_query_completed_envelope(
        self, 
        correlation_id: UUID,
        query_data: Dict[str, Any],
        execution_time_ms: float,
        row_count: Optional[int] = None
    ) -> ModelEventEnvelope:
        """
        Create event envelope for PostgreSQL query completed.
        
        Args:
            correlation_id: Request correlation ID
            query_data: Query execution details
            execution_time_ms: Query execution time
            row_count: Number of rows affected/returned
            
        Returns:
            ModelEventEnvelope with PostgreSQL query completion event
        """
        # Create the ONEX event payload
        event_payload = ModelOnexEvent.create_core_event(
            event_type="core.database.query_completed",
            node_id=self.node_id,
            correlation_id=correlation_id,
            data={
                "database_type": "postgresql",
                "execution_time_ms": execution_time_ms,
                "row_count": row_count,
                "query_hash": query_data.get("query_hash"),
                "operation_type": query_data.get("operation_type", "query"),
                **query_data
            }
        )
        
        # Create topic spec for routing
        topic_spec = ModelOmniNodeTopicSpec.for_postgres_query_completed(str(correlation_id))
        
        # Create direct route to the topic
        route_spec = ModelRouteSpec.create_direct_route(topic_spec.to_topic_string())
        
        # Create and return the event envelope
        envelope = ModelEventEnvelope(
            payload=event_payload,
            route_spec=route_spec,
            source_node_id=self.node_id,
            correlation_id=correlation_id,
            metadata={
                "topic_spec": topic_spec.to_topic_string(),
                "database_operation": "query_completed",
                "omninode_namespace": f"{topic_spec.env}.{topic_spec.tenant}.{topic_spec.context}"
            }
        )
        
        # Add source hop to trace
        envelope.add_source_hop(self.node_id, "PostgreSQL Adapter")
        
        return envelope
    
    def create_postgres_query_failed_envelope(
        self,
        correlation_id: UUID, 
        error_message: str,
        query_data: Dict[str, Any],
        execution_time_ms: float
    ) -> ModelEventEnvelope:
        """
        Create event envelope for PostgreSQL query failure.
        
        Args:
            correlation_id: Request correlation ID
            error_message: Error description
            query_data: Query execution details
            execution_time_ms: Query execution time
            
        Returns:
            ModelEventEnvelope with PostgreSQL query failure event
        """
        # Create the ONEX event payload  
        event_payload = ModelOnexEvent.create_core_event(
            event_type="core.database.query_failed", 
            node_id=self.node_id,
            correlation_id=correlation_id,
            data={
                "database_type": "postgresql",
                "error_message": error_message,
                "execution_time_ms": execution_time_ms,
                "query_hash": query_data.get("query_hash"),
                "operation_type": query_data.get("operation_type", "query"),
                **query_data
            }
        )
        
        # Create topic spec for routing
        topic_spec = ModelOmniNodeTopicSpec.for_postgres_query_failed(str(correlation_id))
        
        # Create direct route to the topic
        route_spec = ModelRouteSpec.create_direct_route(topic_spec.to_topic_string()) 
        
        # Create and return the event envelope
        envelope = ModelEventEnvelope(
            payload=event_payload,
            route_spec=route_spec,
            source_node_id=self.node_id,
            correlation_id=correlation_id,
            metadata={
                "topic_spec": topic_spec.to_topic_string(),
                "database_operation": "query_failed",
                "omninode_namespace": f"{topic_spec.env}.{topic_spec.tenant}.{topic_spec.context}"
            }
        )
        
        # Add source hop to trace
        envelope.add_source_hop(self.node_id, "PostgreSQL Adapter")
        
        return envelope
    
    def create_postgres_health_response_envelope(
        self,
        correlation_id: UUID,
        health_status: str,
        health_data: Dict[str, Any]
    ) -> ModelEventEnvelope:
        """
        Create event envelope for PostgreSQL health check response.
        
        Args:
            correlation_id: Request correlation ID
            health_status: Health check status
            health_data: Health check details
            
        Returns:
            ModelEventEnvelope with PostgreSQL health response event
        """
        # Create the ONEX event payload
        event_payload = ModelOnexEvent.create_core_event(
            event_type="core.database.health_check_response",
            node_id=self.node_id,
            correlation_id=correlation_id,
            data={
                "database_type": "postgresql",
                "health_status": health_status,
                **health_data
            }
        )
        
        # Create topic spec for routing
        topic_spec = ModelOmniNodeTopicSpec.for_postgres_health_check()
        
        # Create direct route to the topic
        route_spec = ModelRouteSpec.create_direct_route(topic_spec.to_topic_string())
        
        # Create and return the event envelope
        envelope = ModelEventEnvelope(
            payload=event_payload,
            route_spec=route_spec,
            source_node_id=self.node_id,
            correlation_id=correlation_id,
            metadata={
                "topic_spec": topic_spec.to_topic_string(),
                "database_operation": "health_check",
                "omninode_namespace": f"{topic_spec.env}.{topic_spec.tenant}.{topic_spec.context}"
            }
        )
        
        # Add source hop to trace
        envelope.add_source_hop(self.node_id, "PostgreSQL Adapter")
        
        return envelope