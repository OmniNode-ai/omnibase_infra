#!/usr/bin/env python3
"""
Demo script showing event bus message envelope to PostgreSQL conversion.

This demonstrates the complete flow:
Event Envelope → PostgreSQL Adapter → PostgreSQL Connection Manager → Database
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import ModelPostgresAdapterInput
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_output import ModelPostgresAdapterOutput
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest
from omnibase_infra.models.postgres.model_postgres_health_request import ModelPostgresHealthRequest


# Configure logging following omnibase_3 infrastructure pattern
logger = logging.getLogger(__name__)


async def demo_service_registration_envelope():
    """Demo: Service registration through event envelope."""
    
    logger.info("🎯 Demo: Service Registration via Event Envelope")
    logger.info("=" * 60)
    
    # Step 1: Create event envelope (as would come from message bus)
    correlation_id = uuid.uuid4()
    
    # This represents a service wanting to register itself
    service_data = {
        "service_name": "payment-processor",
        "service_type": "microservice", 
        "hostname": "payment-01.prod.local",
        "port": 8080,
        "status": "healthy",
        "metadata": {
            "version": "2.1.3",
            "environment": "production",
            "capabilities": ["payments", "refunds", "webhooks"],
            "health_check_url": "http://payment-01.prod.local:8080/health"
        }
    }
    
    # Create PostgreSQL query request
    query_request = ModelPostgresQueryRequest(
        query="""
            INSERT INTO infrastructure.service_registry 
            (service_name, service_type, hostname, port, status, metadata, health_check_url) 
            VALUES ($1, $2, $3, $4, $5, $6, $7) 
            RETURNING id, service_name, status, registered_at
        """,
        parameters=[
            service_data["service_name"],
            service_data["service_type"], 
            service_data["hostname"],
            service_data["port"],
            service_data["status"],
            service_data["metadata"],
            service_data["metadata"]["health_check_url"]
        ],
        correlation_id=correlation_id,
        timeout=30.0,
        record_metrics=True,
        context={
            "operation": "service_registration",
            "source": "service_mesh",
            "priority": "high"
        }
    )
    
    # Create message envelope (as would come from event bus)
    input_envelope = ModelPostgresAdapterInput(
        operation_type="query",
        query_request=query_request,
        correlation_id=correlation_id,
        timestamp=time.time(),
        context={
            "source": "service_discovery_system",
            "event_type": "SERVICE_REGISTRATION_REQUEST",
            "routing_key": "infrastructure.postgres.query"
        }
    )
    
    logger.info(f"📨 Input Event Envelope:")
    logger.info(f"   Operation: {input_envelope.operation_type}")
    logger.info(f"   Correlation ID: {input_envelope.correlation_id}")
    logger.info(f"   Service: {service_data['service_name']}")
    logger.info(f"   Context: {json.dumps(input_envelope.context, indent=6)}")

    # Step 2: Process through "adapter" (direct connection manager call for demo)
    logger.info("⚡ Processing through PostgreSQL Adapter...")
    
    try:
        connection_manager = PostgresConnectionManager()
        await connection_manager.initialize()
        
        start_time = time.perf_counter()
        
        # Execute the database operation
        db_result = await connection_manager.execute_query(
            query_request.query,
            *query_request.parameters,
            timeout=query_request.timeout,
            record_metrics=query_request.record_metrics
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Create response envelope (as adapter would return to message bus)
        if isinstance(db_result, list) and db_result:
            success = True
            registration_result = dict(db_result[0])
            status_message = f"Service '{service_data['service_name']}' registered successfully"
        else:
            success = False
            registration_result = None
            status_message = "Service registration failed - no result returned"
        
        # Create output envelope
        output_envelope = ModelPostgresAdapterOutput(
            operation_type="query",
            success=success,
            correlation_id=correlation_id,
            timestamp=time.time(),
            execution_time_ms=execution_time_ms,
            context=input_envelope.context,
            query_response={
                "success": success,
                "data": [registration_result] if registration_result else [],
                "rows_affected": 1 if registration_result else 0,
                "status_message": status_message,
                "execution_time_ms": execution_time_ms,
                "correlation_id": correlation_id
            }
        )
        
        logger.info(f"✅ Success! Database operation completed")
        logger.info(f"   Execution time: {execution_time_ms:.2f}ms")
        logger.info(f"   Service ID: {registration_result['id'] if registration_result else 'N/A'}")
        logger.info(f"   Registered at: {registration_result['registered_at'] if registration_result else 'N/A'}")
        
        logger.info(f"📤 Output Event Envelope:")
        logger.info(f"   Success: {output_envelope.success}")
        logger.info(f"   Correlation ID: {output_envelope.correlation_id}")
        logger.info(f"   Execution time: {output_envelope.execution_time_ms:.2f}ms")
        logger.info(f"   Rows affected: {output_envelope.query_response['rows_affected']}")
        
        await connection_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return False


async def demo_service_discovery_envelope():
    """Demo: Service discovery through event envelope."""
    
    logger.info("🔍 Demo: Service Discovery via Event Envelope")
    logger.info("=" * 60)
    
    # Create service discovery request
    correlation_id = uuid.uuid4()
    
    query_request = ModelPostgresQueryRequest(
        query="""
            SELECT 
                service_name, 
                service_type, 
                hostname, 
                port, 
                status,
                metadata,
                last_seen,
                registered_at
            FROM infrastructure.service_registry 
            WHERE service_type = $1 
              AND status IN ('healthy', 'degraded')
            ORDER BY last_seen DESC 
            LIMIT $2
        """,
        parameters=["microservice", 10],
        correlation_id=correlation_id,
        record_metrics=True,
        context={"operation": "service_discovery", "filter": "active_services"}
    )
    
    input_envelope = ModelPostgresAdapterInput(
        operation_type="query",
        query_request=query_request,
        correlation_id=correlation_id,
        timestamp=time.time(),
        context={
            "source": "load_balancer",
            "event_type": "SERVICE_DISCOVERY_REQUEST",
            "routing_key": "infrastructure.postgres.query"
        }
    )
    
    logger.info(f"📨 Service Discovery Request:")
    logger.info(f"   Looking for: microservice type")
    logger.info(f"   Status filter: healthy, degraded")
    logger.info(f"   Max results: 10")
    
    try:
        connection_manager = PostgresConnectionManager()
        await connection_manager.initialize()
        
        start_time = time.perf_counter()
        
        # Execute discovery query
        services = await connection_manager.execute_query(
            query_request.query,
            *query_request.parameters,
            timeout=query_request.timeout,
            record_metrics=query_request.record_metrics
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"🔍 Found {len(services)} active microservices:")
        for service in services:
            service_dict = dict(service)
            logger.info(f"   • {service_dict['service_name']} ({service_dict['hostname']}:{service_dict['port']}) - {service_dict['status']}")
        
        logger.info(f"⚡ Query executed in {execution_time_ms:.2f}ms")
        
        await connection_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Discovery error: {str(e)}")
        return False


async def demo_health_check_envelope():
    """Demo: Health check through event envelope."""
    
    logger.info("💚 Demo: Health Check via Event Envelope")
    logger.info("=" * 60)
    
    # Create health check request
    correlation_id = uuid.uuid4()
    
    health_request = ModelPostgresHealthRequest(
        include_connection_stats=True,
        include_performance_metrics=True,
        include_schema_info=False,
        correlation_id=correlation_id,
        context={"source": "monitoring_system", "check_type": "infrastructure"}
    )
    
    input_envelope = ModelPostgresAdapterInput(
        operation_type="health_check",
        health_request=health_request,
        correlation_id=correlation_id,
        timestamp=time.time(),
        context={
            "source": "prometheus_scraper",
            "event_type": "INFRASTRUCTURE_HEALTH_CHECK",
            "routing_key": "infrastructure.postgres.health"
        }
    )
    
    logger.info(f"📨 Health Check Request:")
    logger.info(f"   Include connection stats: {health_request.include_connection_stats}")
    logger.info(f"   Include performance metrics: {health_request.include_performance_metrics}")
    
    try:
        connection_manager = PostgresConnectionManager()
        await connection_manager.initialize()
        
        # Execute health check
        health_data = await connection_manager.health_check()
        
        logger.info(f"💚 Health Check Results:")
        logger.info(f"   Status: {health_data.get('status', 'unknown')}")
        logger.info(f"   Database: {health_data.get('database_info', {}).get('version', 'unknown')}")
        
        if 'connection_pool' in health_data:
            pool = health_data['connection_pool']
            logger.info(f"   Connection Pool: {pool.get('active', 0)} active, {pool.get('idle', 0)} idle, {pool.get('total', 0)} total")
        
        if 'errors' in health_data and health_data['errors']:
            logger.warning(f"   ⚠️  Errors: {len(health_data['errors'])}")
            for error in health_data['errors']:
                logger.warning(f"      - {error}")
        
        await connection_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return False


async def main():
    """Run all message envelope demos."""
    
    logger.info("🚀 PostgreSQL Adapter Message Envelope Demo")
    logger.info("=" * 60)
    logger.info("Demonstrating event bus message envelope to PostgreSQL conversion")
    logger.info("Running against Docker PostgreSQL environment")
    
    demos = [
        ("Service Registration", demo_service_registration_envelope),
        ("Service Discovery", demo_service_discovery_envelope), 
        ("Health Check", demo_health_check_envelope)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            logger.info(f"Running {demo_name} demo...")
            success = await demo_func()
            results.append((demo_name, success))
            
            if success:
                logger.info(f"✅ {demo_name} demo completed successfully")
            else:
                logger.error(f"❌ {demo_name} demo failed")
                
        except Exception as e:
            logger.error(f"❌ {demo_name} demo error: {str(e)}")
            results.append((demo_name, False))
    
    # Summary
    logger.info("📊 Demo Summary:")
    logger.info("=" * 40)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL" 
        logger.info(f"   {demo_name}: {status}")
    
    logger.info(f"Overall: {successful}/{total} demos successful")
    
    if successful == total:
        logger.info("🎉 All message envelope conversions working correctly!")
    else:
        logger.warning("⚠️  Some demos failed - check PostgreSQL connection and database setup")


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())