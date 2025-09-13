"""
Load Testing Framework for PostgreSQL Adapter RedPanda Integration.

Uses Locust for high-volume load testing of the PostgreSQL adapter with
event publishing to validate performance under sustained load conditions.

Usage:
    # Install locust first: pip install locust
    # Run load test: locust -f postgres_adapter_load_test.py --host=http://localhost:8085
    # Web UI available at: http://localhost:8089
"""

import asyncio
import json
import random
import time
import uuid
from typing import Dict, Any, List, Optional

import locust
from locust import HttpUser, task, between
from locust.exception import InterruptTaskSet

# Import ONEX models for proper request formatting
from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.models.model_postgres_adapter_input import ModelPostgresAdapterInput
from omnibase_infra.models.postgres.model_postgres_query_request import ModelPostgresQueryRequest


class PostgresAdapterLoadTestUser(HttpUser):
    """Load testing user for PostgreSQL adapter operations."""
    
    wait_time = between(0.1, 2.0)  # Wait between 100ms to 2s between tasks
    
    def on_start(self):
        """Initialize load test user session."""
        self.correlation_ids = []
        self.test_scenarios = self._generate_test_scenarios()
        
        # Health check to ensure service is available
        try:
            response = self.client.get("/health", timeout=10)
            if response.status_code != 200:
                raise InterruptTaskSet(exception=Exception(f"Service health check failed: {response.status_code}"))
        except Exception as e:
            raise InterruptTaskSet(exception=Exception(f"Cannot connect to service: {e}"))
    
    def _generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate diverse test scenarios for load testing."""
        scenarios = []
        
        # Scenario 1: Simple SELECT queries
        for i in range(10):
            scenarios.append({
                "name": f"simple_select_{i}",
                "query": f"SELECT {i} as test_value, NOW() as timestamp",
                "parameters": [],
                "expected_load": "low"
            })
        
        # Scenario 2: Parameterized queries
        for i in range(10):
            scenarios.append({
                "name": f"parameterized_query_{i}",
                "query": "SELECT $1 as user_id, $2 as action, $3 as timestamp",
                "parameters": [random.randint(1, 1000), f"action_{i}", time.time()],
                "expected_load": "medium"
            })
        
        # Scenario 3: Complex analytical queries
        for i in range(5):
            scenarios.append({
                "name": f"analytical_query_{i}",
                "query": """
                    WITH RECURSIVE series AS (
                        SELECT 1 as n
                        UNION ALL
                        SELECT n + 1 FROM series WHERE n < $1
                    )
                    SELECT COUNT(*) as total, AVG(n) as average FROM series
                """,
                "parameters": [random.randint(10, 100)],
                "expected_load": "high"
            })
        
        # Scenario 4: JSON operations
        for i in range(5):
            scenarios.append({
                "name": f"json_query_{i}",
                "query": "SELECT $1::jsonb as metadata, jsonb_array_length($1::jsonb->'items') as item_count",
                "parameters": [json.dumps({
                    "items": [f"item_{j}" for j in range(random.randint(1, 20))],
                    "user_id": random.randint(1, 1000),
                    "timestamp": time.time()
                })],
                "expected_load": "medium"
            })
        
        return scenarios
    
    def _create_query_request(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create properly formatted query request for the adapter."""
        correlation_id = str(uuid.uuid4())
        self.correlation_ids.append(correlation_id)
        
        # Create request matching ModelPostgresAdapterInput structure
        request_data = {
            "operation_type": "query",
            "correlation_id": correlation_id,
            "context": {
                "load_test": True,
                "scenario": scenario["name"],
                "expected_load": scenario["expected_load"]
            },
            "query_request": {
                "query": scenario["query"],
                "parameters": scenario["parameters"],
                "correlation_id": correlation_id,
                "record_metrics": True,
                "timeout": 30.0,
                "context": {
                    "load_test_scenario": scenario["name"]
                }
            }
        }
        
        return request_data
    
    @task(weight=10)
    def execute_simple_query(self):
        """Execute simple SELECT queries (most common operation)."""
        scenario = random.choice([s for s in self.test_scenarios if s["expected_load"] == "low"])
        request_data = self._create_query_request(scenario)
        
        with self.client.post(
            "/process",
            json=request_data,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="simple_query"
        ) as response:
            self._validate_response(response, scenario, "simple_query")
    
    @task(weight=5)
    def execute_parameterized_query(self):
        """Execute parameterized queries (medium complexity)."""
        scenario = random.choice([s for s in self.test_scenarios if s["expected_load"] == "medium"])
        request_data = self._create_query_request(scenario)
        
        with self.client.post(
            "/process", 
            json=request_data,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="parameterized_query"
        ) as response:
            self._validate_response(response, scenario, "parameterized_query")
    
    @task(weight=2)
    def execute_analytical_query(self):
        """Execute analytical queries (high complexity)."""
        scenario = random.choice([s for s in self.test_scenarios if s["expected_load"] == "high"])
        request_data = self._create_query_request(scenario)
        
        with self.client.post(
            "/process",
            json=request_data, 
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="analytical_query"
        ) as response:
            self._validate_response(response, scenario, "analytical_query")
    
    @task(weight=3)
    def execute_json_query(self):
        """Execute JSON-based queries (medium complexity)."""
        scenario = random.choice([s for s in self.test_scenarios if "json" in s["name"]])
        request_data = self._create_query_request(scenario)
        
        with self.client.post(
            "/process",
            json=request_data,
            headers={"Content-Type": "application/json"}, 
            catch_response=True,
            name="json_query"
        ) as response:
            self._validate_response(response, scenario, "json_query")
    
    @task(weight=1)
    def execute_health_check(self):
        """Execute health check operations."""
        correlation_id = str(uuid.uuid4())
        request_data = {
            "operation_type": "health_check",
            "correlation_id": correlation_id,
            "context": {
                "load_test": True,
                "check_type": "load_test_health"
            }
        }
        
        with self.client.post(
            "/process",
            json=request_data,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="health_check"
        ) as response:
            self._validate_health_response(response, correlation_id)
    
    def _validate_response(self, response, scenario: Dict[str, Any], operation_name: str):
        """Validate adapter response and record metrics."""
        try:
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                if not result.get("success"):
                    response.failure(f"Query failed: {result.get('error_message', 'Unknown error')}")
                    return
                
                # Validate correlation ID
                if result.get("correlation_id") not in self.correlation_ids:
                    response.failure("Correlation ID mismatch")
                    return
                
                # Check execution time (performance validation)
                execution_time = result.get("execution_time_ms", 0)
                
                # Performance thresholds based on expected load
                max_execution_times = {
                    "low": 100,     # Simple queries under 100ms
                    "medium": 500,  # Medium queries under 500ms  
                    "high": 2000    # Complex queries under 2s
                }
                
                expected_load = scenario.get("expected_load", "medium")
                max_time = max_execution_times.get(expected_load, 500)
                
                if execution_time > max_time:
                    response.failure(f"Query too slow: {execution_time}ms > {max_time}ms")
                    return
                
                # Validate that event publishing was attempted
                if "query_response" in result:
                    query_response = result["query_response"]
                    if not query_response.get("success"):
                        response.failure(f"Database query failed: {query_response.get('error_message')}")
                        return
                
                response.success()
                
            elif response.status_code == 503:
                # Service temporarily unavailable (circuit breaker might be open)
                response.failure("Service unavailable (possible circuit breaker)")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
                
        except json.JSONDecodeError:
            response.failure("Invalid JSON response")
        except Exception as e:
            response.failure(f"Response validation error: {e}")
    
    def _validate_health_response(self, response, correlation_id: str):
        """Validate health check response."""
        try:
            if response.status_code == 200:
                result = response.json()
                
                if result.get("operation_type") != "health_check":
                    response.failure("Invalid health check response")
                    return
                
                if result.get("correlation_id") != correlation_id:
                    response.failure("Health check correlation ID mismatch")
                    return
                
                # Health checks should be fast
                execution_time = result.get("execution_time_ms", 0)
                if execution_time > 200:  # Health checks over 200ms are concerning
                    response.failure(f"Health check too slow: {execution_time}ms")
                    return
                
                response.success()
            else:
                response.failure(f"Health check failed with status: {response.status_code}")
                
        except Exception as e:
            response.failure(f"Health check validation error: {e}")


class PostgresAdapterStressTestUser(PostgresAdapterLoadTestUser):
    """Stress testing user with higher load and error injection."""
    
    wait_time = between(0.01, 0.1)  # Much shorter wait times
    
    def _generate_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate stress test scenarios including error conditions."""
        scenarios = super()._generate_test_scenarios()
        
        # Add error-inducing scenarios
        error_scenarios = [
            {
                "name": "syntax_error_query",
                "query": "SELECT * FORM non_existent_table",  # Intentional syntax error
                "parameters": [],
                "expected_load": "low",
                "expect_failure": True
            },
            {
                "name": "invalid_parameter_query", 
                "query": "SELECT $1::integer as num",
                "parameters": ["not_a_number"],  # Invalid parameter type
                "expected_load": "low",
                "expect_failure": True
            },
            {
                "name": "timeout_query",
                "query": "SELECT pg_sleep($1)",
                "parameters": [60],  # Long sleep to trigger timeout
                "expected_load": "high",
                "expect_failure": True
            }
        ]
        
        scenarios.extend(error_scenarios)
        return scenarios
    
    @task(weight=2)
    def execute_error_scenario(self):
        """Execute scenarios designed to trigger errors."""
        error_scenarios = [s for s in self.test_scenarios if s.get("expect_failure")]
        if not error_scenarios:
            return
            
        scenario = random.choice(error_scenarios)
        request_data = self._create_query_request(scenario)
        
        with self.client.post(
            "/process",
            json=request_data,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="error_scenario"
        ) as response:
            # For error scenarios, we expect failure
            if response.status_code == 200:
                result = response.json()
                if not result.get("success"):
                    response.success()  # Expected failure
                else:
                    response.failure("Expected query to fail but it succeeded")
            else:
                response.success()  # Any error status is acceptable for error scenarios


# Custom load test configuration classes
class PostgresAdapterLoadTest(locust.LoadTestShape):
    """Custom load test shape for PostgreSQL adapter testing."""
    
    stages = [
        {"duration": 60, "users": 1, "spawn_rate": 1},      # Warm up: 1 user for 60s
        {"duration": 180, "users": 10, "spawn_rate": 3},    # Ramp up: 10 users for 3 min
        {"duration": 300, "users": 25, "spawn_rate": 5},    # Steady: 25 users for 5 min
        {"duration": 420, "users": 50, "spawn_rate": 10},   # Peak: 50 users for 7 min
        {"duration": 480, "users": 10, "spawn_rate": -10},  # Ramp down: 10 users
        {"duration": 540, "users": 0, "spawn_rate": -5},    # Stop: 0 users
    ]
    
    def tick(self):
        """Define the load test progression."""
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        
        return None  # Test complete


if __name__ == "__main__":
    # Usage example
    import os
    import sys
    
    # Set environment variables for test configuration
    os.environ.setdefault("LOCUST_HOST", "http://localhost:8085")
    os.environ.setdefault("LOCUST_USERS", "25")
    os.environ.setdefault("LOCUST_SPAWN_RATE", "5")
    os.environ.setdefault("LOCUST_RUN_TIME", "300s")
    
    print("PostgreSQL Adapter Load Test")
    print("=" * 40)
    print(f"Target Host: {os.environ.get('LOCUST_HOST')}")
    print(f"Max Users: {os.environ.get('LOCUST_USERS')}")
    print(f"Spawn Rate: {os.environ.get('LOCUST_SPAWN_RATE')}/sec")
    print(f"Run Time: {os.environ.get('LOCUST_RUN_TIME')}")
    print()
    print("Run with: locust -f postgres_adapter_load_test.py")
    print("Web UI: http://localhost:8089")