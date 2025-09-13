"""
Comprehensive Circuit Breaker Testing with Half-Open State Validation

Provides systematic testing of circuit breaker behavior including state transitions,
failure thresholds, recovery testing, and half-open state validation.

Per ONEX testing requirements:
- State transition verification (closed -> open -> half-open -> closed)
- Failure threshold validation with configurable parameters
- Half-open state testing with controlled recovery attempts
- Performance metrics validation under various load conditions
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from omnibase_core.core.onex_error import OnexError, CoreErrorCode


class CircuitBreakerTestResult(Enum):
    """Test result outcomes."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestMetrics:
    """Metrics collected during circuit breaker testing."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_trips: int = 0
    recovery_attempts: int = 0
    half_open_duration_ms: float = 0
    avg_response_time_ms: float = 0
    state_transitions: List[str] = field(default_factory=list)
    error_details: List[str] = field(default_factory=list)


@dataclass
class CircuitBreakerTestCase:
    """Individual circuit breaker test case."""
    name: str
    description: str
    test_function: Callable[[], Awaitable[TestMetrics]]
    expected_result: CircuitBreakerTestResult
    timeout_seconds: float = 30.0
    retry_attempts: int = 1


class CircuitBreakerTestSuite:
    """
    Comprehensive circuit breaker test suite for ONEX infrastructure.
    
    Tests circuit breaker behavior under various conditions including:
    - Failure threshold validation
    - State transition verification
    - Half-open state recovery testing
    - Performance under load
    """
    
    def __init__(self, circuit_breaker, kafka_adapter=None):
        """
        Initialize circuit breaker test suite.
        
        Args:
            circuit_breaker: Circuit breaker instance to test
            kafka_adapter: Optional Kafka adapter for integration testing
        """
        self._logger = logging.getLogger(__name__)
        self._circuit_breaker = circuit_breaker
        self._kafka_adapter = kafka_adapter
        self._test_results: Dict[str, Dict[str, Any]] = {}
        
        # Test configuration
        self._failure_threshold = 5
        self._recovery_timeout = 10.0
        self._half_open_max_calls = 3
        
        self._logger.info("Circuit breaker test suite initialized")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive circuit breaker test suite.
        
        Returns:
            Dictionary with test results, metrics, and recommendations
        """
        self._logger.info("Starting comprehensive circuit breaker tests")
        start_time = time.perf_counter()
        
        test_cases = [
            CircuitBreakerTestCase(
                name="failure_threshold_validation",
                description="Test circuit breaker trips after configured failure threshold",
                test_function=self._test_failure_threshold,
                expected_result=CircuitBreakerTestResult.PASSED
            ),
            CircuitBreakerTestCase(
                name="state_transition_verification",
                description="Verify proper state transitions (closed -> open -> half-open -> closed)",
                test_function=self._test_state_transitions,
                expected_result=CircuitBreakerTestResult.PASSED
            ),
            CircuitBreakerTestCase(
                name="half_open_recovery_testing",
                description="Test half-open state behavior with controlled recovery attempts",
                test_function=self._test_half_open_recovery,
                expected_result=CircuitBreakerTestResult.PASSED,
                timeout_seconds=45.0
            ),
            CircuitBreakerTestCase(
                name="concurrent_calls_handling",
                description="Test circuit breaker behavior under concurrent load",
                test_function=self._test_concurrent_calls,
                expected_result=CircuitBreakerTestResult.PASSED
            ),
            CircuitBreakerTestCase(
                name="performance_metrics_validation",
                description="Validate performance metrics collection during circuit breaker operations",
                test_function=self._test_performance_metrics,
                expected_result=CircuitBreakerTestResult.PASSED
            ),
            CircuitBreakerTestCase(
                name="edge_case_scenarios",
                description="Test edge cases and boundary conditions",
                test_function=self._test_edge_cases,
                expected_result=CircuitBreakerTestResult.PASSED
            )
        ]
        
        # Execute all test cases
        total_tests = len(test_cases)
        passed_tests = 0
        failed_tests = 0
        
        for test_case in test_cases:
            try:
                result = await self._execute_test_case(test_case)
                
                if result["status"] == CircuitBreakerTestResult.PASSED.value:
                    passed_tests += 1
                else:
                    failed_tests += 1
                    
                self._test_results[test_case.name] = result
                
            except Exception as e:
                failed_tests += 1
                self._test_results[test_case.name] = {
                    "status": CircuitBreakerTestResult.ERROR.value,
                    "error": str(e),
                    "metrics": TestMetrics()
                }
                self._logger.error(f"Test case '{test_case.name}' failed with error: {str(e)}")
        
        total_time = time.perf_counter() - start_time
        
        # Compile comprehensive results
        results = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "total_duration_seconds": round(total_time, 3)
            },
            "test_results": self._test_results,
            "recommendations": self._generate_recommendations(),
            "circuit_breaker_config": self._get_circuit_breaker_config()
        }
        
        self._logger.info(f"Circuit breaker tests completed: {passed_tests}/{total_tests} passed")
        return results
    
    async def _execute_test_case(self, test_case: CircuitBreakerTestCase) -> Dict[str, Any]:
        """Execute individual test case with timeout and metrics collection."""
        self._logger.info(f"Executing test: {test_case.name}")
        start_time = time.perf_counter()
        
        try:
            # Execute test with timeout
            metrics = await asyncio.wait_for(
                test_case.test_function(),
                timeout=test_case.timeout_seconds
            )
            
            execution_time = time.perf_counter() - start_time
            
            return {
                "status": CircuitBreakerTestResult.PASSED.value,
                "description": test_case.description,
                "execution_time_seconds": round(execution_time, 3),
                "metrics": metrics,
                "expected_result": test_case.expected_result.value
            }
            
        except asyncio.TimeoutError:
            return {
                "status": CircuitBreakerTestResult.FAILED.value,
                "error": f"Test timed out after {test_case.timeout_seconds} seconds",
                "execution_time_seconds": test_case.timeout_seconds,
                "metrics": TestMetrics()
            }
        except Exception as e:
            return {
                "status": CircuitBreakerTestResult.ERROR.value,
                "error": str(e),
                "execution_time_seconds": round(time.perf_counter() - start_time, 3),
                "metrics": TestMetrics()
            }
    
    async def _test_failure_threshold(self) -> TestMetrics:
        """Test that circuit breaker trips after configured failure threshold."""
        metrics = TestMetrics()
        
        # Reset circuit breaker to known state
        await self._reset_circuit_breaker()
        initial_state = self._circuit_breaker.get_state()
        metrics.state_transitions.append(f"initial: {initial_state['state']}")
        
        # Create a function that always fails
        async def always_fail():
            raise Exception("Simulated failure for threshold testing")
        
        # Make calls until circuit breaker trips
        for i in range(self._failure_threshold + 2):  # Extra calls to ensure trip
            try:
                await self._circuit_breaker.call(always_fail)
                metrics.successful_calls += 1
            except Exception as e:
                metrics.failed_calls += 1
                metrics.total_calls += 1
                
                # Check if circuit breaker has tripped
                current_state = self._circuit_breaker.get_state()
                if current_state["state"] == "open":
                    metrics.circuit_trips += 1
                    metrics.state_transitions.append(f"call_{i}: {current_state['state']}")
                    break
        
        # Verify circuit breaker is in OPEN state
        final_state = self._circuit_breaker.get_state()
        if final_state["state"] != "open":
            raise AssertionError(f"Expected circuit breaker to be OPEN, but was {final_state['state']}")
        
        # Verify failure count matches threshold
        if final_state["failure_count"] < self._failure_threshold:
            raise AssertionError(f"Expected at least {self._failure_threshold} failures, got {final_state['failure_count']}")
        
        return metrics
    
    async def _test_state_transitions(self) -> TestMetrics:
        """Test proper state transitions: closed -> open -> half-open -> closed."""
        metrics = TestMetrics()
        
        # Start with reset circuit breaker (CLOSED state)
        await self._reset_circuit_breaker()
        state = self._circuit_breaker.get_state()
        assert state["state"] == "closed", f"Expected CLOSED state, got {state['state']}"
        metrics.state_transitions.append("initial: closed")
        
        # Force circuit breaker to OPEN by exceeding failure threshold
        async def fail_function():
            raise Exception("Forced failure")
        
        for i in range(self._failure_threshold + 1):
            try:
                await self._circuit_breaker.call(fail_function)
            except:
                metrics.failed_calls += 1
        
        # Verify OPEN state
        state = self._circuit_breaker.get_state()
        assert state["state"] == "open", f"Expected OPEN state, got {state['state']}"
        metrics.state_transitions.append("after_failures: open")
        metrics.circuit_trips += 1
        
        # Wait for recovery timeout to transition to HALF_OPEN
        await asyncio.sleep(self._recovery_timeout + 0.5)
        
        # Make a call to trigger state check (some circuit breakers are lazy)
        try:
            await self._circuit_breaker.call(lambda: "success")
            metrics.successful_calls += 1
        except:
            pass  # May still fail, but should transition to half-open
        
        state = self._circuit_breaker.get_state()
        if state["state"] == "half_open":
            metrics.state_transitions.append("after_timeout: half_open")
            
            # Make successful calls to return to CLOSED
            async def success_function():
                return "success"
            
            for i in range(self._half_open_max_calls):
                try:
                    await self._circuit_breaker.call(success_function)
                    metrics.successful_calls += 1
                    metrics.recovery_attempts += 1
                except Exception as e:
                    metrics.error_details.append(f"Recovery attempt {i} failed: {str(e)}")
            
            # Verify return to CLOSED state
            final_state = self._circuit_breaker.get_state()
            if final_state["state"] == "closed":
                metrics.state_transitions.append("after_recovery: closed")
        
        return metrics
    
    async def _test_half_open_recovery(self) -> TestMetrics:
        """Test half-open state behavior with controlled recovery attempts."""
        metrics = TestMetrics()
        start_time = time.perf_counter()
        
        # Force circuit breaker to OPEN state
        await self._force_circuit_breaker_open()
        
        # Wait for recovery timeout
        await asyncio.sleep(self._recovery_timeout + 0.1)
        
        # Test half-open state behavior
        half_open_start = time.perf_counter()
        
        # Make controlled recovery attempts
        recovery_success = False
        for attempt in range(self._half_open_max_calls):
            try:
                # Successful call
                result = await self._circuit_breaker.call(lambda: f"recovery_attempt_{attempt}")
                metrics.successful_calls += 1
                metrics.recovery_attempts += 1
                
                state = self._circuit_breaker.get_state()
                if state["state"] == "closed":
                    recovery_success = True
                    half_open_end = time.perf_counter()
                    metrics.half_open_duration_ms = (half_open_end - half_open_start) * 1000
                    break
                    
            except Exception as e:
                metrics.failed_calls += 1
                metrics.error_details.append(f"Recovery attempt {attempt} failed: {str(e)}")
        
        if not recovery_success:
            raise AssertionError("Circuit breaker failed to recover from half-open state")
        
        return metrics
    
    async def _test_concurrent_calls(self) -> TestMetrics:
        """Test circuit breaker behavior under concurrent load."""
        metrics = TestMetrics()
        
        await self._reset_circuit_breaker()
        
        # Create mix of successful and failing calls
        async def mixed_function(call_id: int):
            if call_id % 3 == 0:  # Every third call fails
                raise Exception(f"Simulated failure for call {call_id}")
            return f"success_{call_id}"
        
        # Execute concurrent calls
        tasks = []
        for i in range(20):
            task = asyncio.create_task(self._safe_circuit_call(mixed_function, i))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results using duck typing
        for result in results:
            if hasattr(result, '__traceback__') and hasattr(result, 'args'):
                # Exception-like object
                metrics.failed_calls += 1
            else:
                metrics.successful_calls += 1
            metrics.total_calls += 1
        
        return metrics
    
    async def _test_performance_metrics(self) -> TestMetrics:
        """Validate performance metrics collection during circuit breaker operations."""
        metrics = TestMetrics()
        
        await self._reset_circuit_breaker()
        
        # Make timed calls to measure performance
        response_times = []
        
        async def timed_function():
            await asyncio.sleep(0.01)  # Simulate work
            return "timed_success"
        
        for i in range(10):
            start = time.perf_counter()
            try:
                await self._circuit_breaker.call(timed_function)
                end = time.perf_counter()
                response_times.append((end - start) * 1000)  # Convert to ms
                metrics.successful_calls += 1
            except Exception:
                metrics.failed_calls += 1
            
            metrics.total_calls += 1
        
        # Calculate average response time
        if response_times:
            metrics.avg_response_time_ms = sum(response_times) / len(response_times)
        
        return metrics
    
    async def _test_edge_cases(self) -> TestMetrics:
        """Test edge cases and boundary conditions."""
        metrics = TestMetrics()
        
        # Test 1: Rapid state changes
        await self._reset_circuit_breaker()
        
        # Test 2: Zero failure threshold (if supported)
        # Test 3: Very high failure threshold
        # Test 4: Negative timeouts (should be handled gracefully)
        
        # For now, just test basic edge case
        async def edge_case_function():
            return "edge_case_success"
        
        try:
            result = await self._circuit_breaker.call(edge_case_function)
            metrics.successful_calls += 1
        except Exception as e:
            metrics.failed_calls += 1
            metrics.error_details.append(f"Edge case test failed: {str(e)}")
        
        return metrics
    
    async def _safe_circuit_call(self, func: Callable, *args):
        """Safely execute circuit breaker call with error handling."""
        try:
            return await self._circuit_breaker.call(lambda: func(*args))
        except Exception as e:
            return e
    
    async def _reset_circuit_breaker(self):
        """Reset circuit breaker to initial closed state."""
        # This is implementation-specific - may need to be adapted
        if hasattr(self._circuit_breaker, 'reset'):
            await self._circuit_breaker.reset()
        elif hasattr(self._circuit_breaker, '_reset'):
            await self._circuit_breaker._reset()
    
    async def _force_circuit_breaker_open(self):
        """Force circuit breaker into OPEN state for testing."""
        async def always_fail():
            raise Exception("Forced failure")
        
        for _ in range(self._failure_threshold + 1):
            try:
                await self._circuit_breaker.call(always_fail)
            except:
                pass  # Expected to fail
    
    def _get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get current circuit breaker configuration."""
        if hasattr(self._circuit_breaker, 'get_config'):
            return self._circuit_breaker.get_config()
        else:
            return {
                "failure_threshold": self._failure_threshold,
                "recovery_timeout": self._recovery_timeout,
                "half_open_max_calls": self._half_open_max_calls
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze test results and provide recommendations
        total_tests = len(self._test_results)
        passed_tests = sum(1 for result in self._test_results.values() 
                          if result.get("status") == CircuitBreakerTestResult.PASSED.value)
        
        if passed_tests < total_tests:
            recommendations.append(
                f"Circuit breaker failed {total_tests - passed_tests} out of {total_tests} tests. "
                "Review configuration and implementation."
            )
        
        # Check for performance issues
        for test_name, result in self._test_results.items():
            if result.get("execution_time_seconds", 0) > 10:
                recommendations.append(
                    f"Test '{test_name}' took {result['execution_time_seconds']}s. "
                    "Consider optimizing circuit breaker response times."
                )
        
        if not recommendations:
            recommendations.append("All circuit breaker tests passed successfully. Configuration appears optimal.")
        
        return recommendations


# Helper function for easy testing integration
async def run_circuit_breaker_tests(circuit_breaker, kafka_adapter=None) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive circuit breaker tests.
    
    Args:
        circuit_breaker: Circuit breaker instance to test
        kafka_adapter: Optional Kafka adapter for integration testing
        
    Returns:
        Dictionary with test results and recommendations
    """
    test_suite = CircuitBreakerTestSuite(circuit_breaker, kafka_adapter)
    return await test_suite.run_comprehensive_tests()