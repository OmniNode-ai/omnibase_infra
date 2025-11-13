#!/usr/bin/env python3
"""Unit tests for orchestrator main_standalone.py REST API wrapper.

Tests cover:
- FastAPI application initialization
- CORS middleware configuration
- Startup and shutdown events
- Health check endpoint
- Workflow submission endpoint
- Workflow status endpoint
- Metrics endpoint
- Root endpoint
- Error handling
- Environment variable configuration
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Import the module to test
from omninode_bridge.nodes.orchestrator.v1_0_0 import main_standalone


class TestMainStandaloneModule:
    """Test suite for main_standalone module components."""

    def test_global_node_instance_initially_none(self):
        """Test that global node_instance is initially None."""
        assert main_standalone.node_instance is None

    def test_workflow_submission_request_model(self):
        """Test WorkflowSubmissionRequest model validation."""
        # Test with all required fields
        request = main_standalone.WorkflowSubmissionRequest(
            content="test content", correlation_id="test-id", namespace="test.namespace"
        )
        assert request.content == "test content"
        assert request.correlation_id == "test-id"
        assert request.namespace == "test.namespace"

        # Test with default namespace
        request = main_standalone.WorkflowSubmissionRequest(content="test content")
        assert request.content == "test content"
        assert request.correlation_id is None
        assert request.namespace == "omninode.bridge"

        # Test validation error for missing content
        with pytest.raises(ValidationError):
            main_standalone.WorkflowSubmissionRequest()

    def test_workflow_submission_response_model(self):
        """Test WorkflowSubmissionResponse model validation."""
        response = main_standalone.WorkflowSubmissionResponse(
            success=True,
            workflow_id="test-id",
            state="processing",
            message="Test message",
        )
        assert response.success is True
        assert response.workflow_id == "test-id"
        assert response.state == "processing"
        assert response.message == "Test message"

        # Test without optional message
        response = main_standalone.WorkflowSubmissionResponse(
            success=False, workflow_id="test-id", state="failed"
        )
        assert response.success is False
        assert response.message is None

    def test_workflow_status_response_model(self):
        """Test WorkflowStatusResponse model validation."""
        response = main_standalone.WorkflowStatusResponse(
            workflow_id="test-id",
            state="completed",
            current_step="stamp_creation",
            result={"stamp_id": "stamp-123"},
        )
        assert response.workflow_id == "test-id"
        assert response.state == "completed"
        assert response.current_step == "stamp_creation"
        assert response.result == {"stamp_id": "stamp-123"}

        # Test without optional fields
        response = main_standalone.WorkflowStatusResponse(
            workflow_id="test-id", state="processing"
        )
        assert response.current_step is None
        assert response.result is None

    def test_health_response_model(self):
        """Test HealthResponse model validation."""
        response = main_standalone.HealthResponse(
            status="healthy", service="TestService", version="1.0.0", mode="standalone"
        )
        assert response.status == "healthy"
        assert response.service == "TestService"
        assert response.version == "1.0.0"
        assert response.mode == "standalone"


class TestFastAPIApplication:
    """Test suite for FastAPI application configuration."""

    def test_app_creation(self):
        """Test FastAPI application creation."""
        assert main_standalone.app is not None
        assert main_standalone.app.title == "NodeBridgeOrchestrator API (Standalone)"
        assert (
            main_standalone.app.description
            == "Standalone REST API for ONEX v2.0 workflow orchestration"
        )
        assert main_standalone.app.version == "1.0.0"

    def test_cors_middleware_configuration(self):
        """Test CORS middleware configuration."""
        # Check that CORS middleware is added
        cors_middleware = None
        for middleware in main_standalone.app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break

        assert cors_middleware is not None
        # Note: We can't easily inspect the middleware options without more complex setup

    def test_routes_registration(self):
        """Test that all routes are registered."""
        routes = [route.path for route in main_standalone.app.routes]
        assert "/health" in routes
        assert "/workflow/submit" in routes
        assert "/workflow/{workflow_id}/status" in routes
        assert "/metrics" in routes
        assert "/" in routes


class TestStartupEvent:
    """Test suite for application startup event."""

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.os.getenv")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.logger")
    async def test_startup_event_success(self, mock_logger, mock_getenv):
        """Test successful application startup in standalone mode."""
        # Setup environment variable mocks
        mock_getenv.side_effect = lambda key, default=None: {
            "METADATA_STAMPING_SERVICE_URL": "http://test-metadata:8053",
            "ONEXTREE_SERVICE_URL": "http://test-onextree:8080",
            "KAFKA_BOOTSTRAP_SERVERS": "test-kafka:9092",
            "DEFAULT_NAMESPACE": "test.namespace",
            "API_PORT": "8060",
            "METRICS_PORT": "9090",
            "ENVIRONMENT": "test",
        }.get(key, default)

        # Call startup event
        await main_standalone.startup_event()

        # Verify logger was called with configuration
        assert mock_logger.info.call_count >= 2
        # Check that config was logged (contains the expected values)
        config_log_call = [
            call
            for call in mock_logger.info.call_args_list
            if "Standalone configuration loaded:" in str(call)
        ]
        assert len(config_log_call) == 1

        # Verify standalone mode messages were logged
        mock_logger.info.assert_any_call(
            "NodeBridgeOrchestrator API running in standalone mode"
        )
        mock_logger.info.assert_any_call(
            "Full workflow orchestration requires omnibase runtime"
        )

        # Verify node_instance remains None (no full initialization in standalone mode)
        assert main_standalone.node_instance is None

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.os.getenv")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.logger")
    async def test_startup_event_container_no_initialize(
        self, mock_logger, mock_getenv
    ):
        """Test startup with default environment values."""
        # Use default values for all environment variables
        mock_getenv.side_effect = lambda key, default=None: default

        # Call startup event
        await main_standalone.startup_event()

        # Verify logger was called with configuration (using defaults)
        assert mock_logger.info.call_count >= 2

        # Verify standalone mode messages were logged
        mock_logger.info.assert_any_call(
            "NodeBridgeOrchestrator API running in standalone mode"
        )
        mock_logger.info.assert_any_call(
            "Full workflow orchestration requires omnibase runtime"
        )

        # Verify node_instance remains None
        assert main_standalone.node_instance is None

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.os.getenv")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.StandaloneConfig")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.logger")
    async def test_startup_event_exception(
        self, mock_logger, mock_config_class, mock_getenv
    ):
        """Test startup when an exception occurs during config creation."""
        # Setup environment variable mocks
        mock_getenv.side_effect = lambda key, default=None: default

        # Setup config to raise exception
        mock_config_class.side_effect = Exception("Test error")

        # Call startup event
        await main_standalone.startup_event()

        # Verify logger was called with error
        mock_logger.error.assert_called_once()
        assert "Failed to initialize standalone mode: Test error" in str(
            mock_logger.error.call_args
        )

        # Verify node_instance remains None
        assert main_standalone.node_instance is None


class TestShutdownEvent:
    """Test suite for application shutdown event."""

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.logger")
    async def test_shutdown_event_success(self, mock_logger):
        """Test successful application shutdown."""
        # Setup mock node
        mock_node = MagicMock()
        mock_node.shutdown = AsyncMock()
        main_standalone.node_instance = mock_node

        # Call shutdown event
        await main_standalone.shutdown_event()

        # Verify node shutdown was called
        mock_node.shutdown.assert_called_once()

        # Verify logger was called
        mock_logger.info.assert_called_once_with(
            "NodeBridgeOrchestrator shutdown complete"
        )

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.logger")
    async def test_shutdown_event_no_node(self, mock_logger):
        """Test shutdown when node_instance is None."""
        # Ensure node_instance is None
        main_standalone.node_instance = None

        # Call shutdown event
        await main_standalone.shutdown_event()

        # Verify logger was not called
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.logger")
    async def test_shutdown_event_exception(self, mock_logger):
        """Test shutdown when an exception occurs."""
        # Setup mock node to raise exception
        mock_node = MagicMock()
        mock_node.shutdown = AsyncMock(side_effect=Exception("Test error"))
        main_standalone.node_instance = mock_node

        # Call shutdown event
        await main_standalone.shutdown_event()

        # Verify logger was called with error
        mock_logger.error.assert_called_once()
        assert (
            "Error during node shutdown: Test error"
            in mock_logger.error.call_args[0][0]
        )


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check_success(self):
        """Test successful health check."""
        client = TestClient(main_standalone.app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "NodeBridgeOrchestrator"
        assert data["version"] == "1.0.0"
        assert data["mode"] == "standalone"


class TestWorkflowSubmissionEndpoint:
    """Test suite for workflow submission endpoint."""

    def test_submit_workflow_success(self):
        """Test successful workflow submission."""
        client = TestClient(main_standalone.app)
        request_data = {"content": "test content", "namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["state"] == "queued"
        assert "workflow_id" in data
        assert "queued in standalone mode" in data["message"]

    def test_submit_workflow_with_correlation_id(self):
        """Test workflow submission with correlation ID."""
        client = TestClient(main_standalone.app)
        request_data = {
            "content": "test content",
            "correlation_id": "test-correlation-id",
            "namespace": "test.namespace",
        }

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["state"] == "queued"
        assert "workflow_id" in data

    def test_submit_workflow_default_namespace(self):
        """Test workflow submission with default namespace."""
        client = TestClient(main_standalone.app)
        request_data = {"content": "test content"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True

    def test_submit_workflow_invalid_request(self):
        """Test workflow submission with invalid request data."""
        client = TestClient(main_standalone.app)
        # Missing required 'content' field
        request_data = {"namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 422  # Validation error


class TestWorkflowStatusEndpoint:
    """Test suite for workflow status endpoint."""

    def test_get_workflow_status_success(self):
        """Test successful workflow status query."""
        client = TestClient(main_standalone.app)
        workflow_id = str(uuid4())

        response = client.get(f"/workflow/{workflow_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == workflow_id
        assert data["state"] == "queued"
        assert data["current_step"] == "pending"
        assert "note" in data["result"]
        # Case-insensitive check for "standalone mode" to handle variations
        assert "standalone mode" in data["result"]["note"].lower()


class TestMetricsEndpoint:
    """Test suite for metrics endpoint."""

    def test_metrics_success(self):
        """Test successful metrics query."""
        client = TestClient(main_standalone.app)

        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "workflows_total" in data
        assert "workflows_active" in data
        assert "workflows_completed" in data
        assert "workflows_failed" in data
        assert "mode" in data
        assert data["workflows_total"] == 0
        assert data["workflows_active"] == 0
        assert data["workflows_completed"] == 0
        assert data["workflows_failed"] == 0
        assert data["mode"] == "standalone"


class TestRootEndpoint:
    """Test suite for root endpoint."""

    def test_root_success(self):
        """Test successful root endpoint query."""
        client = TestClient(main_standalone.app)

        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "NodeBridgeOrchestrator"
        assert data["version"] == "1.0.0"
        assert data["mode"] == "standalone"
        assert data["status"] == "running"
        assert "message" in data
        assert "endpoints" in data
        assert "health" in data["endpoints"]
        assert "submit_workflow" in data["endpoints"]
        assert "workflow_status" in data["endpoints"]
        assert "metrics" in data["endpoints"]
        assert "docs" in data["endpoints"]


class TestMainExecution:
    """Test suite for main execution block."""

    @pytest.mark.skip(
        reason="uvicorn is imported conditionally in __main__ block, cannot be patched during import"
    )
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone.uvicorn.run")
    def test_main_execution(self, mock_uvicorn_run):
        """Test execution when run as main module."""
        # This test verifies that uvicorn.run is called with correct parameters
        # when the module is executed directly
        # NOTE: uvicorn is only imported inside 'if __name__ == "__main__"' block,
        # so it cannot be patched as a module attribute during normal test execution
        with patch.dict("__main__.__dict__", {"__name__": "__main__"}):
            # Import the module to trigger the main block
            import sys

            # Remove the module from cache if it exists
            if (
                "omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone"
                in sys.modules
            ):
                del sys.modules[
                    "omninode_bridge.nodes.orchestrator.v1_0_0.main_standalone"
                ]

            # This would normally trigger the main block, but in testing we can't
            # easily test that without actually running the server
            # Instead, we verify the uvicorn.run call would be correct
            mock_uvicorn_run.assert_not_called()  # Not called in test environment
