#!/usr/bin/env python3
"""Unit tests for orchestrator main.py REST API wrapper.

Tests cover:
- FastAPI application initialization
- CORS middleware configuration
- Health check endpoint
- Workflow submission endpoint
- Workflow status endpoint
- Metrics endpoint
- Error handling
- Application lifespan management
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Import the module to test
from omninode_bridge.nodes.orchestrator.v1_0_0 import main


class TestMainModule:
    """Test suite for main module components."""

    def test_global_orchestrator_initially_none(self):
        """Test that global orchestrator is initially None."""
        assert main.orchestrator is None

    def test_workflow_submission_request_model(self):
        """Test WorkflowSubmissionRequest model validation."""
        # Test with all required fields
        request = main.WorkflowSubmissionRequest(
            content="test content", correlation_id="test-id", namespace="test.namespace"
        )
        assert request.content == "test content"
        assert request.correlation_id == "test-id"
        assert request.namespace == "test.namespace"

        # Test with default namespace
        request = main.WorkflowSubmissionRequest(content="test content")
        assert request.content == "test content"
        assert request.correlation_id is None
        assert request.namespace == "omninode.bridge"

        # Test validation error for missing content
        with pytest.raises(ValidationError):
            main.WorkflowSubmissionRequest()

    def test_workflow_submission_response_model(self):
        """Test WorkflowSubmissionResponse model validation."""
        response = main.WorkflowSubmissionResponse(
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
        response = main.WorkflowSubmissionResponse(
            success=False, workflow_id="test-id", state="failed"
        )
        assert response.success is False
        assert response.message is None

    def test_workflow_status_response_model(self):
        """Test WorkflowStatusResponse model validation."""
        response = main.WorkflowStatusResponse(
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
        response = main.WorkflowStatusResponse(
            workflow_id="test-id", state="processing"
        )
        assert response.current_step is None
        assert response.result is None

    def test_health_response_model(self):
        """Test HealthResponse model validation."""
        response = main.HealthResponse(
            status="healthy", service="TestService", version="1.0.0"
        )
        assert response.status == "healthy"
        assert response.service == "TestService"
        assert response.version == "1.0.0"


class TestFastAPIApplication:
    """Test suite for FastAPI application configuration."""

    def test_app_creation(self):
        """Test FastAPI application creation."""
        assert main.app is not None
        assert main.app.title == "NodeBridgeOrchestrator API"
        assert (
            main.app.description
            == "REST API wrapper for ONEX v2.0 workflow orchestration"
        )
        assert main.app.version == "1.0.0"

    def test_cors_middleware_configuration(self):
        """Test CORS middleware configuration."""
        # Check that CORS middleware is added
        cors_middleware = None
        for middleware in main.app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break

        assert cors_middleware is not None
        # Note: We can't easily inspect the middleware options without more complex setup

    def test_routes_registration(self):
        """Test that all routes are registered."""
        routes = [route.path for route in main.app.routes]
        assert "/health" in routes
        assert "/workflow/submit" in routes
        assert "/workflow/{workflow_id}/status" in routes
        assert "/metrics" in routes


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check_success(self):
        """Test successful health check."""
        client = TestClient(main.app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "NodeBridgeOrchestrator"
        assert data["version"] == "1.0.0"


class TestWorkflowSubmissionEndpoint:
    """Test suite for workflow submission endpoint."""

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator", None)
    def test_submit_workflow_orchestrator_not_initialized(self):
        """Test workflow submission when orchestrator is not initialized."""
        client = TestClient(main.app)
        request_data = {"content": "test content", "namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == "Orchestrator not initialized"

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.ModelStampRequestInput")
    def test_submit_workflow_success(self, mock_stamp_input, mock_orchestrator):
        """Test successful workflow submission."""
        # Setup mocks
        mock_orchestrator.handle_workflow_start = AsyncMock(
            return_value={"workflow_id": uuid4(), "state": "processing"}
        )
        mock_stamp_input.return_value = MagicMock()

        client = TestClient(main.app)
        request_data = {"content": "test content", "namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["state"] == "processing"
        assert "workflow_id" in data
        assert data["message"] == "Workflow submitted successfully"

        # Verify mocks were called
        mock_stamp_input.assert_called_once()
        mock_orchestrator.handle_workflow_start.assert_called_once()

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.ModelStampRequestInput")
    def test_submit_workflow_orchestrator_returns_none(
        self, mock_stamp_input, mock_orchestrator
    ):
        """Test workflow submission when orchestrator returns None."""
        # Setup mocks
        mock_orchestrator.handle_workflow_start = AsyncMock(return_value=None)
        mock_stamp_input.return_value = MagicMock()

        client = TestClient(main.app)
        request_data = {"content": "test content", "namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Failed to start workflow"

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator")
    def test_submit_workflow_orchestrator_exception(self, mock_orchestrator):
        """Test workflow submission when orchestrator raises exception."""
        # Setup mock to raise exception
        mock_orchestrator.handle_workflow_start = AsyncMock(
            side_effect=Exception("Test error")
        )

        client = TestClient(main.app)
        request_data = {"content": "test content", "namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "Error processing workflow: Test error" in data["detail"]

    def test_submit_workflow_invalid_request(self):
        """Test workflow submission with invalid request data."""
        client = TestClient(main.app)
        # Missing required 'content' field
        request_data = {"namespace": "test.namespace"}

        response = client.post("/workflow/submit", json=request_data)

        assert response.status_code == 422  # Validation error


class TestWorkflowStatusEndpoint:
    """Test suite for workflow status endpoint."""

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator", None)
    def test_get_workflow_status_orchestrator_not_initialized(self):
        """Test workflow status when orchestrator is not initialized."""
        client = TestClient(main.app)
        workflow_id = str(uuid4())

        response = client.get(f"/workflow/{workflow_id}/status")

        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == "Orchestrator not initialized"

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator")
    def test_get_workflow_status_success(self, mock_orchestrator):
        """Test successful workflow status query."""
        client = TestClient(main.app)
        workflow_id = str(uuid4())

        response = client.get(f"/workflow/{workflow_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == workflow_id
        assert data["state"] == "unknown"
        assert data["current_step"] is None
        assert data["result"] is None

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.orchestrator")
    def test_get_workflow_status_exception(self, mock_orchestrator):
        """Test workflow status when orchestrator raises exception."""
        # This test is limited since the current implementation doesn't actually
        # call the orchestrator for status queries, but we test the error handling
        # path that would be used if it did
        with patch(
            "omninode_bridge.nodes.orchestrator.v1_0_0.main.logger"
        ) as mock_logger:
            # Force an exception in the endpoint
            with patch.object(
                main, "WorkflowStatusResponse", side_effect=Exception("Test error")
            ):
                client = TestClient(main.app)
                workflow_id = str(uuid4())

                response = client.get(f"/workflow/{workflow_id}/status")

                assert response.status_code == 500
                data = response.json()
                assert "Error querying workflow: Test error" in data["detail"]


class TestMetricsEndpoint:
    """Test suite for metrics endpoint."""

    def test_metrics_success(self):
        """Test successful metrics query."""
        client = TestClient(main.app)

        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "workflows_total" in data
        assert "workflows_active" in data
        assert "workflows_completed" in data
        assert "workflows_failed" in data
        assert data["workflows_total"] == 0
        assert data["workflows_active"] == 0
        assert data["workflows_completed"] == 0
        assert data["workflows_failed"] == 0


class TestApplicationLifespan:
    """Test suite for application lifespan management."""

    @patch.dict(
        "os.environ",
        {
            "POSTGRES_PASSWORD": "test_password",
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        },
    )
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.ModelONEXContainer")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.NodeBridgeOrchestrator")
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.logger")
    async def test_lifespan_startup(
        self, mock_logger, mock_orchestrator_class, mock_container_class
    ):
        """Test application startup."""
        # Mock container and orchestrator
        mock_container_instance = MagicMock()
        mock_container_class.return_value = mock_container_instance
        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        # Mock container.initialize as async callable
        mock_container_instance.initialize = AsyncMock()
        # Mock orchestrator.startup as async callable
        mock_orchestrator_instance.startup = AsyncMock()

        # Create a new app with the mocked lifespan
        from fastapi import FastAPI

        test_app = FastAPI(lifespan=main.lifespan)

        # Simulate startup
        async with main.lifespan(test_app):
            # Verify container was created
            mock_container_class.assert_called_once()
            # Verify orchestrator was initialized with container
            mock_orchestrator_class.assert_called_once_with(mock_container_instance)
            assert main.orchestrator == mock_orchestrator_instance

            # Verify logger was called
            mock_logger.info.assert_any_call("Initializing NodeBridgeOrchestrator...")

    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.logger")
    async def test_lifespan_shutdown(self, mock_logger):
        """Test application shutdown."""
        # Set up initial state
        main.orchestrator = MagicMock()

        # Create a new app with the mocked lifespan
        from fastapi import FastAPI

        test_app = FastAPI(lifespan=main.lifespan)

        # Simulate shutdown
        async with main.lifespan(test_app):
            pass  # Startup happens here

        # After context exits, shutdown should happen
        mock_logger.info.assert_any_call("Shutting down NodeBridgeOrchestrator...")


class TestMainExecution:
    """Test suite for main execution block."""

    @pytest.mark.skip(
        reason="uvicorn is imported conditionally in __main__ block, cannot be patched during import"
    )
    @patch("omninode_bridge.nodes.orchestrator.v1_0_0.main.uvicorn.run")
    def test_main_execution(self, mock_uvicorn_run):
        """Test execution when run as main module."""
        # This test verifies that uvicorn.run is called with correct parameters
        # when the module is executed directly
        # NOTE: uvicorn is only imported inside 'if __name__ == "__main__"' block,
        # so it cannot be patched as a module attribute during normal test execution
        import sys

        import __main__

        with patch.dict(__main__.__dict__, {"__name__": "__main__"}):
            # Remove the module from cache if it exists
            if "omninode_bridge.nodes.orchestrator.v1_0_0.main" in sys.modules:
                del sys.modules["omninode_bridge.nodes.orchestrator.v1_0_0.main"]

            # This would normally trigger the main block, but in testing we can't
            # easily test that without actually running the server
            # Instead, we verify the uvicorn.run call would be correct
            mock_uvicorn_run.assert_not_called()  # Not called in test environment
