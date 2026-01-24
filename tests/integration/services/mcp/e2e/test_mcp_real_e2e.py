# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Real MCP E2E integration tests using actual MCP SDK client and server.

These tests use the REAL MCP SDK - not mocks. They start an actual MCP server
using FastMCP and connect to it using the real MCP client (streamable_http_client).

This validates:
- Real MCP protocol compliance (initialize handshake, tools/list, tools/call)
- Real HTTP transport with actual TCP sockets
- Real tool registration and execution through FastMCP
- Proper JSON-RPC message flow

The tests require an available port and use background threads for the server.

Related Ticket: OMN-1408
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp import ClientSession

    from omnibase_infra.services.mcp import MCPServerLifecycle

logger = logging.getLogger(__name__)


# ============================================================================
# TEST MARKERS AND MODULE CONFIGURATION
# ============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.real_mcp,
    pytest.mark.timeout(30),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def find_available_port() -> int:
    """Find an available port for the test server.

    Uses the OS to find an ephemeral port by binding to port 0.

    Returns:
        Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port: int = s.getsockname()[1]
        return port


def wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for server to be ready to accept connections.

    Args:
        host: Server hostname.
        port: Server port.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if server is ready, False if timeout reached.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                return True
        except (OSError, TimeoutError):
            time.sleep(0.1)
    return False


# ============================================================================
# MCP SERVER FIXTURE
# ============================================================================


class MCPServerFixture:
    """Wrapper for a real FastMCP server running in a background thread.

    This class manages the lifecycle of a real MCP server using the MCP SDK's
    FastMCP class, running it via uvicorn in a background thread.

    The MCP SDK's streamable_http_app() creates a Starlette app with a route
    at `/mcp`, so the client URL is `http://host:port/mcp`.

    Attributes:
        host: Server hostname.
        port: Server port.
        tools_executed: List of tool calls recorded for assertions.
    """

    def __init__(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        """Initialize the server wrapper.

        Args:
            host: Hostname to bind to.
            port: Port to bind to. If None, finds an available port.
        """
        self.host = host
        self.port = port or find_available_port()
        self.tools_executed: list[dict[str, object]] = []
        self._tools_lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._server: object | None = None  # uvicorn.Server instance

    @property
    def url(self) -> str:
        """Get the server URL.

        The MCP SDK creates a route at /mcp, so the client URL includes /mcp.
        """
        return f"http://{self.host}:{self.port}/mcp"

    def record_tool_execution(self, record: dict[str, object]) -> None:
        """Thread-safe method to record a tool execution.

        Args:
            record: Tool execution record with tool_name, arguments, etc.
        """
        with self._tools_lock:
            self.tools_executed.append(record)

    def start(self) -> None:
        """Start the MCP server in a background thread."""
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to be ready by checking TCP connectivity
        if not wait_for_server(self.host, self.port, timeout=15.0):
            raise RuntimeError(
                f"MCP server not accepting connections on {self.host}:{self.port}"
            )

        logger.info("Real MCP server started", extra={"url": self.url})

    def stop(self) -> None:
        """Stop the MCP server."""
        if self._server is not None:
            self._server.should_exit = True  # type: ignore[attr-defined]
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)
            if self._server_thread.is_alive():
                logger.warning(
                    "MCP server thread did not terminate within timeout",
                    extra={"host": self.host, "port": self.port},
                )
        logger.info("Real MCP server stopped")

    def _run_server(self) -> None:
        """Run the uvicorn server (called from background thread)."""
        import uvicorn
        from mcp.server.fastmcp import FastMCP

        # Create FastMCP server
        mcp = FastMCP(
            "ONEX Real E2E Test Server",
            stateless_http=True,
            json_response=True,
        )

        # Reference to parent for tool execution tracking
        parent = self

        # Register a test tool that echoes input
        @mcp.tool(name="echo_tool", description="Echoes input for testing")
        def echo_tool(message: str = "default") -> str:
            """Echo the input message.

            Args:
                message: Message to echo.

            Returns:
                Echoed message with metadata.
            """
            correlation_id = str(uuid4())
            parent.record_tool_execution(
                {
                    "tool_name": "echo_tool",
                    "arguments": {"message": message},
                    "correlation_id": correlation_id,
                }
            )
            return f"Echo: {message} (correlation_id: {correlation_id})"

        # Register a computation tool
        @mcp.tool(name="compute_sum", description="Computes the sum of two numbers")
        def compute_sum(a: int = 0, b: int = 0) -> str:
            """Compute sum of two numbers.

            Args:
                a: First number.
                b: Second number.

            Returns:
                Sum result as string.
            """
            result = a + b
            parent.record_tool_execution(
                {
                    "tool_name": "compute_sum",
                    "arguments": {"a": a, "b": b},
                    "result": result,
                }
            )
            return f"Sum: {a} + {b} = {result}"

        # The MCP SDK's streamable_http_app() returns a Starlette app with
        # a route at /mcp. We use this directly without Mount wrapper.
        app = mcp.streamable_http_app()

        # Run uvicorn server
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        parent._server = server

        # Run in new event loop
        asyncio.run(server.serve())


@pytest.fixture
async def mcp_server_fixture() -> AsyncIterator[MCPServerFixture]:
    """Fixture providing a real running MCP server.

    Yields:
        MCPServerFixture instance with the server running.
    """
    server = MCPServerFixture()
    server.start()
    yield server
    server.stop()


# ============================================================================
# MCP CLIENT SESSION FIXTURE
# ============================================================================


@contextlib.asynccontextmanager
async def create_mcp_client(url: str) -> AsyncIterator[ClientSession]:
    """Create an MCP client session connected to the given URL.

    Args:
        url: MCP server URL.

    Yields:
        Initialized ClientSession.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session (required by MCP protocol)
            await session.initialize()
            yield session


# ============================================================================
# REAL MCP E2E TESTS
# ============================================================================


class TestRealMCPInitialize:
    """Tests for MCP initialize handshake using real SDK."""

    async def test_initialize_handshake_succeeds(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Real MCP initialize handshake succeeds.

        Verifies:
        - Client can connect to real server
        - Initialize handshake completes successfully
        - Server capabilities are returned
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            # If we get here, initialization succeeded
            # Check server capabilities
            capabilities = session.get_server_capabilities()
            assert capabilities is not None, "Server must return capabilities"
            # MCP servers must report tools capability
            assert capabilities.tools is not None, "Server must support tools"


class TestRealMCPToolDiscovery:
    """Tests for MCP tool discovery using real SDK."""

    async def test_list_tools_returns_registered_tools(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Real MCP tools/list returns registered tools.

        Verifies:
        - tools/list returns the registered test tools
        - Tool metadata (name, description) is correct
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            result = await session.list_tools()

            # Should have tools
            assert result.tools, "Expected at least one tool to be returned"

            # Find our test tools
            tool_names = {tool.name for tool in result.tools}
            assert "echo_tool" in tool_names, f"Expected echo_tool, got: {tool_names}"
            assert "compute_sum" in tool_names, (
                f"Expected compute_sum, got: {tool_names}"
            )

    async def test_tool_has_correct_metadata(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Tools have correct metadata (name, description).

        Verifies the MCP SDK correctly exposes tool metadata.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            result = await session.list_tools()

            # Find echo_tool and verify its metadata
            echo_tool = next((t for t in result.tools if t.name == "echo_tool"), None)
            assert echo_tool is not None, "echo_tool must be present"
            assert echo_tool.description is not None, "echo_tool must have description"
            assert "echo" in echo_tool.description.lower(), (
                f"Expected 'echo' in description, got: {echo_tool.description}"
            )


class TestRealMCPToolInvocation:
    """Tests for MCP tool invocation using real SDK."""

    async def test_call_tool_returns_result(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Real MCP tools/call returns tool result.

        Verifies:
        - Tool can be invoked via MCP protocol
        - Result is returned correctly
        - Tool execution is tracked on server side
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            result = await session.call_tool("echo_tool", {"message": "Hello MCP!"})

            # Should have content
            assert result.content, "Expected content in result"
            assert len(result.content) > 0, "Expected at least one content item"

            # Content should contain our echo message
            first_content = result.content[0]
            assert hasattr(first_content, "text"), "Content should have text attribute"
            assert "Hello MCP!" in first_content.text, (
                f"Expected message in result, got: {first_content.text}"
            )

            # Verify tool was actually executed on server
            assert len(mcp_server_fixture.tools_executed) == 1, (
                "Tool should be executed exactly once"
            )
            assert mcp_server_fixture.tools_executed[0]["tool_name"] == "echo_tool"
            args = mcp_server_fixture.tools_executed[0]["arguments"]
            assert isinstance(args, dict)
            assert args["message"] == "Hello MCP!"

    async def test_call_compute_tool_with_arguments(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Real MCP can invoke tool with typed arguments.

        Verifies argument passing and typed parameter handling.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            result = await session.call_tool("compute_sum", {"a": 5, "b": 7})

            # Should have content
            assert result.content, "Expected content in result"
            first_content = result.content[0]
            assert hasattr(first_content, "text")
            assert "12" in first_content.text, (
                f"Expected sum 12 in result: {first_content.text}"
            )

            # Verify execution on server
            assert len(mcp_server_fixture.tools_executed) == 1
            exec_record = mcp_server_fixture.tools_executed[0]
            assert exec_record["tool_name"] == "compute_sum"
            assert exec_record["result"] == 12

    async def test_multiple_tool_calls_are_independent(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Multiple tool calls are tracked independently.

        Verifies each call has its own tracking entry.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            # Make multiple calls
            await session.call_tool("echo_tool", {"message": "First"})
            await session.call_tool("echo_tool", {"message": "Second"})
            await session.call_tool("compute_sum", {"a": 1, "b": 2})

            # Should have 3 execution records
            assert len(mcp_server_fixture.tools_executed) == 3, (
                f"Expected 3 calls, got {len(mcp_server_fixture.tools_executed)}"
            )

            # Verify order and content
            assert mcp_server_fixture.tools_executed[0]["tool_name"] == "echo_tool"
            first_args = mcp_server_fixture.tools_executed[0]["arguments"]
            assert isinstance(first_args, dict)
            assert first_args["message"] == "First"

            assert mcp_server_fixture.tools_executed[1]["tool_name"] == "echo_tool"
            second_args = mcp_server_fixture.tools_executed[1]["arguments"]
            assert isinstance(second_args, dict)
            assert second_args["message"] == "Second"

            assert mcp_server_fixture.tools_executed[2]["tool_name"] == "compute_sum"


class TestRealMCPErrorHandling:
    """Tests for MCP error handling using real SDK."""

    async def test_nonexistent_tool_returns_error_result(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Calling nonexistent tool returns error result.

        The MCP SDK returns a CallToolResult with isError=True for
        unknown tools, rather than raising an exception. This test
        verifies this behavior.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            result = await session.call_tool("nonexistent_tool_xyz_12345", {})

            # The SDK returns a result with isError=True
            assert result.isError is True, "Expected isError=True for nonexistent tool"
            assert result.content, "Expected error content"
            first_content = result.content[0]
            assert hasattr(first_content, "text"), "Expected text content"
            assert "nonexistent_tool_xyz_12345" in first_content.text, (
                f"Expected tool name in error message: {first_content.text}"
            )

    async def test_tool_with_default_arguments(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Tool with default arguments works when arguments omitted.

        Verifies default argument handling through real MCP protocol.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            # Call echo_tool without message - should use default
            result = await session.call_tool("echo_tool", {})

            assert result.content, "Expected content in result"
            first_content = result.content[0]
            assert hasattr(first_content, "text")
            # Should contain the default value
            assert "default" in first_content.text, (
                f"Expected default message in result: {first_content.text}"
            )


class TestRealMCPConcurrency:
    """Tests for MCP concurrency using real SDK."""

    async def test_concurrent_tool_calls(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Concurrent tool calls are handled correctly.

        Verifies the real MCP server handles concurrent requests.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            # Make concurrent calls
            tasks = [
                session.call_tool("echo_tool", {"message": f"Concurrent {i}"})
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 5
            for result in results:
                assert result.content
                assert len(result.content) > 0

            # All should be tracked
            assert len(mcp_server_fixture.tools_executed) == 5

            # All messages should be present (order may vary due to concurrency)
            messages: set[str] = set()
            for exec_record in mcp_server_fixture.tools_executed:
                args = exec_record.get("arguments")
                if isinstance(args, dict):
                    msg = args.get("message")
                    if isinstance(msg, str):  # Explicit str check for type safety
                        messages.add(msg)
            expected: set[str] = {f"Concurrent {i}" for i in range(5)}
            assert messages == expected, f"Expected {expected}, got {messages}"


class TestRealMCPProtocolCompliance:
    """Tests verifying MCP protocol compliance with real SDK."""

    async def test_server_reports_protocol_version(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Server reports MCP protocol version.

        Verifies the real MCP SDK correctly reports protocol version.
        """
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(mcp_server_fixture.url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize and capture result
                init_result = await session.initialize()

                # Should have protocol version
                assert init_result.protocolVersion, (
                    "Server must report protocol version"
                )
                # Protocol version should be a valid string
                assert isinstance(init_result.protocolVersion, str)
                assert len(init_result.protocolVersion) > 0

    async def test_server_reports_server_info(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Server reports server info.

        Verifies the real MCP SDK correctly reports server information.
        """
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(mcp_server_fixture.url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()

                # Should have server info
                assert init_result.serverInfo, "Server must report server info"
                assert init_result.serverInfo.name, "Server must have a name"
                # Our test server is named "ONEX Real E2E Test Server"
                assert (
                    "ONEX" in init_result.serverInfo.name
                    or "Test" in init_result.serverInfo.name
                )

    async def test_tool_input_schema_is_valid_json_schema(
        self,
        mcp_server_fixture: MCPServerFixture,
    ) -> None:
        """Tool input schemas are valid JSON schemas.

        Verifies the MCP SDK generates valid JSON schemas for tools.
        """
        async with create_mcp_client(mcp_server_fixture.url) as session:
            result = await session.list_tools()

            for tool in result.tools:
                # Each tool should have an input schema
                schema = tool.inputSchema
                assert schema is not None, f"Tool {tool.name} must have input schema"
                # Schema should be a dict (JSON Schema object)
                assert isinstance(schema, dict), (
                    f"Schema must be dict, got {type(schema)}"
                )
                # JSON Schema should have a type field
                assert "type" in schema, f"Schema must have type field: {schema}"


# ============================================================================
# MCP CONSUL SERVER FIXTURE
# ============================================================================


class MCPConsulServerFixture:
    """Wrapper for a real FastMCP server that discovers tools from Consul.

    This class manages the lifecycle of a real MCP server that discovers
    tools from Consul instead of using locally-registered test tools.
    The server runs via uvicorn in a background thread.

    Attributes:
        host: Server hostname.
        port: Server port.
        lifecycle: MCPServerLifecycle for Consul discovery.
        discovered_tools: List of tools discovered from Consul.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        consul_host: str = "localhost",
        consul_port: int = 28500,
    ) -> None:
        """Initialize the server wrapper.

        Args:
            host: Hostname to bind to.
            port: Port to bind to. If None, finds an available port.
            consul_host: Consul server hostname.
            consul_port: Consul server port.
        """
        import os

        self.host = host
        self.port = port or find_available_port()
        self.consul_host = os.getenv("CONSUL_HOST", consul_host)
        self.consul_port = int(os.getenv("CONSUL_PORT", str(consul_port)))
        self.discovered_tools: list[dict[str, object]] = []
        # Use TYPE_CHECKING import pattern - lifecycle is only used at runtime
        self._lifecycle: MCPServerLifecycle | None = None
        self._server_thread: threading.Thread | None = None
        self._server: object | None = None

    @property
    def url(self) -> str:
        """Get the server URL.

        The MCP SDK creates a route at /mcp, so the client URL includes /mcp.
        """
        return f"http://{self.host}:{self.port}/mcp"

    async def discover_tools(self) -> None:
        """Discover tools from Consul using MCPServerLifecycle.

        This must be called before start() to populate discovered_tools.
        """
        from omnibase_infra.services.mcp import MCPServerLifecycle, ModelMCPServerConfig

        config = ModelMCPServerConfig(
            dev_mode=False,
            consul_host=self.consul_host,
            consul_port=self.consul_port,
            http_port=self.port,
            http_host=self.host,
            kafka_enabled=False,
        )

        self._lifecycle = MCPServerLifecycle(config)
        await self._lifecycle.start()

        # Get discovered tools from registry
        if self._lifecycle.registry:
            tools = await self._lifecycle.registry.list_tools()
            self.discovered_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "version": tool.version,
                    "endpoint": tool.endpoint,
                    "orchestrator_node_id": tool.orchestrator_node_id,
                    "orchestrator_service_id": tool.orchestrator_service_id,
                    "timeout_seconds": tool.timeout_seconds,
                }
                for tool in tools
            ]

        logger.info(
            "Discovered tools from Consul",
            extra={
                "tool_count": len(self.discovered_tools),
                "consul_host": self.consul_host,
                "consul_port": self.consul_port,
            },
        )

    def start(self) -> None:
        """Start the MCP server in a background thread.

        The server exposes tools discovered from Consul.
        """
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to be ready by checking TCP connectivity
        if not wait_for_server(self.host, self.port, timeout=15.0):
            raise RuntimeError(
                f"MCP server not accepting connections on {self.host}:{self.port}"
            )

        logger.info(
            "Real MCP server with Consul discovery started",
            extra={
                "url": self.url,
                "tool_count": len(self.discovered_tools),
            },
        )

    def stop(self) -> None:
        """Stop the MCP server."""
        if self._server is not None:
            self._server.should_exit = True  # type: ignore[attr-defined]
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)
            if self._server_thread.is_alive():
                logger.warning(
                    "MCP server thread did not terminate within timeout",
                    extra={"host": self.host, "port": self.port},
                )
        logger.info("Real MCP server with Consul discovery stopped")

    async def shutdown_lifecycle(self) -> None:
        """Shutdown the MCPServerLifecycle (cleanup Consul resources)."""
        if self._lifecycle is not None:
            await self._lifecycle.shutdown()

    def _run_server(self) -> None:
        """Run the uvicorn server (called from background thread)."""
        import uvicorn
        from mcp.server.fastmcp import FastMCP

        # Create FastMCP server
        mcp = FastMCP(
            "ONEX Consul E2E Test Server",
            stateless_http=True,
            json_response=True,
        )

        # Reference to parent for discovered tools
        parent = self

        # Dynamically register discovered tools
        for tool_info in parent.discovered_tools:
            tool_name = str(tool_info["name"])
            tool_desc = str(tool_info.get("description", f"ONEX tool: {tool_name}"))

            # Create a closure to capture tool info
            def make_tool_handler(
                captured_name: str, captured_info: dict[str, object]
            ) -> Callable[[str], str]:
                """Create a tool handler that echoes tool info for testing."""

                def tool_handler(
                    input_data: str = "",
                ) -> str:
                    """Handle tool invocation by returning tool metadata.

                    For E2E testing, we return the tool metadata to verify
                    Consul discovery worked correctly. In a real deployment,
                    this would dispatch to the actual ONEX orchestrator.

                    Args:
                        input_data: Optional input data for the tool.

                    Returns:
                        JSON string with tool info and input data.
                    """
                    import json

                    correlation_id = str(uuid4())
                    return json.dumps(
                        {
                            "tool_name": captured_name,
                            "source": "consul_discovery",
                            "endpoint": captured_info.get("endpoint"),
                            "orchestrator_node_id": captured_info.get(
                                "orchestrator_node_id"
                            ),
                            "input_data": input_data,
                            "correlation_id": correlation_id,
                        }
                    )

                return tool_handler

            # Register the tool with FastMCP
            handler = make_tool_handler(tool_name, tool_info)
            mcp.tool(name=tool_name, description=tool_desc)(handler)

        # If no tools discovered, register a fallback tool
        if not parent.discovered_tools:

            @mcp.tool(
                name="no_tools_discovered",
                description="Fallback tool when no Consul tools discovered",
            )
            def no_tools_discovered() -> str:
                """Fallback when no tools discovered from Consul."""
                return "No MCP-enabled tools discovered from Consul"

        # Get the Starlette app
        app = mcp.streamable_http_app()

        # Run uvicorn server
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        parent._server = server

        # Run in new event loop
        asyncio.run(server.serve())


@pytest.fixture
async def mcp_consul_server_fixture(
    infra_availability: dict[str, bool],
) -> AsyncIterator[MCPConsulServerFixture]:
    """Fixture providing a real MCP server that discovers tools from Consul.

    This fixture requires Consul to be available. If Consul is not reachable,
    the test is skipped.

    Args:
        infra_availability: Infrastructure availability flags from conftest.

    Yields:
        MCPConsulServerFixture instance with the server running.
    """
    if not infra_availability.get("consul", False):
        pytest.skip(
            "Consul not available. Set CONSUL_HOST and ensure Consul is running."
        )

    server = MCPConsulServerFixture()
    await server.discover_tools()
    server.start()
    yield server
    server.stop()
    await server.shutdown_lifecycle()


# ============================================================================
# REAL MCP WITH CONSUL TESTS
# ============================================================================


@pytest.mark.consul
class TestRealMCPWithConsul:
    """Tests for MCP with Consul-discovered tools using real SDK.

    These tests verify that the MCP server can discover tools from Consul
    and expose them via the MCP protocol. They require a running Consul
    instance with MCP-enabled services registered.

    Skip Behavior:
        - Tests skip if Consul is not available (CONSUL_HOST not set or unreachable)
        - Tests skip if no MCP-enabled tools are discovered from Consul
    """

    async def test_consul_discovery_populates_registry(
        self,
        mcp_consul_server_fixture: MCPConsulServerFixture,
    ) -> None:
        """Consul discovery populates the tool registry.

        Verifies:
        - MCPServerLifecycle successfully starts with Consul discovery
        - Tools are discovered from Consul (if any are registered)
        - Discovered tools have expected metadata fields
        """
        # The fixture has already performed discovery
        # Just verify the discovery happened (even if empty)
        assert mcp_consul_server_fixture.discovered_tools is not None

        if not mcp_consul_server_fixture.discovered_tools:
            pytest.skip(
                "No MCP-enabled tools discovered from Consul. "
                "Register services with tags: mcp-enabled, node-type:orchestrator, mcp-tool:<name>"
            )

        # Verify discovered tools have required fields
        for tool in mcp_consul_server_fixture.discovered_tools:
            assert "name" in tool, "Tool must have name"
            assert "description" in tool, "Tool must have description"
            assert tool["name"], "Tool name must not be empty"

        logger.info(
            "Consul discovery test passed",
            extra={"tool_count": len(mcp_consul_server_fixture.discovered_tools)},
        )

    async def test_list_tools_returns_consul_discovered_tools(
        self,
        mcp_consul_server_fixture: MCPConsulServerFixture,
    ) -> None:
        """MCP tools/list returns Consul-discovered tools.

        Verifies:
        - Real MCP client can connect to server with Consul-discovered tools
        - tools/list returns the tools discovered from Consul
        """
        if not mcp_consul_server_fixture.discovered_tools:
            pytest.skip("No MCP-enabled tools discovered from Consul")

        async with create_mcp_client(mcp_consul_server_fixture.url) as session:
            result = await session.list_tools()

            # Should have tools from Consul
            assert result.tools, "Expected tools from Consul discovery"

            # Verify discovered tools are present
            returned_names = {tool.name for tool in result.tools}
            expected_names = {
                str(t["name"]) for t in mcp_consul_server_fixture.discovered_tools
            }

            # All expected tools should be in returned tools
            for expected_name in expected_names:
                assert expected_name in returned_names, (
                    f"Expected tool '{expected_name}' not in returned tools: "
                    f"{returned_names}"
                )

    async def test_consul_discovered_tool_has_correct_metadata(
        self,
        mcp_consul_server_fixture: MCPConsulServerFixture,
    ) -> None:
        """Consul-discovered tools have correct metadata.

        Verifies:
        - Tool name matches Consul tag
        - Tool has description
        - Tool has input schema
        """
        if not mcp_consul_server_fixture.discovered_tools:
            pytest.skip("No MCP-enabled tools discovered from Consul")

        async with create_mcp_client(mcp_consul_server_fixture.url) as session:
            result = await session.list_tools()
            assert result.tools, "Expected tools"

            # Check first tool metadata
            first_tool = result.tools[0]
            assert first_tool.name, "Tool must have name"
            assert first_tool.description, "Tool must have description"
            assert first_tool.inputSchema is not None, "Tool must have input schema"
            assert isinstance(first_tool.inputSchema, dict), (
                "Input schema must be a dict"
            )

    async def test_call_consul_discovered_tool(
        self,
        mcp_consul_server_fixture: MCPConsulServerFixture,
    ) -> None:
        """Can call a Consul-discovered tool via MCP protocol.

        Verifies:
        - Tool can be invoked via MCP protocol
        - Result contains expected fields from our test handler
        - Correlation ID is included for tracing
        """
        if not mcp_consul_server_fixture.discovered_tools:
            pytest.skip("No MCP-enabled tools discovered from Consul")

        # Get first discovered tool
        first_tool = mcp_consul_server_fixture.discovered_tools[0]
        tool_name = str(first_tool["name"])

        async with create_mcp_client(mcp_consul_server_fixture.url) as session:
            result = await session.call_tool(tool_name, {"input_data": "test from E2E"})

            # Should have content
            assert result.content, f"Expected content from tool {tool_name}"
            assert len(result.content) > 0, "Expected at least one content item"

            # Content should be text with our test response
            first_content = result.content[0]
            assert hasattr(first_content, "text"), "Content should have text attribute"

            # Parse the JSON response
            import json

            response_data = json.loads(first_content.text)

            # Verify response structure
            assert response_data["tool_name"] == tool_name, (
                f"Expected tool_name '{tool_name}', got '{response_data.get('tool_name')}'"
            )
            assert response_data["source"] == "consul_discovery", (
                "Response should indicate consul_discovery source"
            )
            assert "correlation_id" in response_data, (
                "Response should include correlation_id"
            )

    async def test_multiple_consul_tools_are_independent(
        self,
        mcp_consul_server_fixture: MCPConsulServerFixture,
    ) -> None:
        """Multiple Consul-discovered tools can be called independently.

        Verifies:
        - Each tool returns its own metadata
        - Tools don't interfere with each other
        """
        if len(mcp_consul_server_fixture.discovered_tools) < 2:
            pytest.skip("Need at least 2 MCP-enabled tools in Consul for this test")

        import json

        async with create_mcp_client(mcp_consul_server_fixture.url) as session:
            # Call first two tools
            tool1 = str(mcp_consul_server_fixture.discovered_tools[0]["name"])
            tool2 = str(mcp_consul_server_fixture.discovered_tools[1]["name"])

            result1 = await session.call_tool(tool1, {"input_data": "call1"})
            result2 = await session.call_tool(tool2, {"input_data": "call2"})

            # Both should succeed
            assert result1.content, f"Expected content from {tool1}"
            assert result2.content, f"Expected content from {tool2}"

            # Verify content has text attribute
            first_content1 = result1.content[0]
            first_content2 = result2.content[0]
            assert hasattr(first_content1, "text"), "Content should have text"
            assert hasattr(first_content2, "text"), "Content should have text"

            # Verify each returns correct tool name
            response1 = json.loads(first_content1.text)
            response2 = json.loads(first_content2.text)

            assert response1["tool_name"] == tool1, "First tool should return its name"
            assert response2["tool_name"] == tool2, "Second tool should return its name"
            assert response1["correlation_id"] != response2["correlation_id"], (
                "Each call should have unique correlation_id"
            )

    async def test_server_reports_consul_tool_count_in_capabilities(
        self,
        mcp_consul_server_fixture: MCPConsulServerFixture,
    ) -> None:
        """Server with Consul tools reports tools capability.

        Verifies the MCP server correctly reports it supports tools.
        """
        if not mcp_consul_server_fixture.discovered_tools:
            pytest.skip("No MCP-enabled tools discovered from Consul")

        async with create_mcp_client(mcp_consul_server_fixture.url) as session:
            capabilities = session.get_server_capabilities()
            assert capabilities is not None, "Server must return capabilities"
            assert capabilities.tools is not None, "Server must support tools"

    async def test_fallback_when_no_consul_tools(
        self,
        infra_availability: dict[str, bool],
    ) -> None:
        """Server provides fallback tool when no Consul tools discovered.

        This test verifies graceful degradation when Consul is available
        but no MCP-enabled services are registered.
        """
        if not infra_availability.get("consul", False):
            pytest.skip("Consul not available")

        # Create server with discovery (may find no tools)
        server = MCPConsulServerFixture()
        await server.discover_tools()

        # If tools were discovered, skip this test
        if server.discovered_tools:
            await server.shutdown_lifecycle()
            pytest.skip("MCP tools found in Consul - fallback test not applicable")

        # Start server with no discovered tools
        server.start()

        try:
            async with create_mcp_client(server.url) as session:
                result = await session.list_tools()

                # Should have fallback tool
                assert result.tools, "Expected fallback tool"
                assert any(t.name == "no_tools_discovered" for t in result.tools), (
                    "Expected 'no_tools_discovered' fallback tool"
                )
        finally:
            server.stop()
            await server.shutdown_lifecycle()
