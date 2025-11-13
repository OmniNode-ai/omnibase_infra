"""Custom exceptions for agent registry."""


class AgentRegistryError(Exception):
    """Base exception for agent registry errors."""

    pass


class AgentNotFoundError(AgentRegistryError):
    """Raised when agent is not found in registry."""

    def __init__(self, agent_id: str, message: str = "") -> None:
        """
        Initialize AgentNotFoundError.

        Args:
            agent_id: ID of the agent that was not found
            message: Optional error message
        """
        self.agent_id = agent_id
        super().__init__(message or f"Agent '{agent_id}' not found in registry")


class NoAgentFoundError(AgentRegistryError):
    """Raised when no suitable agent is found for a task."""

    def __init__(self, message: str, required_capabilities: list[str] = None) -> None:
        """
        Initialize NoAgentFoundError.

        Args:
            message: Error message
            required_capabilities: Required capabilities that couldn't be matched
        """
        self.required_capabilities = required_capabilities or []
        super().__init__(message)


class DuplicateAgentError(AgentRegistryError):
    """Raised when attempting to register an agent with duplicate ID."""

    def __init__(self, agent_id: str) -> None:
        """
        Initialize DuplicateAgentError.

        Args:
            agent_id: ID of the duplicate agent
        """
        self.agent_id = agent_id
        super().__init__(f"Agent '{agent_id}' is already registered")
