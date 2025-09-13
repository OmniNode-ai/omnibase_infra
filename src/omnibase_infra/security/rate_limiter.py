"""
Rate Limiting for ONEX Infrastructure Services

Provides comprehensive rate limiting for event publishing and database operations
to prevent abuse and ensure system stability under load.

Per ONEX security requirements:
- Token bucket algorithm for smooth rate limiting
- Sliding window rate limiting for burst protection
- Per-client rate limiting with IP/user identification
- Configurable limits based on operation type
"""

import time
import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from omnibase_core.core.onex_error import OnexError, CoreErrorCode


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    max_requests: int
    window_seconds: int
    burst_limit: Optional[int] = None
    penalty_seconds: int = 0
    
    def __post_init__(self):
        if self.burst_limit is None:
            self.burst_limit = min(self.max_requests * 2, self.max_requests + 10)


@dataclass
class ClientRateLimitState:
    """Per-client rate limiting state."""
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    request_times: deque = field(default_factory=deque)
    penalty_until: float = 0.0
    total_requests: int = 0
    blocked_requests: int = 0


class ONEXRateLimiter:
    """
    ONEX rate limiter with token bucket and sliding window algorithms.
    
    Features:
    - Token bucket for smooth rate limiting
    - Sliding window for burst protection
    - Per-client tracking with automatic cleanup
    - Configurable penalties for abuse
    - Async-safe implementation
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._rules: Dict[str, RateLimitRule] = {}
        self._client_states: Dict[str, ClientRateLimitState] = defaultdict(ClientRateLimitState)
        self._cleanup_interval = 300.0  # 5 minutes
        self._last_cleanup = time.time()
        
        # Default rate limiting rules
        self._setup_default_rules()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def _setup_default_rules(self):
        """Set up default rate limiting rules for infrastructure operations."""
        
        # Event publishing limits
        self._rules["event_publish"] = RateLimitRule(
            max_requests=1000,     # 1000 events per minute
            window_seconds=60,
            burst_limit=100,       # Allow 100 event burst
            penalty_seconds=30     # 30 second penalty for abuse
        )
        
        # Database query limits
        self._rules["database_query"] = RateLimitRule(
            max_requests=500,      # 500 queries per minute
            window_seconds=60,
            burst_limit=50,        # Allow 50 query burst
            penalty_seconds=60     # 1 minute penalty for abuse
        )
        
        # Health check limits (more permissive)
        self._rules["health_check"] = RateLimitRule(
            max_requests=120,      # 2 per second average
            window_seconds=60,
            burst_limit=10,        # Small burst allowance
            penalty_seconds=5      # Short penalty
        )
        
        # Admin operations (restrictive)
        self._rules["admin_operation"] = RateLimitRule(
            max_requests=10,       # 10 admin ops per minute
            window_seconds=60,
            burst_limit=3,         # Very small burst
            penalty_seconds=300    # 5 minute penalty
        )
        
        self._logger.info(f"Initialized rate limiter with {len(self._rules)} rules")
    
    def configure_rule(self, operation_type: str, rule: RateLimitRule):
        """
        Configure rate limiting rule for operation type.
        
        Args:
            operation_type: Type of operation (e.g., 'event_publish')
            rule: Rate limiting rule configuration
        """
        self._rules[operation_type] = rule
        self._logger.info(f"Configured rate limit rule for {operation_type}: "
                         f"{rule.max_requests}/{rule.window_seconds}s")
    
    async def check_rate_limit(self, client_id: str, operation_type: str, 
                              request_count: int = 1) -> bool:
        """
        Check if client is within rate limits for operation.
        
        Args:
            client_id: Unique client identifier (IP, user ID, etc.)
            operation_type: Type of operation being performed
            request_count: Number of requests (default: 1)
            
        Returns:
            True if request is allowed, False if rate limited
            
        Raises:
            OnexError: If operation type is not configured
        """
        if operation_type not in self._rules:
            raise OnexError(
                f"Rate limit rule not configured for operation: {operation_type}",
                CoreErrorCode.CONFIGURATION_ERROR
            )
        
        rule = self._rules[operation_type]
        state = self._client_states[client_id]
        current_time = time.time()
        
        # Check if client is currently penalized
        if current_time < state.penalty_until:
            state.blocked_requests += request_count
            self._logger.warning(f"Client {client_id} blocked due to penalty until "
                               f"{state.penalty_until - current_time:.1f}s")
            return False
        
        # Token bucket algorithm
        time_delta = current_time - state.last_refill
        state.last_refill = current_time
        
        # Refill tokens
        tokens_to_add = (rule.max_requests / rule.window_seconds) * time_delta
        state.tokens = min(rule.burst_limit, state.tokens + tokens_to_add)
        
        # Check if enough tokens available
        if state.tokens >= request_count:
            state.tokens -= request_count
            state.total_requests += request_count
            
            # Add to sliding window
            state.request_times.append(current_time)
            
            # Cleanup old entries in sliding window
            cutoff_time = current_time - rule.window_seconds
            while state.request_times and state.request_times[0] < cutoff_time:
                state.request_times.popleft()
            
            # Check sliding window limits
            window_requests = len(state.request_times)
            if window_requests > rule.max_requests:
                # Apply penalty
                state.penalty_until = current_time + rule.penalty_seconds
                state.blocked_requests += request_count
                
                self._logger.warning(f"Client {client_id} exceeded window limit for {operation_type}: "
                                   f"{window_requests}/{rule.max_requests}, penalty applied")
                return False
            
            # Cleanup client states periodically
            if current_time - self._last_cleanup > self._cleanup_interval:
                await self._cleanup_inactive_clients()
                self._last_cleanup = current_time
            
            return True
        
        else:
            state.blocked_requests += request_count
            self._logger.info(f"Client {client_id} rate limited for {operation_type}: "
                            f"tokens={state.tokens:.1f}, needed={request_count}")
            return False
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """
        Get rate limiting statistics for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with client statistics
        """
        if client_id not in self._client_states:
            return {"exists": False}
        
        state = self._client_states[client_id]
        current_time = time.time()
        
        return {
            "exists": True,
            "tokens": round(state.tokens, 2),
            "total_requests": state.total_requests,
            "blocked_requests": state.blocked_requests,
            "success_rate": (state.total_requests - state.blocked_requests) / max(state.total_requests, 1),
            "penalized": current_time < state.penalty_until,
            "penalty_remaining": max(0, state.penalty_until - current_time),
            "recent_requests": len(state.request_times)
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global rate limiting statistics.
        
        Returns:
            Dictionary with global statistics
        """
        total_clients = len(self._client_states)
        total_requests = sum(state.total_requests for state in self._client_states.values())
        total_blocked = sum(state.blocked_requests for state in self._client_states.values())
        
        penalized_clients = sum(
            1 for state in self._client_states.values()
            if time.time() < state.penalty_until
        )
        
        return {
            "total_clients": total_clients,
            "total_requests": total_requests,
            "total_blocked": total_blocked,
            "global_success_rate": (total_requests - total_blocked) / max(total_requests, 1),
            "penalized_clients": penalized_clients,
            "configured_rules": list(self._rules.keys())
        }
    
    async def _cleanup_inactive_clients(self):
        """Clean up inactive client states to prevent memory leaks."""
        current_time = time.time()
        inactive_threshold = 3600.0  # 1 hour
        
        inactive_clients = [
            client_id for client_id, state in self._client_states.items()
            if (current_time - state.last_refill > inactive_threshold and
                current_time >= state.penalty_until)
        ]
        
        for client_id in inactive_clients:
            del self._client_states[client_id]
        
        if inactive_clients:
            self._logger.info(f"Cleaned up {len(inactive_clients)} inactive client states")
    
    def reset_client(self, client_id: str):
        """
        Reset rate limiting state for client.
        
        Args:
            client_id: Client identifier to reset
        """
        if client_id in self._client_states:
            del self._client_states[client_id]
            self._logger.info(f"Reset rate limiting state for client: {client_id}")
    
    def set_penalty(self, client_id: str, penalty_seconds: int):
        """
        Apply manual penalty to client.
        
        Args:
            client_id: Client identifier
            penalty_seconds: Duration of penalty in seconds
        """
        state = self._client_states[client_id]
        state.penalty_until = time.time() + penalty_seconds
        
        self._logger.warning(f"Applied {penalty_seconds}s penalty to client: {client_id}")


class RateLimitDecorator:
    """
    Decorator for applying rate limits to functions.
    
    Usage:
        @RateLimitDecorator("event_publish", lambda args: args[0].client_id)
        async def publish_event(self, client_id, event):
            ...
    """
    
    def __init__(self, operation_type: str, client_id_extractor, rate_limiter: Optional[ONEXRateLimiter] = None):
        self.operation_type = operation_type
        self.client_id_extractor = client_id_extractor
        self.rate_limiter = rate_limiter or get_rate_limiter()
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Extract client ID
            client_id = self.client_id_extractor(args, kwargs)
            
            # Check rate limit
            allowed = await self.rate_limiter.check_rate_limit(client_id, self.operation_type)
            
            if not allowed:
                raise OnexError(
                    f"Rate limit exceeded for operation: {self.operation_type}",
                    CoreErrorCode.RATE_LIMIT_EXCEEDED
                )
            
            # Execute function
            return await func(*args, **kwargs)
        
        return wrapper


# Global rate limiter instance
_rate_limiter: Optional[ONEXRateLimiter] = None


def get_rate_limiter() -> ONEXRateLimiter:
    """
    Get global rate limiter instance.
    
    Returns:
        ONEXRateLimiter singleton instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = ONEXRateLimiter()
    
    return _rate_limiter