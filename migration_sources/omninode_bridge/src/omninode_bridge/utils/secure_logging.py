"""
Secure Logging Utilities

Provides utilities for sanitizing logs, masking sensitive data,
and managing production-safe logging without emojis.
"""

import logging
import re
from typing import Any, Optional

from ..config.registry_config import get_registry_config


class SecureLogFormatter(logging.Formatter):
    """
    Secure log formatter that sanitizes sensitive data and handles emojis.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        environment: str = "development",
    ):
        """
        Initialize secure log formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            environment: Environment name
        """
        super().__init__(fmt, datefmt)
        self.environment = environment.lower()
        self._load_config()

    def _load_config(self) -> None:
        """Load secure logging configuration."""
        try:
            registry_config = get_registry_config(self.environment)
            self.sanitize_logs = registry_config.sanitize_logs_in_production
            self.enable_emoji = registry_config.enable_emoji_logs
            self.mask_character = registry_config.security_mask_character
        except (ImportError, AttributeError, KeyError, ValueError) as e:
            # Fallback defaults if config not available
            # Don't log here to avoid recursion in logger initialization
            self.sanitize_logs = self.environment == "production"
            self.enable_emoji = self.environment != "production"
            self.mask_character = "*"

        # Patterns for sensitive data detection
        self.sensitive_patterns = [
            # Passwords
            re.compile(
                r'(["\']?password["\']?\s*[:=]\s*["\']?)([^"\'}\s,]+)', re.IGNORECASE
            ),
            re.compile(
                r'(["\']?pwd["\']?\s*[:=]\s*["\']?)([^"\'}\s,]+)', re.IGNORECASE
            ),
            re.compile(
                r'(["\']?pass["\']?\s*[:=]\s*["\']?)([^"\'}\s,]+)', re.IGNORECASE
            ),
            # API keys and tokens (handle common prefixes like sk-, pk-, etc.)
            re.compile(
                r'(["\']?(?:api_?key|token|secret)["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9+/=_-]{8,})',
                re.IGNORECASE,
            ),
            # Handle specific API key patterns with prefixes
            re.compile(
                r'(["\']?(?:api_?key|token|secret)["\']?\s*[:=]\s*["\']?[a-zA-Z0-9+-]{2,4}-?)([a-zA-Z0-9+/=_-]{16,})',
                re.IGNORECASE,
            ),
            re.compile(
                r'(["\']?(?:bearer|auth)["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9+/=_-]{20,})',
                re.IGNORECASE,
            ),
            # Database credentials
            re.compile(r"(://[^:]+:)([^@]+)(@)", re.IGNORECASE),  # URL with credentials
            re.compile(
                r'(["\']?(?:host|user|database)["\']?\s*[:=]\s*["\']?)([^"\'}\s,]+)',
                re.IGNORECASE,
            ),
            # Connection strings
            re.compile(r"(postgresql|mysql|mongodb)://[^@]+@[^/\s]+", re.IGNORECASE),
            # Email addresses (partial masking)
            re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
        ]

        # Emoji patterns
        self.emoji_patterns = [
            re.compile(r"[\U0001F600-\U0001F64F]"),  # Emoticons
            re.compile(r"[\U0001F300-\U0001F5FF]"),  # Symbols & pictographs
            re.compile(r"[\U0001F680-\U0001F6FF]"),  # Transport & map symbols
            re.compile(r"[\U0001F700-\U0001F77F]"),  # Alchemical symbols
            re.compile(r"[\U0001F780-\U0001F7FF]"),  # Geometric shapes
            re.compile(r"[\U0001F800-\U0001F8FF]"),  # Supplemental arrows-C
            re.compile(
                r"[\U0001F900-\U0001F9FF]"
            ),  # Supplemental symbols and pictographs
            re.compile(r"[\U0001FA00-\U0001FA6F]"),  # Chess symbols
            re.compile(
                r"[\U0001FA70-\U0001FAFF]"
            ),  # Symbols and pictographs extended-A
            # Common emoji patterns
            re.compile(r"[✓✗x⚠🚨⭐🔥🚀💡⚡🔔📊📈📉]"),
        ]

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with security sanitization.

        Args:
            record: Log record to format

        Returns:
            Formatted and sanitized log message
        """
        # Get base formatted message
        formatted = super().format(record)

        # Apply sanitization if enabled
        if self.sanitize_logs:
            formatted = self._sanitize_sensitive_data(formatted)

        # Remove emojis if disabled
        if not self.enable_emoji:
            formatted = self._remove_emojis(formatted)

        return formatted

    def _sanitize_sensitive_data(self, message: str) -> str:
        """
        Sanitize sensitive data in log message.

        Args:
            message: Log message to sanitize

        Returns:
            Sanitized log message
        """
        sanitized = message

        for pattern in self.sensitive_patterns:

            def replacer(match):
                groups = match.groups()
                if len(groups) >= 2:
                    # Mask the sensitive part
                    prefix = groups[0]
                    sensitive = groups[1]

                    # Keep first and last characters if long enough
                    if len(sensitive) > 4:
                        masked = (
                            sensitive[0]
                            + self.mask_character * (len(sensitive) - 2)
                            + sensitive[-1]
                        )
                    else:
                        masked = self.mask_character * len(sensitive)

                    return prefix + masked
                return match.group(0)

            sanitized = pattern.sub(replacer, sanitized)

        return sanitized

    def _remove_emojis(self, message: str) -> str:
        """
        Remove emojis from log message.

        Args:
            message: Log message to clean

        Returns:
            Message with emojis removed
        """
        cleaned = message

        for pattern in self.emoji_patterns:
            cleaned = pattern.sub("", cleaned)

        return cleaned


class SecureContextLogger:
    """
    Secure logger that provides context-aware logging with sanitization.
    """

    def __init__(self, name: str, environment: str = "development"):
        """
        Initialize secure context logger.

        Args:
            name: Logger name
            environment: Environment name
        """
        self.logger = logging.getLogger(name)
        self.environment = environment.lower()
        self._load_config()

    def _load_config(self) -> None:
        """Load secure logging configuration."""
        try:
            registry_config = get_registry_config(self.environment)
            self.sanitize_logs = registry_config.sanitize_logs_in_production
            self.log_sensitive_data = registry_config.log_sensitive_data
            self.mask_character = registry_config.security_mask_character
        except (ImportError, AttributeError, KeyError, ValueError) as e:
            # Fallback defaults if config not available
            # Don't log here to avoid recursion in logger initialization
            self.sanitize_logs = self.environment == "production"
            self.log_sensitive_data = False
            self.mask_character = "*"

    def _sanitize_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize context dictionary for logging.

        Args:
            context: Context dictionary to sanitize

        Returns:
            Sanitized context dictionary
        """
        if not self.sanitize_logs or self.log_sensitive_data:
            return context

        sanitized = {}
        sensitive_keys = {
            # Passwords and authentication
            "password",
            "pwd",
            "pass",
            "passphrase",
            # Secrets and keys (note: public keys are NOT sensitive)
            "secret",
            "api_secret",
            "client_secret",
            "webhook_secret",
            "signing_key",
            "master_key",
            "encryption_key",
            "private_key",
            # Tokens and credentials
            "token",
            "access_token",
            "refresh_token",
            "api_key",
            "api_token",
            "access_key",
            "auth",
            "authorization",
            "bearer",
            "credential",
            # Session and cookies
            "session",
            "session_id",
            "cookie",
            "csrf",
            "csrf_token",
            # OAuth and JWT
            "jwt",
            "oauth",
            "oauth_token",
            "client_id",
            # Database credentials
            "db_password",
            "db_user",
            "database_password",
            "database_url",
            "connection_string",
            # Certificates
            "certificate",
            "cert",
            "private_cert",
        }

        for key, value in context.items():
            key_lower = key.lower()
            is_sensitive = any(sensitive in key_lower for sensitive in sensitive_keys)

            if isinstance(value, dict):
                # Always recursively sanitize nested dicts
                sanitized[key] = self._sanitize_context(value)
            elif is_sensitive and isinstance(value, str):
                # Mask sensitive string values
                if len(value) > 4:
                    sanitized[key] = (
                        value[0] + self.mask_character * (len(value) - 2) + value[-1]
                    )
                else:
                    sanitized[key] = self.mask_character * len(value)
            elif "connection_string" in key_lower or "url" in key_lower:
                # Special handling for connection strings
                sanitized[key] = self._mask_connection_string(str(value))
            else:
                sanitized[key] = value

        return sanitized

    def _mask_connection_string(self, connection_string: str) -> str:
        """
        Mask connection string while preserving non-sensitive parts.

        Args:
            connection_string: Connection string to mask

        Returns:
            Masked connection string
        """
        # Pattern for connection strings with credentials
        pattern = r"(://[^:]+:)([^@]+)(@)"
        return re.sub(
            pattern,
            lambda m: m.group(1) + self.mask_character * 8 + m.group(3),
            connection_string,
        )

    def _format_context(self, context: dict[str, Any]) -> str:
        """
        Format context dictionary for logging.

        Args:
            context: Context dictionary

        Returns:
            Formatted context string
        """
        if not context:
            return ""

        sanitized = self._sanitize_context(context)
        context_parts = [f"{k}={v}" for k, v in sanitized.items()]
        return f" | {' '.join(context_parts)}"

    def debug(self, message: str, **context) -> None:
        """Log debug message with context."""
        formatted_context = self._format_context(context)
        full_message = f"{message}{formatted_context}" if formatted_context else message
        self.logger.debug(full_message)

    def info(self, message: str, **context) -> None:
        """Log info message with context."""
        formatted_context = self._format_context(context)
        full_message = f"{message}{formatted_context}" if formatted_context else message
        self.logger.info(full_message)

    def warning(self, message: str, **context) -> None:
        """Log warning message with context."""
        formatted_context = self._format_context(context)
        full_message = f"{message}{formatted_context}" if formatted_context else message
        self.logger.warning(full_message)

    def error(self, message: str, **context) -> None:
        """Log error message with context."""
        formatted_context = self._format_context(context)
        full_message = f"{message}{formatted_context}" if formatted_context else message
        self.logger.error(full_message)

    def critical(self, message: str, **context) -> None:
        """Log critical message with context."""
        formatted_context = self._format_context(context)
        full_message = f"{message}{formatted_context}" if formatted_context else message
        self.logger.critical(full_message)


def setup_secure_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_format: Optional[str] = None,
) -> None:
    """
    Setup secure logging for the application.

    Args:
        environment: Environment name
        log_level: Log level
        log_format: Optional log format
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with secure formatter
    console_handler = logging.StreamHandler()
    formatter = SecureLogFormatter(
        fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S", environment=environment
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Log setup completion
    setup_logger = SecureContextLogger("secure_logging", environment)
    setup_logger.info(
        f"Secure logging configured for {environment}",
        log_level=log_level,
        sanitize_logs=environment == "production",
        emoji_enabled=environment != "production",
    )


def get_secure_logger(
    name: str, environment: str = "development"
) -> SecureContextLogger:
    """
    Get a secure logger instance.

    Args:
        name: Logger name
        environment: Environment name

    Returns:
        SecureContextLogger instance
    """
    return SecureContextLogger(name, environment)


def sanitize_log_data(data: Any, environment: str = "development") -> Any:
    """
    Sanitize data for logging.

    Args:
        data: Data to sanitize
        environment: Environment name

    Returns:
        Sanitized data
    """
    if environment != "production":
        return data

    if isinstance(data, dict):
        logger = SecureContextLogger("sanitizer", environment)
        return logger._sanitize_context(data)
    elif isinstance(data, str):
        formatter = SecureLogFormatter(environment=environment)
        return formatter._sanitize_sensitive_data(data)
    else:
        return data
