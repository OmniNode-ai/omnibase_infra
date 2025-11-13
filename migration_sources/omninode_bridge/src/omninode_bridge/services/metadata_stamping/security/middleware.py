"""
Security middleware for O.N.E. v0.1 protocol.

This module provides security middleware for request filtering,
trust validation, and signature verification.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .signature_validator import SignatureValidator
from .trust_zones import TrustContext, TrustLevel, TrustZone, TrustZoneManager

logger = logging.getLogger(__name__)


class ONESecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for O.N.E. protocol compliance.

    Provides trust zone assignment, signature validation,
    and security policy enforcement.
    """

    def __init__(self, app, enable_security: bool = True):
        """
        Initialize security middleware.

        Args:
            app: FastAPI application
            enable_security: Whether to enable security checks
        """
        super().__init__(app)
        self.enable_security = enable_security
        self.trust_zone_manager = TrustZoneManager()
        self.signature_validator = SignatureValidator()

        # Paths that bypass security
        self.bypass_paths = {
            "/health",
            "/metrics",
            "/",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

        logger.info(f"ONESecurityMiddleware initialized (enabled: {enable_security})")

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request through security middleware.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response: Processed response
        """
        # Skip security for disabled middleware
        if not self.enable_security:
            response = await call_next(request)
            return response

        # Skip security for bypass paths
        if request.url.path in self.bypass_paths:
            response = await call_next(request)
            return response

        try:
            # Extract client information
            client_host = "unknown"
            if request.client:
                client_host = request.client.host

            # Assign trust zone
            trust_zone = self.trust_zone_manager.assign_trust_zone(client_host)

            # Determine operation type
            operation = self._get_operation_type(request.method)

            # Get required trust level
            required_trust = self.trust_zone_manager.get_required_trust_level(
                trust_zone, operation
            )

            # Validate trust requirements
            trust_context = await self._validate_trust_requirements(
                request, required_trust, trust_zone
            )

            if not trust_context:
                logger.warning(
                    f"Trust validation failed for {client_host}: "
                    f"zone={trust_zone}, operation={operation}, "
                    f"required={required_trust}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient trust level for {operation} operation in {trust_zone}",
                )

            # Add trust context to request state
            request.state.trust_context = trust_context

            # Process request
            response = await call_next(request)

            # Add security headers to response
            response.headers["X-Trust-Zone"] = trust_zone.value
            response.headers["X-Trust-Level"] = trust_context.trust_level.value

            return response

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # Allow request through on unexpected errors
            response = await call_next(request)
            return response

    def _get_operation_type(self, method: str) -> str:
        """
        Get operation type from HTTP method.

        Args:
            method: HTTP method

        Returns:
            str: Operation type (read, write, delete)
        """
        if method in ["GET", "HEAD", "OPTIONS"]:
            return "read"
        elif method in ["POST", "PUT", "PATCH"]:
            return "write"
        elif method == "DELETE":
            return "delete"
        else:
            return "write"  # Default to write for safety

    async def _validate_trust_requirements(
        self, request: Request, required_trust: TrustLevel, trust_zone: TrustZone
    ) -> Optional[TrustContext]:
        """
        Validate trust requirements for request.

        Args:
            request: Incoming request
            required_trust: Required trust level
            trust_zone: Assigned trust zone

        Returns:
            TrustContext: Valid trust context or None
        """
        # For UNVERIFIED level, always allow
        if required_trust == TrustLevel.UNVERIFIED:
            return TrustContext(
                trust_level=TrustLevel.UNVERIFIED, trust_zone=trust_zone
            )

        # Check for signature headers
        headers = dict(request.headers)
        signature = headers.get("x-onf-signature")
        public_key = headers.get("x-onf-public-key")
        key_id = headers.get("x-onf-key-id")

        # No signature provided
        if not signature:
            logger.debug("No signature provided in request")
            return None

        # Get public key if only key ID provided
        if key_id and not public_key:
            public_key = self.signature_validator.get_public_key(key_id)
            if not public_key:
                logger.warning(f"Unknown key ID: {key_id}")
                return None

        if not public_key:
            logger.warning("No public key available for signature validation")
            return None

        # Validate signature
        try:
            # Get request body for signature validation
            body = await request.body()

            # Verify signature
            if self.signature_validator.verify_ed25519_signature(
                body, signature, public_key
            ):
                # Determine trust level based on key trust
                trust_level = TrustLevel.SIGNED
                if key_id and self.signature_validator.is_trusted_key(key_id):
                    trust_level = TrustLevel.VERIFIED

                # Check if trust level meets requirement
                if self.trust_zone_manager.validate_trust_level(
                    required_trust, trust_level
                ):
                    logger.debug(
                        f"Trust validation successful: "
                        f"level={trust_level}, zone={trust_zone}"
                    )
                    return TrustContext(
                        trust_level=trust_level,
                        trust_zone=trust_zone,
                        signature=signature,
                        public_key=public_key,
                    )
                else:
                    logger.warning(
                        f"Trust level {trust_level} insufficient for "
                        f"requirement {required_trust}"
                    )
                    return None

            else:
                logger.warning("Signature validation failed")
                return None

        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return None

    def add_trusted_key(self, key_id: str, public_key: str):
        """
        Add trusted public key.

        Args:
            key_id: Key identifier
            public_key: Base64 encoded public key
        """
        self.signature_validator.add_trusted_public_key(key_id, public_key)

    def add_zone_mapping(self, pattern: str, zone: TrustZone):
        """
        Add custom zone mapping pattern.

        Args:
            pattern: Address pattern
            zone: Trust zone to assign
        """
        self.trust_zone_manager.add_zone_assignment(pattern, zone)

    def update_bypass_paths(self, paths: set):
        """
        Update paths that bypass security.

        Args:
            paths: Set of paths to bypass
        """
        self.bypass_paths.update(paths)
        logger.info(f"Updated bypass paths: {self.bypass_paths}")
