"""Security middleware for FastAPI application."""

from __future__ import annotations

import logging
from typing import List

from fastapi import Request, Response
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Configure rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",  # Use memory for now, can be changed to Redis later
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://api.openai.com; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp

        # HSTS (HTTP Strict Transport Security) - only in production
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests for security monitoring."""

    async def dispatch(self, request: Request, call_next):
        # Log request details for security monitoring
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} with UA: {user_agent[:100]}..."
        )

        response = await call_next(request)

        # Log response status
        logger.info(f"Response: {response.status_code} for {request.url.path}")

        return response


def setup_security_middleware(app):
    """Configure all security middleware for the FastAPI application."""

    # Rate limiting middleware
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # HTTPS redirect middleware (only in production)
    if settings.enable_https_redirect and settings.environment == "production":
        app.add_middleware(HTTPSRedirectMiddleware)

    # Trusted host middleware
    trusted_hosts = [host.strip() for host in settings.trusted_hosts.split(",") if host.strip()]
    if trusted_hosts and settings.environment == "production":
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

    logger.info("Security middleware configured successfully")


# Rate limiting decorators for specific endpoints
def chat_rate_limit():
    """Rate limit decorator for chat endpoints."""
    return limiter.limit(f"{settings.rate_limit_chat_requests}/minute")


def feedback_rate_limit():
    """Rate limit decorator for feedback endpoints."""
    return limiter.limit(f"{settings.rate_limit_feedback_requests}/hour")
