"""
Cloudflare Workers entry point for TheLawSays backend.
This file serves as a bridge between Cloudflare Workers and the existing FastAPI application.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path so we can import from api/
sys.path.append(str(Path(__file__).parent.parent))

# Import the FastAPI app from the existing codebase
try:
    from api.main import app
except ImportError as e:
    print(f"Failed to import FastAPI app: {e}")
    app = None

def handle_request(request, env, ctx):
    """
    Main request handler for Cloudflare Workers.
    Converts Cloudflare request to ASGI format for FastAPI.
    """
    if app is None:
        return Response.json({"error": "Application not initialized"}, status=500)

    # Convert Cloudflare request to ASGI format
    asgi_scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": request.method,
        "path": request.url.pathname,
        "raw_path": request.url.pathname.encode(),
        "query_string": request.url.searchParams.toString().encode(),
        "root_path": "",
        "headers": [[k.encode(), v.encode()] for k, v in request.headers.entries()],
        "server": ("localhost", 8000),
        "client": ("127.0.0.1", 0),
    }

    # Add environment variables to the request context
    asgi_scope["env"] = env

    # Handle the request through FastAPI
    return handle_asgi_request(app, asgi_scope, request.body)

async def handle_asgi_request(app, scope, body):
    """
    Handle ASGI request through FastAPI application.
    This is a simplified ASGI to Workers adapter.
    """
    # Create ASGI receive callable
    async def receive():
        return {
            "type": "http.request",
            "body": body or b"",
            "more_body": False,
        }

    # Create ASGI send callable
    messages = []
    async def send(message):
        messages.append(message)

    # Call the ASGI application
    await app(scope, receive, send)

    # Process ASGI messages and create Cloudflare Response
    status_code = 200
    headers = {}
    body_parts = []

    for message in messages:
        if message["type"] == "http.response.start":
            status_code = message["status"]
            headers = {k.decode(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            body_parts.append(message.get("body", b""))

    # Create and return Cloudflare Response
    response_body = b"".join(body_parts)
    return Response(response_body, status=status_code, headers=headers)

# Export the handler function for Cloudflare Workers
# This follows the Cloudflare Workers Python runtime pattern
def on_fetch(request, env, ctx):
    """
    Cloudflare Workers fetch handler.
    This function is automatically called by Cloudflare Workers runtime.
    """
    return handle_request(request, env, ctx)

# Export the handler
__all__ = ["on_fetch"]
