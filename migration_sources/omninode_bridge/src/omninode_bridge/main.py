"""Main entry point for OmniNode Bridge HookReceiver service."""

import os

import uvicorn

from omninode_bridge.services.hook_receiver import create_app


def main() -> None:
    """Main entry point for the HookReceiver service."""
    # Get configuration from environment variables
    host = os.getenv("HOOK_RECEIVER_HOST", "127.0.0.1")
    port = int(os.getenv("HOOK_RECEIVER_PORT", "8001"))
    workers = int(os.getenv("HOOK_RECEIVER_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    # Create the FastAPI app
    app = create_app()

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
