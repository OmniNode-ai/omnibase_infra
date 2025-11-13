"""Main entry point for Model Metrics API service."""

import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .services.model_metrics_api import create_metrics_app
from .services.smart_responder_integration import smart_responder_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        (
            logging.FileHandler("/app/logs/model_metrics.log", mode="a")
            if os.path.exists("/app/logs")
            else logging.StreamHandler(sys.stdout)
        ),
    ],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Model Metrics API service...")

    try:
        # Initialize Smart Responder client
        logger.info("Initializing Smart Responder Chain integration...")

        # Health check AI lab nodes on startup
        async with smart_responder_client:
            health_status = await smart_responder_client.health_check_lab_nodes()
            logger.info(f"AI Lab health check: {health_status['overall_status']}")
            logger.info(
                f"Healthy nodes: {health_status['healthy_nodes']}/{health_status['total_nodes']}",
            )

            for node_name, node_status in health_status["nodes"].items():
                if node_status["status"] == "healthy":
                    logger.info(
                        f"✅ {node_name} ({node_status['host']}): {node_status['available_models']} models available",
                    )
                else:
                    logger.warning(
                        f"❌ {node_name} ({node_status['host']}): {node_status['status']} - {node_status.get('error', 'Unknown error')}",
                    )

        logger.info("Model Metrics API service startup completed")
        yield

    except Exception as e:
        logger.error(f"Failed to start Model Metrics API service: {e}")
        raise
    finally:
        logger.info("Model Metrics API service shutdown completed")


def create_app() -> FastAPI:
    """Create the main FastAPI application."""

    # Create the metrics API app
    app = create_metrics_app()

    # Update app metadata for the combined service
    app.title = "OmniNode Bridge - AI Lab Model Metrics"
    app.description = """
    Advanced AI Lab model performance tracking and intelligent routing system.

    ## Features
    - **Intelligent Model Selection**: Automatically routes tasks to optimal models based on performance history
    - **Real-time Metrics**: Tracks latency, quality, success rates, and context efficiency
    - **Model Comparison**: Compare performance across different models and tiers
    - **AI Lab Integration**: Full integration with 4-node AI lab infrastructure
    - **Smart Escalation**: Automatically escalates failed tasks to higher-tier models
    - **Performance Analytics**: Comprehensive performance reports and trend analysis

    ## AI Lab Infrastructure
    - **Mac Studio M2 Ultra** (192.168.86.200): 192GB RAM, LangGraph orchestration, 70B+ models
    - **AI PC RTX 5090** (192.168.86.201): GPU acceleration, inference optimization
    - **Mac Mini M4** (192.168.86.101): Balanced compute, medium models
    - **MacBook Air M3** (192.168.86.102): Lightweight compute, small models

    ## Smart Responder Chain
    - **8 Model Tiers**: From 3B (tiny) to 70B+ (huge) models
    - **Automatic Escalation**: Failed tasks automatically escalated to higher tiers
    - **Health Monitoring**: Real-time monitoring of all lab nodes
    - **Load Balancing**: Intelligent distribution across available resources
    """
    app.version = "0.2.0"
    app.lifespan = lifespan

    return app


def main():
    """Main entry point for the Model Metrics API service."""

    # Configuration
    host = os.getenv("MODEL_METRICS_HOST", "127.0.0.1")
    port = int(os.getenv("MODEL_METRICS_PORT", "8005"))
    workers = int(os.getenv("MODEL_METRICS_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info(f"Starting Model Metrics API service on {host}:{port}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Workers: {workers}")

    # Create app
    app = create_app()

    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        loop="asyncio",
    )

    # Create and run server
    server = uvicorn.Server(config)

    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
