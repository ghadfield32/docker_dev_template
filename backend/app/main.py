"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.api_v1.api import api_router
from app.services.ml.model_service import model_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info("🚀 Starting ML Full Stack API...")

    try:
        # Initialize ML service
        logger.info("🔧 Initializing ML model service...")
        await model_service.initialize()
        logger.info("✅ ML model service initialized successfully")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize ML service: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down ML Full Stack API...")
        await model_service.cleanup()


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Full Stack ML API with Iris and Cancer prediction models",
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "ML Full Stack API",
            "version": settings.VERSION,
            "docs_url": f"{settings.API_V1_STR}/docs",
            "health_url": f"{settings.API_V1_STR}/health"
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return await model_service.get_health_status()

    return app


# Create the application instance
app = create_application()
