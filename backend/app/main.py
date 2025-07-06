"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 🔧 Adjusted imports to use "app." as the top-level package
#     so that when rootDir=backend, Python can resolve modules.
from app.core.config import settings
from app.api.api_v1.api import api_router
from app.services.ml.model_service import model_service
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info(f"🌀 PORT env inside container: {os.getenv('PORT')}")
    logger.info("🔧 Initializing ML model service...")
    logger.info("🌀 [lifespan] Entering lifespan—about to init services")

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
        logger.info("🌀 [lifespan] After yield—now shutting down")
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

    @app.head("/", include_in_schema=False)
    async def root_head():
        """Root HEAD endpoint for Render probes."""
        from fastapi.responses import Response
        return Response(status_code=200)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return await model_service.get_health_status()

    @app.head("/health", include_in_schema=False)
    async def health_head():
        """Health check HEAD endpoint for Render probes."""
        from fastapi.responses import Response
        return Response(status_code=200)

    return app


# Create the application instance
app = create_application()

