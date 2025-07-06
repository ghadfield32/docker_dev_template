"""Main API router that combines all endpoints."""

from fastapi import APIRouter

# 🔧 Adjusted imports to use relative paths for Render deployment
from app.api.api_v1.endpoints import iris, cancer, models, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(iris.router, prefix="/iris", tags=["Iris Predictions"])
api_router.include_router(cancer.router, prefix="/cancer", tags=["Cancer Predictions"])
api_router.include_router(models.router, prefix="/models", tags=["Model Management"])
