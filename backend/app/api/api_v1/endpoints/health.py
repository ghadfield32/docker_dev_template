"""Health endpoints at /api/v1/health for frontend and Render compatibility."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from ....schemas.common import HealthResponse
from ....services.ml.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=Dict[str, Any], tags=["Health"])
async def api_v1_health():
    """
    GET /api/v1/health - Detailed JSON health report.
    
    This endpoint provides comprehensive health information including:
    - Overall service status
    - Individual model status
    - MLflow connection status
    - Model metrics
    
    Used by frontend applications and monitoring systems.
    """
    try:
        return await model_service.get_health_status()
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.head("/health", include_in_schema=False)
async def api_v1_health_head():
    """
    HEAD /api/v1/health - Lightweight probe for load balancers.
    
    Returns only HTTP status code without response body.
    Used by Render health probes and other monitoring systems
    that only need to verify service availability.
    """
    try:
        # Just check if the service is initialized
        health_status = await model_service.get_health_status()
        # Return 200 if service is running, regardless of model status
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Health check HEAD error: {e}")
        return Response(status_code=503)  # Service Unavailable 
