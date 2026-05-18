from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Status report for a single downstream dependency (database, parser, etc.).

    Used as the value type in HealthResponse.services — one instance per
    service being monitored. message is optional so a healthy service can
    omit it, while an unhealthy one can include a human-readable reason.
    """

    status: str = Field(..., description="Service status", example="healthy")
    message: Optional[str] = Field(None, description="Status message", example="Connected successfully")


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint.

    Returns two levels of health information:
    - Top-level: overall API status, version, environment, and service name.
    - Per-service (optional): a dict of ServiceStatus entries keyed by service
      name (e.g. "database", "pdf_parser"). services is None on a lightweight
      liveness check and populated on a full readiness check that probes
      each dependency.

    json_schema_extra provides an OpenAPI example so the Swagger UI renders
    a realistic response without needing a live request.
    """

    status: str = Field(..., description="Overall health status", example="ok")
    version: str = Field(..., description="Application version", example="0.1.0")
    environment: str = Field(..., description="Deployment environment", example="development")
    service_name: str = Field(..., description="Service identifier", example="rag-api")
    services: Optional[Dict[str, ServiceStatus]] = Field(None, description="Individual service statuses")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "rag-api",
                "services": {
                    "database": {"status": "healthy", "message": "Connected successfully"},
                    "pdf_parser": {"status": "healthy", "message": "Docling parser ready"},
                },
            }
        }
