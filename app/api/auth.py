"""API authentication for governance-sensitive endpoints.

Uses a simple API key check via X-API-Key header.
In development, a default key is available; in production, keys must be configured.
"""
import os
from typing import Optional

from fastapi import Header, HTTPException


def get_allowed_keys() -> set:
    """Get the set of allowed governance API keys from environment."""
    env = os.environ.get("APP_ENV", "development")
    raw = os.environ.get("GOVERNANCE_API_KEYS", "")
    if raw:
        return {k.strip() for k in raw.split(",") if k.strip()}
    if env != "production":
        return {"governance-dev-key"}
    return set()


def require_governance_key(x_api_key: Optional[str] = Header(default=None)):
    """FastAPI dependency: require a valid governance API key.

    Apply to sensitive endpoints like /governance/approve, /governance/reject,
    and /runtime/rollback.
    """
    allowed = get_allowed_keys()
    if not allowed:
        raise HTTPException(
            status_code=503,
            detail="Governance API keys not configured",
        )
    if x_api_key is None or x_api_key not in allowed:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing governance API key. "
                   "Include X-API-Key header.",
        )
