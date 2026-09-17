"""API authentication for governance-sensitive endpoints.

Uses a simple API key check via X-API-Key header.

Security defaults (Phase 7):
- NO default/development key is embedded. ``GOVERNANCE_API_KEYS`` must be
  configured explicitly (comma-separated) or the endpoints fail closed with
  503. This removes the old ``governance-dev-key`` fallback so no build
  ships with a known credential.
"""
import os
from typing import Optional

from fastapi import Header, HTTPException


def get_allowed_keys() -> set:
    """Get the set of allowed governance API keys from the environment."""
    raw = os.environ.get("GOVERNANCE_API_KEYS", "")
    if raw:
        return {k.strip() for k in raw.split(",") if k.strip()}
    return set()


def require_governance_key(x_api_key: Optional[str] = Header(default=None)):
    """FastAPI dependency: require a valid, explicitly configured API key.

    Apply to sensitive endpoints like /governance/evaluate, /governance/approve,
    /governance/reject, /runtime/rollback, /amendment/propose, /evaluation/run,
    /memory/lesson*, and /steward/analyze.
    """
    allowed = get_allowed_keys()
    if not allowed:
        raise HTTPException(
            status_code=503,
            detail="Governance API keys not configured; set GOVERNANCE_API_KEYS",
        )
    if x_api_key is None or x_api_key not in allowed:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing governance API key. "
                   "Include X-API-Key header.",
        )