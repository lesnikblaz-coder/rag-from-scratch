import hmac

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader


from src.config import settings

api_key_header = APIKeyHeader(
    name="X-API-Key"
)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not hmac.compare_digest(api_key, settings.API_SECRET):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )

    return api_key