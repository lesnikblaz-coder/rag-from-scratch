from fastapi import FastAPI
from google import genai
from contextlib import asynccontextmanager

from src.config import settings
from src.db.base import async_engine
from src.api.routes.health import router as health
from src.api.routes.rag import router as rag


@asynccontextmanager
async def lifespan(_app: FastAPI):
    aclient = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=genai.types.HttpOptions(
            retry_options=genai.types.HttpRetryOptions(
                attempts=settings.API_RETRY_ATTEMPTS,
                initial_delay=settings.API_RETRY_INITIAL_DELAY,
                max_delay=settings.API_RETRY_MAX_DELAY,
                exp_base=settings.API_RETRY_EXP_BASE,
                jitter=settings.API_RETRY_JITTER
            )
        )
    ).aio

    _app.state.aclient = aclient

    try:
        yield
    finally:
        await aclient.aclose()
        await async_engine.dispose()


app = FastAPI(lifespan=lifespan)


app.include_router(health)
app.include_router(rag)