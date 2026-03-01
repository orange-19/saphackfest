"""
main.py
-------
FastAPI application entry point for the Supervising AI Governance Agent.
Handles startup/shutdown lifecycle: DB init, Redis connection, router wiring.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from db.database import close_db, init_db, setup_engine
from api.routes import governance_router, hitl_router

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Redis client (populated at startup)
# ---------------------------------------------------------------------------
redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """FastAPI dependency that returns the shared Redis client."""
    if redis_client is None:
        raise RuntimeError("Redis not initialized. App startup may have failed.")
    return redis_client


# ---------------------------------------------------------------------------
# Lifespan: startup + shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager that wraps the full app lifespan."""
    global redis_client

    # ---- STARTUP ----
    logger.info("Starting Supervising AI Governance Agent...")

    # 1. Initialize database engine and run schema.sql
    setup_engine(settings.DATABASE_URL)
    try:
        await init_db()
        logger.info("✅ PostgreSQL: schema ready")
    except Exception as e:
        logger.error(f"❌ PostgreSQL init failed: {e}")
        raise

    # 2. Connect to Redis (falls back to in-memory FakeRedis if unavailable)
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis_client.ping()
        logger.info("✅ Redis: connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}). Using in-memory FakeRedis.")
        from services.fake_redis import FakeRedis
        redis_client = FakeRedis()

    # Expose Redis on app.state so route handlers can access it via request.app.state.redis
    app.state.redis = redis_client

    # 3. Seed Layer 1 static rules into Redis (Phase 2 will read these)
    if redis_client:
        await _seed_static_rules(redis_client)

    logger.info(f"🚀 {settings.APP_TITLE} v{settings.APP_VERSION} is ready")
    yield

    # ---- SHUTDOWN ----
    logger.info("Shutting down...")
    if redis_client:
        await redis_client.aclose()
    await close_db()
    logger.info("Shutdown complete.")


async def _seed_static_rules(r: aioredis.Redis) -> None:
    """
    Write static governance rules to Redis so Layer 1 can evaluate them in O(1).
    These are also configurable via env vars — Redis acts as a fast cache.
    """
    rules = {
        "rule:max_discount_pct":     str(settings.MAX_DISCOUNT_PCT),
        "rule:max_increase_pct":     str(settings.MAX_PRICE_INCREASE_PCT),
        "rule:min_absolute_price":   str(settings.MIN_ABSOLUTE_PRICE),
        "rule:essential_categories": ",".join(settings.ESSENTIAL_GOODS_CATEGORIES),
    }
    await r.mset(rules)
    logger.info(f"✅ Redis: seeded {len(rules)} static governance rules")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "Centralized AI governance layer that intercepts, validates, and logs "
        "automated pricing decisions before they go live."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(governance_router)
app.include_router(hitl_router)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred. Check server logs."},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Quick liveness probe for load balancers and monitoring."""
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "redis": "connected" if redis_ok else "unavailable",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": (
            settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini"
            else settings.MISTRAL_MODEL if settings.LLM_PROVIDER == "mistral"
            else settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq"
            else "mock (rule-based)"
        ),
    }


@app.get("/", tags=["System"])
async def root() -> dict:
    return {"message": f"Welcome to {settings.APP_TITLE}", "docs": "/docs"}
