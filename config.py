"""
config.py
---------
Application-wide settings loaded from environment variables / .env file.
Uses Pydantic v2 BaseSettings for automatic validation and type coercion.
"""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://governance:governance@localhost:5432/governance_db",
        description="SQLAlchemy asyncpg connection string",
    )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # ------------------------------------------------------------------
    # LLM Provider  (lightweight models only)
    # ------------------------------------------------------------------
    LLM_PROVIDER: str = Field(
        default="gemini",
        description="'gemini' (Gemini 2.0 Flash Lite) or 'groq' (Llama 3 8B)",
    )

    # Google Gemini 2.0 Flash Lite  — free-tier quota, ~1 s latency
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google AI Studio API key for Gemini 2.0 Flash Lite",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash-lite",
        description="Gemini model name",
    )

    # Groq  Llama 3 8B  — optional alternative, ~200 tokens/sec
    GROQ_API_KEY: str = Field(default="", description="Groq API key for Llama 3 8B inference")
    GROQ_MODEL: str = Field(default="llama3-8b-8192", description="Groq model name")

    # Mistral Codestral
    MISTRAL_API_KEY: str = Field(default="", description="Mistral AI API key for Codestral")
    MISTRAL_MODEL: str = Field(default="codestral-latest", description="Mistral model name")

    # ------------------------------------------------------------------
    # Governance Static Rules (Layer 1 circuit breakers)
    # ------------------------------------------------------------------
    MAX_DISCOUNT_PCT: float = Field(
        default=20.0,
        gt=0,
        le=100,
        description="Maximum allowed discount percentage before auto-reject",
    )
    MAX_PRICE_INCREASE_PCT: float = Field(
        default=50.0,
        gt=0,
        description="Maximum allowed price increase percentage before auto-reject",
    )
    MIN_ABSOLUTE_PRICE: float = Field(
        default=0.50,
        gt=0,
        description="Proposed price must be >= this value (prevents $0.01 drops)",
    )

    # ------------------------------------------------------------------
    # LLM Routing Thresholds
    # ------------------------------------------------------------------
    AI_HIGH_RISK_THRESHOLD: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Confidence score below which a request is escalated to HITL",
    )
    AI_AUTO_APPROVE_THRESHOLD: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence score at or above which a request is auto-approved",
    )

    # ------------------------------------------------------------------
    # Sentiment Circuit Breaker
    # ------------------------------------------------------------------
    ESSENTIAL_GOODS_CATEGORIES: list[str] = Field(
        default=["food", "medicine", "water", "fuel", "baby"],
        description="Product categories subject to the crisis price lock",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_TITLE:   str = "Supervising AI Governance Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG:       bool = Field(default=False)
    CORS_ORIGINS: list[str] = Field(default=["*"])

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"gemini", "groq", "mock", "mistral"}
        if v.lower() not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {allowed}")
        return v.lower()


# Singleton instance — import this everywhere
settings = Settings()
