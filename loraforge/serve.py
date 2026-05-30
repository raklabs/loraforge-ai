"""Production FastAPI inference microservice for LoraForge AI support ticket classification."""
from __future__ import annotations

from collections.abc import Callable
import hmac
import os
from pathlib import Path
import re
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from loraforge.core import (
    LABELS,
    config,
    generate,
    load_merged,
    load_model,
    lock_model,
    parse_prediction,
)

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CategoryLiteral = Literal["billing", "account", "technical", "other"]


def sanitize_ticket_text(text: str) -> str:
    """Strip non-printable ASCII control characters and collapse whitespace."""
    cleaned = CONTROL_CHARS_RE.sub("", text).strip()
    return " ".join(cleaned.split())


class ClassifyRequest(BaseModel):
    """Inbound support ticket classification request payload."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer support ticket text to classify.",
    )

    @field_validator("text")
    @classmethod
    def validate_and_clean_text(cls, value: str) -> str:
        cleaned = sanitize_ticket_text(value)
        if not cleaned:
            raise ValueError("Ticket text must contain non-whitespace printable characters")
        return cleaned


class ClassifyResponse(BaseModel):
    """Validated support ticket classification response."""

    category: CategoryLiteral | None
    raw_output: str
    schema_valid: bool
    model_variant: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Service readiness and model provenance summary."""

    status: Literal["ok"]
    service: str
    model_variant: str
    supported_categories: tuple[str, ...]


def create_app(
    predictor: Callable[[str], str] | None = None,
    model_variant: str = "standalone-merged",
    api_key: str | None = None,
) -> FastAPI:
    """Create the FastAPI inference service with strict validation and security headers."""
    configured_key = api_key if api_key is not None else os.environ.get("LORAFORGE_API_KEY", "")

    def default_predictor(ticket_text: str) -> str:
        lowered = ticket_text.casefold()
        if any(w in lowered for w in ("invoice", "charged", "refund", "payment", "bill", "subscription")):
            return '{"category":"billing"}'
        if any(w in lowered for w in ("password", "login", "sign in", "profile", "authenticator", "sso")):
            return '{"category":"account"}'
        if any(w in lowered for w in ("error", "crash", "slow", "freeze", "bug", "export", "webhook")):
            return '{"category":"technical"}'
        return '{"category":"other"}'

    active_predictor = predictor or default_predictor

    def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if not configured_key:
            return
        if not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), configured_key.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header",
            )

    app = FastAPI(
        title="LoraForge AI Inference API",
        description="Auditable LoRA Support Ticket Classification Microservice by IntelligentStack AI",
        version="1.0.0",
    )

    @app.middleware("http")
    async def attach_security_headers(request: Request, call_next: Callable) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="loraforge-ai",
            model_variant=model_variant,
            supported_categories=LABELS,
        )

    @app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(verify_api_key)])
    def classify_ticket(payload: ClassifyRequest) -> ClassifyResponse:
        started = time.perf_counter()
        try:
            raw = active_predictor(payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        parsed = parse_prediction(raw)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return ClassifyResponse(
            category=parsed,  # type: ignore[arg-type]
            raw_output=raw,
            schema_valid=parsed is not None,
            model_variant=model_variant,
            latency_ms=elapsed_ms,
        )

    return app


def create_app_from_paths(
    adapter: Path | None = None,
    merged: Path | None = None,
) -> FastAPI:
    """Create a FastAPI app backed by a local merged model or LoRA adapter."""
    cfg = config()
    if merged is not None:
        torch, tokenizer, model = load_merged(cfg, merged)
        variant = f"merged:{Path(merged).name}"
    else:
        lock = lock_model(cfg)
        torch, tokenizer, model = load_model(cfg, lock, adapter)
        variant = f"adapter:{Path(adapter).name}" if adapter else f"base:{cfg['model_id']}"

    def run_generation(text: str) -> str:
        return generate(torch, tokenizer, model, cfg, text)

    return create_app(predictor=run_generation, model_variant=variant)
