#!/usr/bin/env python3
"""
vertex_llm.py — Google Cloud Vertex AI (Gemini) via OpenAI-compatible endpoint.

Auth modes (llm.auth_mode):
  - service_account: JSON key file (llm.credentials_path)
  - adc: Application Default Credentials (user OAuth or GCE metadata)
  - impersonate: ADC + impersonate llm.service_account_email (no key file)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CLOUD_PLATFORM = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_ADC = Path.home() / ".config/gcloud/application_default_credentials.json"


def vertex_model_id(model: str) -> str:
    """Vertex OpenAI-compat expects e.g. google/gemini-2.5-flash."""
    model = (model or "gemini-2.5-flash").strip()
    if model.startswith("google/"):
        return model
    if model.startswith("gemini-"):
        return f"google/{model}"
    return model


def vertex_base_url(project_id: str, location: str = "us-central1") -> str:
    loc = location or "us-central1"
    return (
        f"https://{loc}-aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{loc}/endpoints/openapi"
    )


def _resolve_credentials_path(cfg: dict) -> Optional[str]:
    raw = cfg.get("credentials_path") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p)
    # Relative paths: resolve against TRADING_DIR or common mount points
    for root in (
        os.environ.get("TRADING_DIR"),
        "/data/trading",
        str(Path.home() / "trading"),
    ):
        if not root:
            continue
        candidate = Path(root) / p
        if candidate.is_file():
            return str(candidate)
    return None


def auth_mode(cfg: dict) -> str:
    mode = (cfg.get("auth_mode") or "").strip().lower()
    if mode in ("service_account", "adc", "impersonate"):
        return mode
    if _resolve_credentials_path(cfg):
        return "service_account"
    return "adc"


def is_vertex_configured(cfg: dict) -> bool:
    if not cfg.get("project_id"):
        return False
    mode = auth_mode(cfg)
    if mode == "service_account":
        return _resolve_credentials_path(cfg) is not None
    if mode == "impersonate":
        return bool(cfg.get("service_account_email"))
    # adc: env var, explicit path, or default gcloud ADC file
    if _resolve_credentials_path(cfg):
        return True
    return _DEFAULT_ADC.is_file()


def _load_credentials(cfg: dict):
    import google.auth
    from google.oauth2 import service_account

    mode = auth_mode(cfg)
    cred_path = _resolve_credentials_path(cfg)

    if mode == "service_account":
        if not cred_path:
            raise FileNotFoundError(
                "Vertex service_account auth: set llm.credentials_path or "
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
        return service_account.Credentials.from_service_account_file(
            cred_path, scopes=[_CLOUD_PLATFORM],
        )

    if cred_path:
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cred_path)

    source, _ = google.auth.default(scopes=[_CLOUD_PLATFORM])

    if mode == "impersonate":
        from google.auth import impersonated_credentials

        target = cfg.get("service_account_email", "").strip()
        if not target:
            raise ValueError(
                "Vertex impersonate auth: set llm.service_account_email"
            )
        return impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=target,
            target_scopes=[_CLOUD_PLATFORM],
            lifetime=3600,
        )

    return source


def _access_token(cfg: dict) -> str:
    import google.auth.transport.requests

    creds = _load_credentials(cfg)
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def vertex_chat(
    prompt: str,
    system: str,
    cfg: dict,
    *,
    stream: bool = False,
) -> str:
    """Chat completion against Vertex Gemini (OpenAI-compatible API)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai not installed — run: pip install openai") from exc

    project_id = cfg.get("project_id", "")
    location = cfg.get("location", "us-central1")
    model = vertex_model_id(cfg.get("model", "gemini-2.5-flash"))
    max_tokens = int(cfg.get("max_tokens", 4096))

    client = OpenAI(
        base_url=vertex_base_url(project_id, location),
        api_key=_access_token(cfg),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    if stream:
        chunks: list[str] = []
        with client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=messages, stream=True,
        ) as stream_resp:
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    chunks.append(delta)
        return "".join(chunks)

    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens, messages=messages,
    )
    return resp.choices[0].message.content or ""


def vertex_vision(
    prompt: str,
    image_b64: str,
    media_type: str,
    cfg: dict,
) -> str:
    """Vision call for distill chart analysis."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai not installed — run: pip install openai") from exc

    project_id = cfg.get("project_id", "")
    location = cfg.get("location", "us-central1")
    model = vertex_model_id(cfg.get("model", "gemini-2.5-flash"))

    client = OpenAI(
        base_url=vertex_base_url(project_id, location),
        api_key=_access_token(cfg),
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=min(int(cfg.get("max_tokens", 4096)), 1024),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{image_b64}",
                }},
            ],
        }],
    )
    return resp.choices[0].message.content or ""
