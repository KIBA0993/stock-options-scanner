"""Tests for vertex_llm.py"""

from __future__ import annotations

import sys
from pathlib import Path

TRADING_DIR = Path.home() / "trading"
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

import vertex_llm as vl


def test_vertex_model_id_prefix():
    assert vl.vertex_model_id("gemini-2.5-flash") == "google/gemini-2.5-flash"
    assert vl.vertex_model_id("google/gemini-2.5-flash") == "google/gemini-2.5-flash"


def test_vertex_base_url():
    url = vl.vertex_base_url("stock-500202", "us-central1")
    assert "stock-500202" in url
    assert "us-central1" in url
    assert url.endswith("/endpoints/openapi")


def test_auth_mode_defaults(tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    assert vl.auth_mode({"credentials_path": str(sa)}) == "service_account"
    assert vl.auth_mode({"auth_mode": "adc"}) == "adc"
    assert vl.auth_mode({"auth_mode": "impersonate"}) == "impersonate"


def test_is_vertex_configured_service_account(tmp_path, monkeypatch):
    monkeypatch.setattr(vl, "_DEFAULT_ADC", tmp_path / "missing.json")
    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account"}')
    cfg = {"project_id": "stock-500202", "credentials_path": str(sa)}
    assert vl.is_vertex_configured(cfg) is True
    assert vl.is_vertex_configured({"project_id": "x"}) is False


def test_is_vertex_configured_adc(tmp_path):
    adc = tmp_path / "adc.json"
    adc.write_text('{"type":"authorized_user"}')
    cfg = {
        "project_id": "stock-500202",
        "auth_mode": "adc",
        "credentials_path": str(adc),
    }
    assert vl.is_vertex_configured(cfg) is True


def test_is_vertex_configured_impersonate():
    cfg = {
        "project_id": "stock-500202",
        "auth_mode": "impersonate",
        "service_account_email": "tradingscan@stock-500202.iam.gserviceaccount.com",
    }
    assert vl.is_vertex_configured(cfg) is True
    assert vl.is_vertex_configured({"project_id": "x", "auth_mode": "impersonate"}) is False
