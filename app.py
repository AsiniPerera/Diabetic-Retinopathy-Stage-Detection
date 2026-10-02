"""Streamlit Cloud entry point that shows the same website as viva_server.py.

index.html and its CSS/JS are combined into one page and rendered as a
Streamlit component. streamlit_bridge.js forwards the page's /api requests
here, and they are answered by the same handler the web server uses.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from model_inference import TrainedPredictor
from viva_server import API_ENDPOINTS, handle_api


ROOT = Path(__file__).resolve().parent
COMPONENT_DIR = Path(tempfile.gettempdir()) / "retiscreen_component"

# Hide Streamlit's own page furniture so only the website is visible.
STREAMLIT_CHROME_CSS = """
<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], footer { display: none !important; }
html, body, .stApp, [data-testid="stMain"] { background: #050607; overflow: hidden; }
[data-testid="stMainBlockContainer"], .block-container { padding: 0 !important; max-width: none !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
iframe { display: block; border: 0; }
</style>
"""


@st.cache_resource
def get_predictor() -> TrainedPredictor:
    return TrainedPredictor()


@st.cache_resource
def get_website_component():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    page = re.sub(
        r'<link rel="stylesheet" href="([\w.-]+\.css)(?:\?[^"]*)?">',
        lambda match: f"<style>\n{(ROOT / match[1]).read_text(encoding='utf-8')}\n</style>",
        page,
    )
    page = re.sub(r'\s*<script src="script\.js[^"]*" defer></script>', "", page)
    scripts = "".join(
        f"<script>\n{(ROOT / name).read_text(encoding='utf-8')}\n</script>\n"
        for name in ("streamlit_bridge.js", "script.js")
    )
    page = page.replace("</body>", scripts + "</body>")
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    (COMPONENT_DIR / "index.html").write_text(page, encoding="utf-8")
    return components.declare_component("retiscreen_website", path=str(COMPONENT_DIR))


def answer_request(request: dict) -> dict:
    endpoint = request.get("endpoint")
    if endpoint in API_ENDPOINTS:
        payload = base64.b64decode(request.get("data", ""))
        filename = Path(request.get("filename") or "upload.png").name
        status, body = handle_api(get_predictor(), endpoint, payload, filename)
    else:
        status, body = 404, {"error": "Unknown endpoint."}
    if isinstance(body, dict):
        content_type, raw = "application/json; charset=utf-8", json.dumps(body, separators=(",", ":")).encode("utf-8")
    else:
        content_type, raw = "application/pdf", body
    return {
        "id": request["id"],
        "status": status,
        "content_type": content_type,
        "body": base64.b64encode(raw).decode("ascii"),
    }


def main() -> None:
    st.set_page_config(
        page_title="RetiScreen AI · Retinal image analysis",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(STREAMLIT_CHROME_CSS)

    website = get_website_component()
    request = website(response=st.session_state.get("api_response"), key="website", default=None)
    if request and request.get("id") != st.session_state.get("api_request_id"):
        st.session_state["api_request_id"] = request["id"]
        st.session_state["api_response"] = answer_request(request)
        st.rerun()


if __name__ == "__main__":
    main()
