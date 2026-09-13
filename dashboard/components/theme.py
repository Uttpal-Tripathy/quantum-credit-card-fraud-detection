"""Shared dark cybersecurity/financial-security theme for every dashboard page."""

from __future__ import annotations

import streamlit as st

DARK_CSS = """
<style>
:root {
    --qgfda-bg: #0b0f19;
    --qgfda-panel: #131a29;
    --qgfda-border: #1f2937;
    --qgfda-accent: #22d3ee;
    --qgfda-accent2: #a855f7;
    --qgfda-green: #34d399;
    --qgfda-amber: #fbbf24;
    --qgfda-red: #f87171;
    --qgfda-text: #e5e7eb;
    --qgfda-muted: #9ca3af;
}
.stApp {
    background: radial-gradient(circle at 15% 0%, #111827 0%, #0b0f19 55%) fixed;
    color: var(--qgfda-text);
}
section[data-testid="stSidebar"] {
    background-color: #0a0e17;
    border-right: 1px solid var(--qgfda-border);
}
h1, h2, h3 { color: var(--qgfda-text) !important; letter-spacing: 0.02em; }
h1 { text-shadow: 0 0 18px rgba(34, 211, 238, 0.35); }

.qgfda-banner {
    background: linear-gradient(90deg, rgba(168,85,247,0.15), rgba(34,211,238,0.15));
    border: 1px solid var(--qgfda-border);
    border-left: 4px solid var(--qgfda-accent);
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 18px;
    font-size: 0.85rem;
    color: var(--qgfda-muted);
}
.qgfda-card {
    background: var(--qgfda-panel);
    border: 1px solid var(--qgfda-border);
    border-radius: 10px;
    padding: 16px 18px;
}
.qgfda-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.qgfda-pill-approve { background: rgba(52,211,153,0.15); color: var(--qgfda-green); border: 1px solid rgba(52,211,153,0.4); }
.qgfda-pill-review  { background: rgba(251,191,36,0.15); color: var(--qgfda-amber); border: 1px solid rgba(251,191,36,0.4); }
.qgfda-pill-block   { background: rgba(248,113,113,0.15); color: var(--qgfda-red); border: 1px solid rgba(248,113,113,0.4); }

div[data-testid="stMetric"] {
    background: var(--qgfda-panel);
    border: 1px solid var(--qgfda-border);
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
"""


def inject_theme() -> None:
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown(
        '<div class="qgfda-banner">🔒 <b>Research Prototype — Not for Production Financial Authorization.</b> '
        "All transaction data on this dashboard is synthetic or anonymized; no real card numbers, "
        "CVVs, account numbers, or PII are displayed anywhere in this application.</div>",
        unsafe_allow_html=True,
    )


def decision_pill(decision: str) -> str:
    cls = {"APPROVE": "qgfda-pill-approve", "REVIEW": "qgfda-pill-review", "BLOCK": "qgfda-pill-block"}.get(decision, "")
    return f'<span class="qgfda-pill {cls}">{decision}</span>'
