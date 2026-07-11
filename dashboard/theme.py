"""
theme.py
PASS dashboard brand system: type, palette tokens, card primitive, callouts.

Design intent (deliberately not the generator defaults):
  * Type pairing. Headlines/wordmark in Fraunces (a warm, high-contrast serif:
    trust + human care) against Geist for body, UI and data (clean grotesque,
    strong numerals). Both are self-hosted (base64 in _fonts.css) so they load
    with no runtime network dependency. Regenerate _fonts.css from Google Fonts
    (Fraunces + Geist, latin subset) if the weights ever change.
  * Sentence case. Labels use weight + colour for hierarchy, NOT ALL CAPS.
  * One narrow palette (below). One card primitive, repeated everywhere.
Pure presentation, no data or functionality here.
"""

from __future__ import annotations

import pathlib

import streamlit as st

# ── PASS palette (design tokens) ──────────────────────────────────────────
# A deliberately narrow, teal-committed system: ONE brand primary, ONE progress
# accent, a graded neutral ramp, and no third hue (cautions read neutral).
PRIMARY = "#0E4C5E"     # brand deep teal: headers, wordmark, primary actions, values
PRIMARY_DK = "#0A3A48"  # emphasis / metric values
CHART = "#127C91"       # data-viz teal (same family, tuned for legibility on white)
POSITIVE = "#1B9C85"    # the one accent: progress / positive deltas / "on track"
INK = "#16262E"         # primary text
TEXT = INK              # alias
MUTED = "#5E6E77"       # secondary text and labels
FAINT = "#8A97A0"       # captions / tertiary
BORDER = "#E4E9EE"      # hairlines / card borders
BG = "#F6F8FA"          # app canvas
SURFACE = "#FFFFFF"     # cards / sidebar
CAUTION = "#5E6E77"     # cautions render neutral (no third hue); kept for callers

FONT_HEAD = "'Fraunces', Georgia, 'Times New Roman', serif"
FONT_BODY = "'Geist', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

_FONTS_CSS = (pathlib.Path(__file__).parent / "_fonts.css").read_text(encoding="utf-8")


def logo_svg(size: int = 30) -> str:
    """A minimal 'measured angle' mark: an angle with an arc, evoking knee
    flexion measurement. Rendered in the brand primary."""
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 32 32" fill="none"
     xmlns="http://www.w3.org/2000/svg" style="display:block">
  <path d="M6.5 25.5 L27 25.5" stroke="{PRIMARY}" stroke-width="2.6"
        stroke-linecap="round"/>
  <path d="M6.5 25.5 L24 8.5" stroke="{PRIMARY}" stroke-width="2.6"
        stroke-linecap="round"/>
  <path d="M18.5 25.5 A12 12 0 0 0 15.1 17.1" stroke="{POSITIVE}"
        stroke-width="2.2" stroke-linecap="round"/>
  <circle cx="6.5" cy="25.5" r="2.7" fill="{PRIMARY}"/>
</svg>"""


def inject_css():
    """Self-hosted fonts + the full component stylesheet."""
    st.markdown(f"""
<style>
{_FONTS_CSS}

:root {{ --font-head: {FONT_HEAD}; --font-body: {FONT_BODY}; }}

.stApp {{ background: {BG}; color: {INK}; }}

/* body font: Geist, scoped to text surfaces so icon fonts are untouched */
.stApp, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stApp label, .stButton button, [data-testid="stFormSubmitButton"] button,
button[data-baseweb="tab"], .stApp input, .stApp textarea {{
    font-family: var(--font-body) !important;
}}

/* headline font: Fraunces on titles + wordmark (higher specificity wins) */
.stApp h1, .stApp h2, .stApp h3, .stApp .brand-word {{
    font-family: var(--font-head) !important; font-weight: 600 !important;
    color: {PRIMARY_DK}; letter-spacing: -0.005em;
}}
.stApp h1 {{ font-size: 2rem !important; }}
.stApp h2 {{ font-size: 1.5rem !important; }}
.stApp h3 {{ font-size: 1.15rem !important; }}

/* remove Streamlit chrome for a product feel */
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* section label: sentence case, hierarchy via weight + colour (no all-caps) */
.section-label {{
    font-size: 0.9rem; font-weight: 600; color: {MUTED};
    margin: 10px 0 4px; letter-spacing: 0;
}}

/* ── the one card primitive (every metric is this, nothing else) ── */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(16,40,50,0.05);
    min-height: 108px;
}}
[data-testid="stMetricLabel"] {{ overflow: visible; white-space: normal; }}
[data-testid="stMetricLabel"] p {{
    font-size: 0.82rem !important; font-weight: 500; color: {MUTED};
    letter-spacing: 0;                 /* sentence case, not uppercase */
    white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.3;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--font-body);
    font-size: 2rem; font-weight: 600; color: {PRIMARY_DK};
    line-height: 1.15; font-feature-settings: "tnum" 1, "cv01" 1;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600; }}

/* brand header */
.brand-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 2px 2px 12px;
}}
.brand-left {{ display: flex; align-items: center; gap: 11px; }}
.brand-word {{ font-size: 1.5rem; letter-spacing: 0.01em; line-height: 1; }}
.brand-home {{
    display: flex; align-items: center; gap: 11px;
    text-decoration: none; color: inherit; cursor: pointer;
    transition: opacity 0.15s ease;
}}
.brand-home:hover {{ opacity: 0.72; }}
.brand-home--static {{ cursor: default; }}
.brand-home--static:hover {{ opacity: 1; }}
.brand-tag {{ font-size: 0.82rem; color: {MUTED}; }}
.brand-rule {{ border: none; border-top: 1px solid {BORDER}; margin: 0 0 20px; }}

/* sidebar brand */
.sb-brand {{ display: flex; align-items: center; gap: 9px; margin-bottom: 2px; }}
.sb-brand .brand-word {{ font-size: 1.25rem; }}

/* branded callouts (no default alert icons, no emoji) */
.callout {{
    border-radius: 10px; padding: 12px 16px; font-size: 0.9rem;
    margin: 4px 0 14px; border: 1px solid transparent; line-height: 1.45;
}}
.callout-positive {{ background: #E9F5F1; border-color: #CBE8DF; color: #0C5A49; }}
.callout-info     {{ background: #EAF2F4; border-color: #D3E4E8; color: #134350; }}
.callout-caution  {{ background: #F1F4F6; border-color: {BORDER}; color: {MUTED}; }}
.callout b {{ font-weight: 600; color: {INK}; }}

/* login form as a card */
[data-testid="stForm"] {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 26px 26px 20px; box-shadow: 0 4px 18px rgba(16,40,50,0.06);
}}

/* buttons */
.stButton > button, [data-testid="stFormSubmitButton"] button {{
    border-radius: 9px; font-weight: 500; border-color: {BORDER};
}}

/* tabs: sentence case, weight for state */
button[data-baseweb="tab"] {{ font-weight: 500; letter-spacing: 0; }}
button[data-baseweb="tab"][aria-selected="true"] {{ font-weight: 600; }}
[data-baseweb="tab-highlight"] {{ background-color: {PRIMARY}; }}

[data-testid="stDataFrame"] {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


def render_brand_header(tagline: str = "Patient Assessment Sensing System",
                        home_href: str | None = None):
    """Consistent PASS wordmark header. If home_href is given, the wordmark is a
    link back to the role's home; otherwise it is plain (non-interactive)."""
    mark = f'{logo_svg(30)}<span class="brand-word">PASS</span>'
    if home_href:
        left = (f'<a class="brand-home" href="{home_href}" target="_self" '
                f'title="Back to home">{mark}</a>')
    else:
        left = f'<span class="brand-home brand-home--static">{mark}</span>'
    st.markdown(f"""
<div class="brand-header">
  <div class="brand-left">{left}</div>
  <div class="brand-tag">{tagline}</div>
</div>
<hr class="brand-rule"/>
""", unsafe_allow_html=True)


def render_sidebar_brand():
    st.markdown(f'<div class="sb-brand">{logo_svg(24)}'
                f'<span class="brand-word">PASS</span></div>',
                unsafe_allow_html=True)


def callout(kind: str, text: str):
    """Branded inline callout. kind in {'positive','info','caution'}. No emoji."""
    st.markdown(f"<div class='callout callout-{kind}'>{text}</div>",
                unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f"<div class='section-label'>{text}</div>", unsafe_allow_html=True)
