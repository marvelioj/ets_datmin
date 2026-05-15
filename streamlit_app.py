# SteamVault Pro - Steam Game Discovery & Hybrid Recommendation Dashboard
# Put this file in the same folder as steam_top_games_2026.csv, or upload the CSV from the sidebar.

from __future__ import annotations

import html
import io
import math
import re
import textwrap
from urllib.parse import quote
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None


APP_TITLE = "SteamVault Pro"
DEFAULT_CSV = Path(__file__).parent / "steam_top_games_2026.csv"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="SV",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
def render_html(markup: str, **_ignored_kwargs) -> None:
    """Render custom HTML/CSS as HTML, not Markdown.

    This prevents Markdown from turning nested card HTML into visible text/code blocks.
    """
    cleaned = textwrap.dedent(str(markup)).strip()
    if not cleaned:
        return
    try:
        st.html(cleaned)
    except Exception:
        st.markdown(cleaned, unsafe_allow_html=True)


def inject_css() -> None:
    render_html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        :root {
            --mist: #A5C5CC;
            --ink: #021334;
            --deep: #012A61;
            --mid: #275A91;
            --rose: #977086;
            --gold: #FDC787;
            --bg-0: #020817;
            --bg-1: #021334;
            --panel: rgba(1, 42, 97, 0.36);
            --panel-strong: rgba(2, 19, 52, 0.86);
            --panel-soft: rgba(39, 90, 145, 0.16);
            --line: rgba(165, 197, 204, 0.18);
            --line-strong: rgba(253, 199, 135, 0.36);
            --text: #EEF8FA;
            --text-soft: #C7DCE2;
            --muted: rgba(165, 197, 204, 0.72);
            --shadow: rgba(0, 0, 0, 0.46);
        }

        html {
            scroll-behavior: smooth;
        }

        html, body, .stApp {
            min-height: 100%;
            background: var(--bg-0) !important;
            color: var(--text) !important;
            font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% -8%, rgba(253, 199, 135, 0.16), transparent 22rem),
                radial-gradient(circle at 86% 3%, rgba(39, 90, 145, 0.36), transparent 33rem),
                radial-gradient(circle at 50% 105%, rgba(151, 112, 134, 0.20), transparent 36rem),
                linear-gradient(135deg, #020817 0%, #021334 45%, #010714 100%) !important;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background-image:
                linear-gradient(rgba(165, 197, 204, 0.030) 1px, transparent 1px),
                linear-gradient(90deg, rgba(165, 197, 204, 0.026) 1px, transparent 1px);
            background-size: 72px 72px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,.95), rgba(0,0,0,.30));
        }

        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
                radial-gradient(circle at 50% 0%, transparent 0, rgba(2, 19, 52, 0.25) 42%, rgba(2, 8, 23, 0.72) 100%),
                linear-gradient(to bottom, rgba(2, 19, 52, 0.05), rgba(0,0,0,0.28));
        }

        .main .block-container, .block-container {
            position: relative;
            z-index: 1;
            max-width: 1540px;
            padding-top: 1rem;
            padding-bottom: 4rem;
        }

        #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {
            visibility: hidden !important;
            height: 0 !important;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li, .stApp label, .stApp span,
        .stApp [data-testid="stMarkdownContainer"] {
            color: var(--text) !important;
        }

        .stApp a {
            color: var(--mist) !important;
        }

        h1, h2, h3 {
            letter-spacing: -0.055em;
        }

        .muted, .stApp small, [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
        }

        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 20% 0%, rgba(253, 199, 135, 0.10), transparent 16rem),
                linear-gradient(180deg, rgba(1, 10, 28, 0.98), rgba(2, 19, 52, 0.96)) !important;
            border-right: 1px solid rgba(165, 197, 204, 0.16);
            box-shadow: 22px 0 60px rgba(0,0,0,.32);
        }

        section[data-testid="stSidebar"] > div {
            background: transparent !important;
            padding-top: 1.2rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
            color: var(--text-soft) !important;
        }

        .brand-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(253, 199, 135, 0.22);
            border-radius: 24px;
            padding: 18px 16px;
            margin: 0 0 18px;
            background:
                radial-gradient(circle at top right, rgba(253,199,135,.16), transparent 9rem),
                linear-gradient(145deg, rgba(1,42,97,.42), rgba(2,19,52,.84));
            box-shadow: 0 22px 60px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.08);
        }
        .brand-mark {
            width: 46px;
            height: 46px;
            display: grid;
            place-items: center;
            border-radius: 16px;
            margin-bottom: 12px;
            color: #021334 !important;
            font-weight: 950;
            letter-spacing: -0.08em;
            background: linear-gradient(135deg, var(--gold), var(--mist));
            box-shadow: 0 0 35px rgba(253,199,135,.28);
        }
        .brand-card h2 {
            margin: 0;
            font-size: 1.26rem;
            line-height: 1.05;
        }
        .brand-card p {
            margin: 7px 0 0;
            color: var(--muted) !important;
            font-size: .82rem;
            line-height: 1.5;
        }

        .sidebar-note {
            margin: 10px 0 18px;
            padding: 11px 13px;
            border-radius: 16px;
            background: rgba(165,197,204,.06);
            border: 1px solid rgba(165,197,204,.12);
            color: var(--muted) !important;
            font-size: .80rem;
            line-height: 1.45;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(2, 19, 52, 0.92) !important;
            border: 1px solid rgba(165, 197, 204, 0.18) !important;
            border-radius: 15px !important;
            color: var(--text) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.05) !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            caret-color: var(--gold) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg,
        div[data-baseweb="select"] div {
            color: var(--text) !important;
            fill: var(--text) !important;
        }

        div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
        div[role="listbox"], ul[role="listbox"] {
            background: #03112d !important;
            border: 1px solid rgba(253,199,135,.22) !important;
            border-radius: 16px !important;
            color: var(--text) !important;
            box-shadow: 0 28px 70px rgba(0,0,0,.55) !important;
        }

        div[role="option"], li[role="option"] {
            background: #03112d !important;
            color: var(--text) !important;
        }
        div[role="option"]:hover, li[role="option"]:hover {
            background: rgba(39,90,145,.35) !important;
        }
        div[data-baseweb="tag"] {
            background: rgba(39,90,145,.36) !important;
            border: 1px solid rgba(165,197,204,.26) !important;
            color: var(--text) !important;
            border-radius: 999px !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(2, 19, 52, 0.74) !important;
            border: 1px dashed rgba(253, 199, 135, 0.34) !important;
            border-radius: 20px !important;
            color: var(--text) !important;
            box-shadow: inset 0 0 40px rgba(39,90,145,.10);
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--text) !important;
        }

        [data-testid="stFileUploaderDropzone"] button,
        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(135deg, rgba(253,199,135,.92), rgba(165,197,204,.74)) !important;
            border: 1px solid rgba(253,199,135,.55) !important;
            border-radius: 15px !important;
            color: #021334 !important;
            font-weight: 900 !important;
            box-shadow: 0 14px 34px rgba(253,199,135,.18) !important;
            transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover,
        .stButton button:hover,
        .stDownloadButton button:hover {
            transform: translateY(-2px);
            filter: brightness(1.05);
            box-shadow: 0 18px 48px rgba(253,199,135,.26) !important;
        }

        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--gold) !important;
            font-weight: 950 !important;
            text-shadow: 0 0 20px rgba(253,199,135,.22);
        }
        [data-testid="stSlider"] p {
            color: var(--text-soft) !important;
            font-weight: 800 !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            min-height: 430px;
            border-radius: 34px;
            padding: clamp(26px, 4vw, 54px);
            margin: 0 0 24px 0;
            border: 1px solid rgba(253,199,135,.30);
            background:
                radial-gradient(circle at 78% 16%, rgba(253,199,135,.24), transparent 13rem),
                radial-gradient(circle at 18% 10%, rgba(39,90,145,.50), transparent 25rem),
                linear-gradient(115deg, rgba(2,19,52,.98) 0%, rgba(1,42,97,.78) 52%, rgba(2,19,52,.96) 100%);
            box-shadow: 0 38px 110px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.08);
            isolation: isolate;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: -1px;
            z-index: -1;
            background:
                linear-gradient(90deg, rgba(253,199,135,.14), transparent 34%),
                repeating-linear-gradient(115deg, rgba(165,197,204,.055) 0 1px, transparent 1px 22px);
            opacity: .72;
        }
        .hero::after {
            content: "";
            position: absolute;
            right: -120px;
            top: -90px;
            width: 420px;
            height: 420px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(253,199,135,.20), rgba(151,112,134,.10) 45%, transparent 68%);
            filter: blur(1px);
            animation: floatAura 9s ease-in-out infinite alternate;
        }
        @keyframes floatAura {
            from { transform: translate3d(0,0,0) scale(1); opacity: .78; }
            to { transform: translate3d(-24px, 26px, 0) scale(1.06); opacity: 1; }
        }
        @keyframes shimmer {
            from { transform: translateX(-140%); }
            to { transform: translateX(140%); }
        }
        @keyframes drift {
            from { transform: translateY(0); opacity: .28; }
            50% { opacity: .90; }
            to { transform: translateY(-18px); opacity: .36; }
        }
        .hero-grid {
            position: relative;
            z-index: 2;
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(300px, .72fr);
            gap: clamp(22px, 4vw, 48px);
            align-items: center;
        }
        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 9px;
            padding: 8px 13px;
            margin-bottom: 16px;
            border-radius: 999px;
            color: var(--gold) !important;
            font-size: .78rem;
            font-weight: 950;
            letter-spacing: .08em;
            text-transform: uppercase;
            background: rgba(253,199,135,.10);
            border: 1px solid rgba(253,199,135,.25);
            box-shadow: 0 0 34px rgba(253,199,135,.12);
        }
        .hero-kicker::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--gold);
            box-shadow: 0 0 16px var(--gold);
        }
        .hero h1 {
            max-width: 920px;
            margin: 0;
            color: #ffffff !important;
            font-size: clamp(2.9rem, 6.4vw, 6.8rem);
            line-height: .86;
            letter-spacing: -0.075em;
            text-shadow: 0 18px 58px rgba(0,0,0,.46);
        }
        .hero h1 .accent {
            color: var(--gold) !important;
            text-shadow: 0 0 36px rgba(253,199,135,.26);
        }
        .hero-subtitle {
            max-width: 860px;
            color: var(--text-soft) !important;
            font-size: clamp(1.02rem, 1.25vw, 1.20rem);
            line-height: 1.72;
            margin: 20px 0 0;
        }
        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 13px;
            margin-top: 28px;
        }
        .cta {
            position: relative;
            overflow: hidden;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            min-height: 48px;
            padding: 0 20px;
            border-radius: 999px;
            font-weight: 950;
            text-decoration: none !important;
            transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        }
        .cta-primary {
            color: #021334 !important;
            background: linear-gradient(135deg, var(--gold), #ffe2ad 45%, var(--mist));
            border: 1px solid rgba(253,199,135,.55);
            box-shadow: 0 18px 48px rgba(253,199,135,.25);
        }
        .cta-primary::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,.42), transparent);
            animation: shimmer 3.8s infinite;
        }
        .cta-secondary {
            color: var(--text) !important;
            background: rgba(165,197,204,.08);
            border: 1px solid rgba(165,197,204,.26);
            box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
        }
        .cta:hover {
            transform: translateY(-3px);
            box-shadow: 0 26px 70px rgba(253,199,135,.22);
        }
        .hero-panel {
            position: relative;
            z-index: 2;
            border-radius: 28px;
            padding: 18px;
            background: rgba(2,19,52,.62);
            border: 1px solid rgba(165,197,204,.20);
            box-shadow: 0 30px 82px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.08);
            backdrop-filter: blur(18px);
        }
        .launcher-screen {
            position: relative;
            min-height: 280px;
            overflow: hidden;
            border-radius: 22px;
            background:
                radial-gradient(circle at 72% 25%, rgba(253,199,135,.24), transparent 8rem),
                linear-gradient(145deg, rgba(39,90,145,.34), rgba(1,42,97,.30));
            border: 1px solid rgba(165,197,204,.16);
        }
        .launcher-screen::before,
        .launcher-screen::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            background: rgba(253,199,135,.78);
            box-shadow: 0 0 24px rgba(253,199,135,.40);
            animation: drift 4.8s ease-in-out infinite alternate;
        }
        .launcher-screen::before { width: 7px; height: 7px; left: 18%; top: 22%; }
        .launcher-screen::after { width: 5px; height: 5px; right: 18%; bottom: 26%; animation-delay: 1.3s; }
        .mock-row {
            position: absolute;
            left: 18px;
            right: 18px;
            display: grid;
            grid-template-columns: 70px 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 12px;
            border-radius: 18px;
            background: rgba(2,19,52,.58);
            border: 1px solid rgba(165,197,204,.12);
        }
        .mock-row.one { top: 22px; }
        .mock-row.two { top: 112px; transform: translateX(18px); opacity: .92; }
        .mock-row.three { top: 202px; transform: translateX(-8px); opacity: .82; }
        .mock-img {
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--gold), var(--mid));
            box-shadow: 0 12px 24px rgba(0,0,0,.22);
        }
        .mock-line b, .mock-line span { display: block; }
        .mock-line b { color: #fff !important; font-size: .88rem; margin-bottom: 5px; }
        .mock-line span { color: var(--muted) !important; font-size: .74rem; }
        .mock-score {
            display: grid;
            place-items: center;
            width: 50px;
            height: 34px;
            border-radius: 999px;
            color: var(--gold) !important;
            font-weight: 950;
            background: rgba(253,199,135,.10);
            border: 1px solid rgba(253,199,135,.25);
        }
        .hero-stats {
            display: flex;
            flex-wrap: wrap
            gap: 12px;
            margin-top: 16px;
        }
        .hero-stat {
            flex: 0 1 auto;
            border-radius: 18px;
            padding: 13px;
            background: rgba(165,197,204,.06);
            border: 1px solid rgba(165,197,204,.13);
        }
        .hero-stat strong {
            display: block;
            color: #fff !important;
            font-size: 1.2rem;
            line-height: 1;
            white-space: nowrap;
        }
        .hero-stat span {
            display: block;
            margin-top: 5px;
            color: var(--muted) !important;
            font-size: .73rem;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 0 0 24px;
        }
        .feature-card {
            position: relative;
            overflow: hidden;
            min-height: 118px;
            border-radius: 24px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(1,42,97,.42), rgba(2,19,52,.72));
            border: 1px solid rgba(165,197,204,.16);
            box-shadow: 0 18px 60px rgba(0,0,0,.26), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .feature-card::after {
            content: "";
            position: absolute;
            width: 96px;
            height: 96px;
            right: -35px;
            bottom: -35px;
            border-radius: 999px;
            background: rgba(253,199,135,.10);
            filter: blur(1px);
        }
        .feature-card b {
            display: block;
            color: #fff !important;
            font-size: .96rem;
            margin-bottom: 8px;
        }
        .feature-card span {
            color: var(--muted) !important;
            font-size: .82rem;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(1,42,97,.44), rgba(2,19,52,.72));
            border: 1px solid rgba(165,197,204,.16);
            border-radius: 22px;
            padding: 1rem 1rem;
            box-shadow: 0 22px 64px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.06);
        }
        div[data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(253,199,135,.10), transparent 45%);
            opacity: .74;
        }
        div[data-testid="stMetric"] label { color: var(--muted) !important; font-weight: 800 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 950 !important; }

        .glass-panel {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(1,42,97,.42), rgba(2,19,52,.76));
            border: 1px solid rgba(165,197,204,.16);
            border-radius: 24px;
            padding: 18px;
            box-shadow: 0 20px 60px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.06);
            margin-bottom: 16px;
            color: var(--text) !important;
        }
        .glass-panel b { color: #fff !important; }

        .section-title {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 14px;
            margin: 18px 0 14px 0;
            padding-top: 4px;
        }
        .section-title h3 {
            margin: 0;
            color: #ffffff !important;
            font-size: clamp(1.35rem, 2vw, 2.05rem);
            letter-spacing: -0.05em;
        }
        .section-title span {
            color: var(--muted) !important;
            font-size: .84rem;
        }

        .game-card {
            position: relative;
            height: 100%;
            min-height: 100%;
            overflow: hidden;
            border-radius: 28px;
            background:
                linear-gradient(180deg, rgba(1,42,97,.54), rgba(2,19,52,.88));
            border: 1px solid rgba(165,197,204,.17);
            box-shadow: 0 24px 72px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.06);
            transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
            margin-bottom: 20px;
            isolation: isolate;
        }
        .game-card::before {
            content: "";
            position: absolute;
            inset: -1px;
            z-index: -1;
            background: linear-gradient(135deg, rgba(253,199,135,.23), transparent 28%, rgba(39,90,145,.30));
            opacity: 0;
            transition: opacity .22s ease;
        }
        .game-card:hover {
            transform: translateY(-7px);
            border-color: rgba(253,199,135,.38);
            box-shadow: 0 34px 100px rgba(0,0,0,.46), 0 0 48px rgba(39,90,145,.18);
        }
        .game-card:hover::before { opacity: 1; }
        .game-img-wrap {
            position: relative;
            width: 100%;
            aspect-ratio: 2.16 / 1;
            background:
                radial-gradient(circle at 72% 18%, rgba(253,199,135,.18), transparent 8rem),
                linear-gradient(135deg, rgba(39,90,145,.44), rgba(2,19,52,.84));
            overflow: hidden;
            border-bottom: 1px solid rgba(165,197,204,.13);
        }
        .game-img-wrap::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(to top, rgba(2,19,52,.74), transparent 62%);
            pointer-events: none;
        }
        .game-img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transform: scale(1.01);
            transition: transform .34s ease, filter .34s ease;
        }
        .game-card:hover .game-img-wrap img {
            transform: scale(1.065);
            filter: saturate(1.08) contrast(1.05);
        }
        .game-img-fallback {
            position: absolute;
            inset: 0;
            display: grid;
            place-items: center;
            text-align: center;
            padding: 18px;
        }
        .game-img-fallback span {
            display: grid;
            place-items: center;
            width: 64px;
            height: 64px;
            margin-bottom: 8px;
            border-radius: 22px;
            background: linear-gradient(135deg, var(--gold), var(--mid));
            color: var(--ink) !important;
            font-weight: 950;
        }
        .game-img-fallback b { color: #fff !important; }
        .game-topline {
            position: absolute;
            left: 14px;
            right: 14px;
            top: 14px;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .rank-badge, .score-badge {
            display: inline-flex;
            align-items: center;
            min-height: 30px;
            padding: 0 10px;
            border-radius: 999px;
            font-weight: 950;
            font-size: .70rem;
            border: 1px solid rgba(253,199,135,.32);
            background: rgba(2,19,52,.58);
            color: var(--gold) !important;
            backdrop-filter: blur(12px);
        }
        .score-badge { color: var(--mist) !important; border-color: rgba(165,197,204,.30); }
        .game-body { padding: 17px 17px 18px; }
        .game-title {
            font-size: 1.13rem;
            font-weight: 950;
            color: #ffffff !important;
            line-height: 1.20;
            margin-bottom: 7px;
            letter-spacing: -0.03em;
        }
        .game-title a { color: #ffffff !important; text-decoration: none; }
        .game-title a:hover { color: var(--gold) !important; }
        .meta-line {
            color: var(--muted) !important;
            font-size: .82rem;
            margin-bottom: 10px;
        }
        .pill-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0; }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: .70rem;
            font-weight: 950;
            letter-spacing: .01em;
            border: 1px solid rgba(165,197,204,.16);
            color: var(--text-soft) !important;
            background: rgba(165,197,204,.08);
            white-space: nowrap;
        }
        .pill-blue { background: rgba(39,90,145,.22); color: #D8F1F6 !important; border-color: rgba(165,197,204,.22); }
        .pill-green { background: rgba(165,197,204,.14); color: #EAF7F9 !important; border-color: rgba(165,197,204,.25); }
        .pill-amber { background: rgba(253,199,135,.14); color: var(--gold) !important; border-color: rgba(253,199,135,.30); }
        .pill-red { background: rgba(151,112,134,.22); color: #FFDCE9 !important; border-color: rgba(151,112,134,.36); }
        .tag {
            display: inline-flex;
            padding: 5px 9px;
            border-radius: 999px;
            background: rgba(165,197,204,.075);
            border: 1px solid rgba(165,197,204,.15);
            color: var(--text-soft) !important;
            font-size: .70rem;
            margin: 0 5px 6px 0;
        }
        .why {
            position: relative;
            border: 1px solid rgba(253,199,135,.16);
            background: linear-gradient(180deg, rgba(253,199,135,.08), rgba(39,90,145,.10));
            margin-top: 12px;
            padding: 12px 13px;
            border-radius: 18px;
            color: var(--text-soft) !important;
            font-size: .81rem;
            line-height: 1.5;
        }
        .why b, .why-label {
            color: var(--gold) !important;
            font-weight: 950;
        }
        .card-actions {
            display: flex;
            gap: 9px;
            margin-top: 13px;
        }
        .card-action {
            flex: 1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 38px;
            border-radius: 999px;
            text-decoration: none !important;
            color: var(--ink) !important;
            font-weight: 950;
            font-size: .78rem;
            background: linear-gradient(135deg, var(--gold), var(--mist));
            border: 1px solid rgba(253,199,135,.38);
            transition: transform .16s ease, filter .16s ease;
        }
        .card-action.secondary {
            color: var(--text) !important;
            background: rgba(165,197,204,.08);
            border-color: rgba(165,197,204,.16);
        }
        .card-action:hover { transform: translateY(-2px); filter: brightness(1.06); }
        .bar-row { margin: 9px 0; }
        .bar-label {
            display: flex;
            justify-content: space-between;
            color: var(--muted) !important;
            font-size: .72rem;
            margin-bottom: 5px;
        }
        .bar-label span { color: var(--muted) !important; }
        .bar-track {
            height: 8px;
            border-radius: 999px;
            background: rgba(165,197,204,.12);
            overflow: hidden;
            box-shadow: inset 0 1px 5px rgba(0,0,0,.30);
        }
        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--mid), var(--mist), var(--gold));
            box-shadow: 0 0 16px rgba(253,199,135,.22);
        }
        .mini-note {
            border-left: 3px solid var(--gold);
            background: rgba(253,199,135,.075);
            border-radius: 16px;
            padding: 13px 15px;
            color: var(--text-soft) !important;
            margin: 8px 0 16px 0;
            border-top: 1px solid rgba(253,199,135,.10);
            border-right: 1px solid rgba(253,199,135,.10);
            border-bottom: 1px solid rgba(253,199,135,.10);
        }
        .method-card {
            background: linear-gradient(180deg, rgba(1,42,97,.42), rgba(2,19,52,.72));
            border: 1px solid rgba(165,197,204,.16);
            border-radius: 22px;
            padding: 18px;
            height: 100%;
            box-shadow: 0 18px 54px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .method-card h4 { margin-top: 0; color: #fff !important; }
        .method-card p { color: var(--text-soft) !important; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 8px;
            border-radius: 999px;
            background: rgba(2,19,52,.42);
            border: 1px solid rgba(165,197,204,.12);
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(165,197,204,.055);
            border: 1px solid rgba(165,197,204,.12);
            border-radius: 999px;
            padding: 10px 16px;
            color: var(--text-soft) !important;
        }
        .stTabs [data-baseweb="tab"] p { color: var(--text-soft) !important; font-weight: 900 !important; }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(253,199,135,.18), rgba(39,90,145,.25)) !important;
            border-color: rgba(253,199,135,.34) !important;
            box-shadow: 0 0 28px rgba(253,199,135,.10);
        }
        .stTabs [aria-selected="true"] p { color: var(--gold) !important; }

        [data-testid="stExpander"] {
            background: rgba(2,19,52,.66) !important;
            border: 1px solid rgba(165,197,204,.15) !important;
            border-radius: 20px !important;
            box-shadow: 0 18px 50px rgba(0,0,0,.22);
        }
        .stAlert {
            background: rgba(2,19,52,.88) !important;
            color: var(--text) !important;
            border-radius: 18px !important;
            border: 1px solid rgba(165,197,204,.16) !important;
        }
        [data-testid="stDataFrame"] {
            border-radius: 20px !important;
            overflow: hidden !important;
            border: 1px solid rgba(165,197,204,.16) !important;
            box-shadow: 0 20px 60px rgba(0,0,0,.28);
        }

        @media (max-width: 1100px) {
            .hero-grid { grid-template-columns: 1fr; }
            .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero { min-height: auto; }
        }
        @media (max-width: 700px) {
            .block-container { padding-left: .85rem !important; padding-right: .85rem !important; }
            .hero { border-radius: 24px; padding: 24px 18px; }
            .hero h1 { font-size: clamp(2.45rem, 16vw, 4rem); }
            .hero-actions { flex-direction: column; }
            .cta { width: 100%; }
            .hero-stats { grid-template-columns: 1fr; }
            .feature-grid { grid-template-columns: 1fr; }
            .section-title { display: block; }
            .stTabs [data-baseweb="tab-list"] { border-radius: 22px; flex-wrap: wrap; }
            .game-card { border-radius: 22px; }
        }


        /* Final premium pass: cinematic depth, clickable UI, stronger identity */
        .stApp [data-testid="stVerticalBlock"] { animation: softReveal .55s ease both; }
        @keyframes softReveal {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseGlow {
            0%, 100% { opacity: .52; transform: scale(1); }
            50% { opacity: .98; transform: scale(1.035); }
        }
        @keyframes slowRotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes scanline {
            0% { transform: translateX(-130%) skewX(-18deg); opacity: 0; }
            35% { opacity: .65; }
            100% { transform: translateX(150%) skewX(-18deg); opacity: 0; }
        }
        @keyframes particleFloat {
            from { transform: translate3d(0, 14px, 0) scale(.88); opacity: .25; }
            45% { opacity: .96; }
            to { transform: translate3d(10px, -22px, 0) scale(1.08); opacity: .40; }
        }

        .hero {
            min-height: clamp(540px, 58vh, 690px);
            border-radius: 42px;
            padding: clamp(30px, 5vw, 70px);
            border-color: rgba(253,199,135,.42);
            background:
                radial-gradient(circle at 78% 18%, rgba(253,199,135,.27), transparent 13rem),
                radial-gradient(circle at 18% 7%, rgba(165,197,204,.14), transparent 22rem),
                radial-gradient(circle at 68% 88%, rgba(151,112,134,.21), transparent 22rem),
                linear-gradient(118deg, rgba(2,19,52,.99) 0%, rgba(1,42,97,.86) 48%, rgba(0,5,17,.98) 100%);
            box-shadow:
                0 55px 150px rgba(0,0,0,.56),
                0 0 92px rgba(39,90,145,.19),
                inset 0 1px 0 rgba(255,255,255,.10);
        }
        .hero::before {
            background:
                linear-gradient(105deg, rgba(253,199,135,.20), transparent 30%, rgba(165,197,204,.06) 64%, transparent),
                repeating-linear-gradient(116deg, rgba(165,197,204,.062) 0 1px, transparent 1px 28px),
                linear-gradient(to bottom, rgba(255,255,255,.04), transparent 28%);
        }
        .hero-grid {
            grid-template-columns: minmax(0, 1.12fr) minmax(340px, .78fr);
        }
        .hero-copy { position: relative; z-index: 3; }
        .hero h1 {
            max-width: 960px;
            font-size: clamp(3.35rem, 7.8vw, 8.2rem);
            line-height: .80;
            letter-spacing: -0.095em;
        }
        .hero h1 .ghost-word {
            display: block;
            color: rgba(165,197,204,.40) !important;
            -webkit-text-stroke: 1px rgba(253,199,135,.18);
            text-shadow: none;
        }
        .hero h1 .accent {
            background: linear-gradient(115deg, var(--gold), #fff2cf 42%, var(--mist));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
            text-shadow: 0 0 42px rgba(253,199,135,.18);
        }
        .hero-subtitle {
            max-width: 780px;
            font-size: clamp(1.05rem, 1.32vw, 1.32rem);
            line-height: 1.82;
            color: rgba(238,248,250,.82) !important;
        }
        .hero-proof-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }
        .hero-proof {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 34px;
            padding: 0 12px;
            border-radius: 999px;
            font-size: .76rem;
            font-weight: 900;
            color: var(--text-soft) !important;
            background: rgba(165,197,204,.075);
            border: 1px solid rgba(165,197,204,.15);
            backdrop-filter: blur(14px);
        }
        .hero-proof::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: var(--gold);
            box-shadow: 0 0 16px rgba(253,199,135,.72);
        }
        .hero-actions { margin-top: 34px; }
        .cta {
            min-height: 56px;
            padding: 0 24px;
            letter-spacing: -.01em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.12);
        }
        .cta-primary {
            box-shadow: 0 24px 64px rgba(253,199,135,.28), 0 0 34px rgba(253,199,135,.18);
        }
        .cta-secondary {
            background: rgba(2,19,52,.48);
            border-color: rgba(165,197,204,.28);
            backdrop-filter: blur(16px);
        }
        .cta:hover { transform: translateY(-4px) scale(1.012); }

        .hero-panel {
            transform: perspective(1000px) rotateY(-7deg) rotateX(3deg);
            border-color: rgba(253,199,135,.24);
            box-shadow:
                0 42px 110px rgba(0,0,0,.50),
                0 0 62px rgba(39,90,145,.18),
                inset 0 1px 0 rgba(255,255,255,.10);
        }
        .hero-panel::before {
            content: "";
            position: absolute;
            inset: -80px -42px auto auto;
            width: 150px;
            height: 150px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(253,199,135,.38), transparent 68%);
            filter: blur(2px);
            animation: pulseGlow 5s ease-in-out infinite;
        }
        .launcher-screen {
            min-height: 378px;
            background:
                radial-gradient(circle at 48% 42%, rgba(253,199,135,.30), transparent 6rem),
                radial-gradient(circle at 74% 26%, rgba(165,197,204,.18), transparent 11rem),
                linear-gradient(145deg, rgba(39,90,145,.34), rgba(1,42,97,.20), rgba(2,19,52,.72));
        }
        .signature-orb {
            position: absolute;
            left: 50%;
            top: 47%;
            z-index: 0;
            width: 164px;
            height: 164px;
            border-radius: 999px;
            transform: translate(-50%, -50%);
            background:
                radial-gradient(circle at 38% 32%, #fff6df 0 8%, var(--gold) 12%, rgba(253,199,135,.34) 36%, rgba(39,90,145,.16) 58%, transparent 72%);
            box-shadow: 0 0 58px rgba(253,199,135,.42), 0 0 118px rgba(39,90,145,.28);
            animation: pulseGlow 5.4s ease-in-out infinite;
        }
        .signature-orb::before,
        .signature-orb::after {
            content: "";
            position: absolute;
            inset: -16px;
            border-radius: inherit;
            border: 1px solid rgba(253,199,135,.22);
            animation: slowRotate 16s linear infinite;
        }
        .signature-orb::after {
            inset: -34px;
            border-color: rgba(165,197,204,.18);
            animation-duration: 24s;
            animation-direction: reverse;
        }
        .particle {
            position: absolute;
            z-index: 1;
            width: 5px;
            height: 5px;
            border-radius: 999px;
            background: var(--gold);
            box-shadow: 0 0 18px rgba(253,199,135,.70);
            animation: particleFloat 4.8s ease-in-out infinite alternate;
        }
        .particle.p1 { left: 18%; top: 24%; animation-delay: .2s; }
        .particle.p2 { right: 23%; top: 18%; animation-delay: 1s; width: 7px; height: 7px; }
        .particle.p3 { left: 32%; bottom: 21%; animation-delay: 1.8s; }
        .particle.p4 { right: 16%; bottom: 27%; animation-delay: 2.5s; width: 4px; height: 4px; }
        .mock-row { z-index: 2; backdrop-filter: blur(18px); }
        .mock-row.one { top: 26px; }
        .mock-row.two { top: 138px; }
        .mock-row.three { top: 250px; }

        .spotlight-deck {
            position: relative;
            overflow: hidden;
            margin: 16px 0 24px;
            padding: clamp(18px, 2.4vw, 28px);
            border-radius: 32px;
            border: 1px solid rgba(165,197,204,.16);
            background:
                radial-gradient(circle at 12% 18%, rgba(253,199,135,.11), transparent 18rem),
                radial-gradient(circle at 86% 0%, rgba(39,90,145,.24), transparent 25rem),
                linear-gradient(180deg, rgba(1,42,97,.24), rgba(2,19,52,.60));
            box-shadow: 0 30px 96px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .spotlight-deck::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(90deg, transparent, rgba(253,199,135,.06), transparent);
            animation: scanline 7s ease-in-out infinite;
        }
        .active-filter-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 12px 0 18px;
            padding: 13px 15px;
            border-radius: 20px;
            background: rgba(253,199,135,.10);
            border: 1px solid rgba(253,199,135,.26);
            color: var(--text-soft) !important;
            box-shadow: 0 18px 52px rgba(0,0,0,.22);
        }
        .active-filter-card b { color: var(--gold) !important; }
        .active-filter-card a {
            color: var(--ink) !important;
            text-decoration: none !important;
            font-weight: 950;
            border-radius: 999px;
            padding: 8px 12px;
            background: linear-gradient(135deg, var(--gold), var(--mist));
        }

        .game-card::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(105deg, transparent 20%, rgba(253,199,135,.12), transparent 46%);
            transform: translateX(-120%) skewX(-18deg);
            opacity: 0;
        }
        .game-card:hover::after { animation: scanline 1.15s ease; }
        .cover-link {
            position: absolute;
            inset: 0;
            z-index: 1;
            display: block;
            text-decoration: none !important;
        }
        .cover-link img, .cover-link .game-img-fallback { position: absolute; inset: 0; }
        .cover-link .game-img-fallback { display: grid; place-items: center; }
        .preview-chip {
            position: absolute;
            left: 50%;
            bottom: 18px;
            z-index: 3;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 36px;
            padding: 0 13px;
            border-radius: 999px;
            color: var(--ink) !important;
            font-size: .72rem;
            font-weight: 950;
            background: linear-gradient(135deg, var(--gold), var(--mist));
            box-shadow: 0 15px 38px rgba(0,0,0,.32), 0 0 28px rgba(253,199,135,.24);
            transform: translate(-50%, 16px) scale(.92);
            opacity: 0;
            transition: opacity .22s ease, transform .22s ease;
            pointer-events: none;
            white-space: nowrap;
        }
        .game-card:hover .preview-chip { opacity: 1; transform: translate(-50%, 0) scale(1); }
        .game-card:hover .rank-badge { box-shadow: 0 0 28px rgba(253,199,135,.24); }
        .tag {
            text-decoration: none !important;
            transition: transform .16s ease, background .16s ease, border-color .16s ease, color .16s ease, box-shadow .16s ease;
        }
        a.tag:hover {
            transform: translateY(-2px);
            color: var(--gold) !important;
            background: rgba(253,199,135,.12);
            border-color: rgba(253,199,135,.34);
            box-shadow: 0 10px 26px rgba(0,0,0,.22), 0 0 24px rgba(253,199,135,.12);
        }
        .tag-active {
            color: var(--gold) !important;
            border-color: rgba(253,199,135,.38) !important;
            background: rgba(253,199,135,.12) !important;
        }
        .card-actions { position: relative; z-index: 3; }

        div[role="radiogroup"] {
            gap: 10px !important;
        }
        div[role="radiogroup"] label {
            border-radius: 999px !important;
            padding: 8px 14px !important;
            border: 1px solid rgba(165,197,204,.14) !important;
            background: rgba(165,197,204,.06) !important;
            transition: transform .16s ease, border-color .16s ease, background .16s ease;
        }
        div[role="radiogroup"] label:hover {
            transform: translateY(-2px);
            border-color: rgba(253,199,135,.28) !important;
            background: rgba(253,199,135,.08) !important;
        }

        .top-nav-shell {
            position: sticky;
            top: 0;
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin: 0 0 18px;
            padding: 10px;
            border: 1px solid rgba(165,197,204,.16);
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(2,19,52,.88), rgba(1,42,97,.50));
            box-shadow: 0 20px 60px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.08);
            backdrop-filter: blur(22px);
        }
        .top-nav-brand {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding-left: 8px;
            min-width: max-content;
            color: var(--text) !important;
            font-weight: 950;
            letter-spacing: -.04em;
        }
        .top-nav-brand span {
            width: 30px;
            height: 30px;
            display: inline-grid;
            place-items: center;
            border-radius: 11px;
            color: var(--ink) !important;
            background: linear-gradient(135deg, var(--gold), var(--mist));
            box-shadow: 0 0 26px rgba(253,199,135,.22);
            font-size: .78rem;
        }
        .top-nav-links {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            flex-wrap: wrap;
            gap: 8px;
        }
        .top-nav-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 36px;
            padding: 0 13px;
            border-radius: 999px;
            border: 1px solid rgba(165,197,204,.13);
            color: var(--text-soft) !important;
            text-decoration: none !important;
            background: rgba(165,197,204,.055);
            font-size: .80rem;
            font-weight: 900;
            transition: transform .16s ease, border-color .16s ease, color .16s ease, background .16s ease, box-shadow .16s ease;
        }
        .top-nav-link:hover {
            transform: translateY(-2px);
            color: var(--gold) !important;
            border-color: rgba(253,199,135,.34);
            background: rgba(253,199,135,.10);
            box-shadow: 0 14px 34px rgba(0,0,0,.22);
        }
        .top-nav-link.active {
            color: var(--ink) !important;
            border-color: rgba(253,199,135,.48);
            background: linear-gradient(135deg, var(--gold), var(--mist));
            box-shadow: 0 14px 36px rgba(253,199,135,.20);
        }
        .top-nav-meta {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-left: 6px;
            padding: 0 12px;
            min-height: 32px;
            border-radius: 999px;
            color: var(--muted) !important;
            border: 1px solid rgba(165,197,204,.12);
            background: rgba(2,19,52,.42);
            font-size: .72rem;
            font-weight: 800;
        }
        .hero-action-note {
            max-width: 740px;
            margin-top: 11px;
            color: var(--muted) !important;
            font-size: .82rem;
            line-height: 1.55;
        }
        .hero-action-note b {
            color: var(--gold) !important;
            font-weight: 950;
        }
        @media (max-width: 760px) {
            .top-nav-shell {
                position: relative;
                align-items: stretch;
                flex-direction: column;
                border-radius: 24px;
                margin-top: 0;
            }
            .top-nav-brand { justify-content: center; padding-left: 0; }
            .top-nav-links { justify-content: center; }
            .top-nav-link { flex: 1 1 42%; }
            .top-nav-meta { width: 100%; justify-content: center; margin-left: 0; }
        }

        .nav-intro {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin: 20px 0 10px;
            padding: 12px 15px;
            border-radius: 20px;
            border: 1px solid rgba(165,197,204,.16);
            background: linear-gradient(135deg, rgba(1,42,97,.34), rgba(2,19,52,.72));
            box-shadow: 0 18px 52px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .nav-intro span {
            color: var(--gold) !important;
            font-size: .76rem;
            font-weight: 950;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .nav-intro b {
            color: var(--muted) !important;
            font-size: .82rem;
            font-weight: 800;
            text-align: right;
        }

        @media (max-width: 900px) {
            .hero-panel { transform: none; }
            .hero-grid { grid-template-columns: 1fr; }
            .hero { min-height: auto; }
            .launcher-screen { min-height: 310px; }
            .hero h1 { font-size: clamp(3.0rem, 16vw, 4.8rem); }
            .active-filter-card { align-items: flex-start; flex-direction: column; }
        }


        .card-grid {
            display: grid;
            grid-template-columns: repeat(var(--cards-per-row, 3), minmax(0, 1fr));
            gap: 18px;
            align-items: stretch;
        }
        .card-grid .game-card { margin-bottom: 0; }
        @media (max-width: 1180px) { .card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 760px) { .card-grid { grid-template-columns: 1fr; } }


        /* Detail page + gameplay description patch */
        .game-desc {
            margin: 11px 0 0;
            color: rgba(238,248,250,.78) !important;
            font-size: .82rem;
            line-height: 1.55;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .detail-hero {
            position: relative;
            overflow: hidden;
            margin: 2px 0 24px;
            padding: clamp(20px, 3.2vw, 38px);
            border-radius: 34px;
            border: 1px solid rgba(253,199,135,.32);
            background:
                radial-gradient(circle at 78% 18%, rgba(253,199,135,.22), transparent 18rem),
                radial-gradient(circle at 18% 0%, rgba(39,90,145,.42), transparent 28rem),
                linear-gradient(135deg, rgba(2,19,52,.96), rgba(1,42,97,.72) 54%, rgba(2,8,23,.96));
            box-shadow: 0 40px 120px rgba(0,0,0,.48), inset 0 1px 0 rgba(255,255,255,.08);
        }
        .detail-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(105deg, rgba(253,199,135,.12), transparent 34%),
                repeating-linear-gradient(116deg, rgba(165,197,204,.045) 0 1px, transparent 1px 26px);
        }
        .detail-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(300px, .82fr) minmax(0, 1.18fr);
            gap: clamp(20px, 3vw, 34px);
            align-items: stretch;
        }
        .detail-cover {
            position: relative;
            overflow: hidden;
            min-height: 310px;
            border-radius: 26px;
            background:
                radial-gradient(circle at 80% 10%, rgba(253,199,135,.18), transparent 10rem),
                linear-gradient(135deg, rgba(39,90,145,.38), rgba(2,19,52,.88));
            border: 1px solid rgba(165,197,204,.18);
            box-shadow: 0 28px 80px rgba(0,0,0,.40), inset 0 1px 0 rgba(255,255,255,.08);
        }
        .detail-cover img {
            width: 100%;
            height: 100%;
            min-height: 310px;
            object-fit: cover;
            display: block;
            filter: saturate(1.08) contrast(1.05);
            transform: scale(1.01);
        }
        .detail-cover::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(to top, rgba(2,19,52,.72), transparent 58%);
            pointer-events: none;
        }
        .detail-copy {
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .back-link {
            position: relative;
            z-index: 2;
            display: inline-flex;
            width: fit-content;
            align-items: center;
            gap: 8px;
            margin-bottom: 16px;
            padding: 8px 13px;
            border-radius: 999px;
            color: var(--text-soft) !important;
            text-decoration: none !important;
            font-size: .78rem;
            font-weight: 900;
            background: rgba(165,197,204,.075);
            border: 1px solid rgba(165,197,204,.16);
        }
        .back-link:hover {
            color: var(--gold) !important;
            border-color: rgba(253,199,135,.34);
            background: rgba(253,199,135,.10);
        }
        .detail-kicker {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            color: var(--gold) !important;
            background: rgba(253,199,135,.10);
            border: 1px solid rgba(253,199,135,.24);
            font-size: .76rem;
            font-weight: 950;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .detail-title {
            margin: 13px 0 12px;
            color: #fff !important;
            font-size: clamp(2.25rem, 5.4vw, 5.6rem);
            line-height: .90;
            letter-spacing: -.075em;
            text-shadow: 0 18px 58px rgba(0,0,0,.42);
        }
        .detail-description {
            max-width: 900px;
            color: rgba(238,248,250,.84) !important;
            font-size: clamp(.98rem, 1.25vw, 1.18rem);
            line-height: 1.78;
            margin: 0 0 16px;
        }
        .detail-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 19px;
        }
        .detail-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            padding: 0 18px;
            border-radius: 999px;
            text-decoration: none !important;
            font-weight: 950;
            transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
        }
        .detail-cta.primary {
            color: var(--ink) !important;
            background: linear-gradient(135deg, var(--gold), #ffe1aa 45%, var(--mist));
            border: 1px solid rgba(253,199,135,.50);
            box-shadow: 0 18px 52px rgba(253,199,135,.24), 0 0 34px rgba(253,199,135,.13);
        }
        .detail-cta.secondary {
            color: var(--text) !important;
            background: rgba(165,197,204,.08);
            border: 1px solid rgba(165,197,204,.18);
        }
        .detail-cta:hover { transform: translateY(-3px); filter: brightness(1.05); }
        .detail-panels {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 18px 0 24px;
        }
        .detail-panel {
            min-height: 140px;
            border-radius: 24px;
            padding: 17px;
            background: linear-gradient(180deg, rgba(1,42,97,.38), rgba(2,19,52,.76));
            border: 1px solid rgba(165,197,204,.16);
            box-shadow: 0 20px 60px rgba(0,0,0,.26), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .detail-panel b {
            display: block;
            margin-bottom: 8px;
            color: #fff !important;
            font-size: .98rem;
        }
        .detail-panel span, .detail-panel p {
            color: var(--muted) !important;
            font-size: .84rem;
            line-height: 1.55;
            margin: 0;
        }
        @media (max-width: 980px) {
            .detail-grid { grid-template-columns: 1fr; }
            .detail-cover, .detail-cover img { min-height: 230px; }
            .detail-panels { grid-template-columns: 1fr; }
        }

        </style>
        """
    )

# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def clean_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_name(c) for c in df.columns]
    aliases = {
        "app_id": ["appid", "steam_appid", "steam_id", "id"],
        "name": ["title", "game", "game_name"],
        "release_date": ["release", "date", "released", "release_year"],
        "price_usd": ["price", "initial_price", "final_price", "price_dollar", "usd_price"],
        "discount_pct": ["discount", "discount_percent", "discount_percentage"],
        "metacritic_score": ["metacritic", "meta_score"],
        "recommendations": ["recommendation_count", "recommendation", "reviews", "review_count"],
        "positive_reviews": ["positive", "positive_review", "positive_ratings"],
        "negative_reviews": ["negative", "negative_review", "negative_ratings"],
        "avg_playtime_forever": ["average_playtime", "avg_playtime", "playtime_forever"],
        "avg_playtime_2weeks": ["playtime_2weeks", "avg_2weeks"],
        "median_playtime": ["median_playtime_forever"],
        "peak_ccu": ["peak_players", "ccu", "concurrent_users"],
        "required_age": ["age", "required_age_years"],
        "dlc_count": ["dlcs", "dlc"],
        "achievements": ["achievement_count"],
        "genres": ["genre"],
        "categories": ["category"],
        "tags": ["tag", "steamspy_tags"],
        "developer": ["developers"],
        "publisher": ["publishers"],
        "short_description": ["description", "about", "short_desc"],
        "header_image": ["image", "thumbnail", "capsule_image", "cover"],
        "estimated_owners": ["owners", "owner_range"],
        "is_free": ["free", "free_to_play"],
    }
    for canonical, variants in aliases.items():
        if canonical in df.columns:
            continue
        for variant in variants:
            if variant in df.columns:
                df = df.rename(columns={variant: canonical})
                break
    return df


def split_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    text = re.sub(r"[\[\]\{\}\(\)'\"]", " ", text)
    text = text.replace("/", ",")
    parts = re.split(r"[,;|]+", text)
    tokens = []
    seen = set()
    for part in parts:
        token = re.sub(r"\s+", " ", part).strip()
        if token and token.lower() not in seen and token.lower() not in {"nan", "none", "null"}:
            tokens.append(token)
            seen.add(token.lower())
    return tokens


def parse_owners(value: object) -> float:
    if pd.isna(value):
        return np.nan
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", str(value))]
    if not nums:
        return np.nan
    return float(np.mean(nums) / 1_000_000)


def to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def to_bool(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes", "y", "free", "f2p"}
    false_values = {"false", "0", "no", "n", "paid", ""}

    def _convert(x: object) -> bool:
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        val = str(x).strip().lower()
        if val in true_values:
            return True
        if val in false_values:
            return False
        return False

    return series.apply(_convert).astype(bool)


def robust_minmax(series: pd.Series, invert: bool = False, default: float = 0.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.notna().sum() == 0:
        out = pd.Series(default, index=series.index, dtype=float)
        return 1 - out if invert else out
    q_low = s.quantile(0.01)
    q_high = s.quantile(0.99)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        out = pd.Series(default, index=series.index, dtype=float)
    else:
        out = (s.clip(q_low, q_high) - q_low) / (q_high - q_low)
        out = out.fillna(default).clip(0, 1)
    return 1 - out if invert else out


def percentage_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().gt(1).any():
        return (s / 100).clip(0, 1).fillna(0.5)
    return s.clip(0, 1).fillna(0.5)


def weighted_content_text(row: pd.Series) -> str:
    tags = split_tokens(row.get("tags", ""))
    genres = split_tokens(row.get("genres", ""))
    categories = split_tokens(row.get("categories", ""))
    developer = split_tokens(row.get("developer", ""))
    publisher = split_tokens(row.get("publisher", ""))
    desc = str(row.get("short_description", ""))
    parts: list[str] = []
    parts.extend(tags * 5)
    parts.extend(genres * 4)
    parts.extend(categories * 2)
    parts.extend(developer * 2)
    parts.extend(publisher)
    parts.extend([desc] * 2)
    cleaned = " ".join(parts).lower()
    return cleaned if cleaned.strip() else "unknown game"


REQUIRED_COLUMNS = {
    "app_id": np.nan,
    "name": "Unknown Game",
    "release_date": "",
    "price_usd": np.nan,
    "discount_pct": 0,
    "metacritic_score": np.nan,
    "recommendations": 0,
    "positive_reviews": 0,
    "negative_reviews": 0,
    "avg_playtime_forever": np.nan,
    "avg_playtime_2weeks": np.nan,
    "median_playtime": np.nan,
    "peak_ccu": np.nan,
    "required_age": np.nan,
    "dlc_count": 0,
    "achievements": 0,
    "genres": "",
    "categories": "",
    "tags": "",
    "developer": "",
    "publisher": "",
    "short_description": "",
    "header_image": "",
    "estimated_owners": "",
    "is_free": False,
}


def prepare_games(raw: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(raw)
    for col, default in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    df = df.copy().reset_index(drop=True)
    if df["app_id"].isna().all():
        df["app_id"] = np.arange(1, len(df) + 1)

    numeric_cols = [
        "price_usd",
        "discount_pct",
        "metacritic_score",
        "recommendations",
        "positive_reviews",
        "negative_reviews",
        "avg_playtime_forever",
        "avg_playtime_2weeks",
        "median_playtime",
        "peak_ccu",
        "required_age",
        "dlc_count",
        "achievements",
    ]
    for col in numeric_cols:
        df[col] = to_number(df[col])

    text_cols = [
        "name",
        "release_date",
        "genres",
        "categories",
        "tags",
        "developer",
        "publisher",
        "short_description",
        "header_image",
        "estimated_owners",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).replace("nan", "")

    df["is_free"] = to_bool(df["is_free"])
    df.loc[df["price_usd"].fillna(np.inf) <= 0, "is_free"] = True

    if df["release_date"].str.fullmatch(r"\d{4}(\.0)?").all():
        df["year"] = pd.to_numeric(df["release_date"], errors="coerce")
    else:
        df["year"] = df["release_date"].str.extract(r"((?:19|20)\d{2})")[0]
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Clean obvious playtime sentinels without erasing valid long games.
    for col in ["avg_playtime_forever", "avg_playtime_2weeks", "median_playtime"]:
        if df[col].notna().sum() > 20:
            upper = df[col].quantile(0.995)
            df[col] = df[col].where(df[col] <= upper, np.nan)

    df["genre_list"] = df["genres"].apply(split_tokens)
    df["tag_list"] = df["tags"].apply(split_tokens)
    df["category_list"] = df["categories"].apply(split_tokens)
    df["genre_primary"] = df["genre_list"].apply(lambda x: x[0] if x else "Unknown")

    combined = (
        df["categories"].fillna("") + " " + df["tags"].fillna("") + " " + df["genres"].fillna("")
    ).str.lower()
    df["is_singleplayer"] = combined.str.contains("single-player|single player|singleplayer", regex=True, na=False)
    df["is_multiplayer"] = combined.str.contains("multi-player|multiplayer|online pvp|pvp", regex=True, na=False)
    df["is_coop"] = combined.str.contains("co-op|coop|cooperative", regex=True, na=False)

    df["total_reviews"] = df["positive_reviews"].fillna(0) + df["negative_reviews"].fillna(0)
    df["review_volume"] = df[["recommendations", "total_reviews"]].max(axis=1).fillna(0)
    df["positivity"] = np.where(
        df["total_reviews"] > 0,
        (df["positive_reviews"] / df["total_reviews"] * 100),
        np.nan,
    )
    # Fallback: if review polarity is unavailable, use metacritic as imperfect rating proxy.
    df["positivity"] = df["positivity"].fillna(df["metacritic_score"])

    valid_rating = df["positivity"].dropna()
    C = float(valid_rating.mean()) if len(valid_rating) else 70.0
    m = float(df["review_volume"].quantile(0.70)) if df["review_volume"].notna().any() else 50.0
    if not np.isfinite(m) or m <= 0:
        m = 50.0
    v = df["review_volume"].fillna(0)
    R = df["positivity"].fillna(C)
    df["bayes_rating"] = ((v / (v + m)) * R + (m / (v + m)) * C).clip(0, 100)

    df["owners_m"] = df["estimated_owners"].apply(parse_owners)
    df["price_effective"] = np.where(df["is_free"], 0.0, df["price_usd"].fillna(df["price_usd"].median()))
    df["playtime_h"] = df["avg_playtime_forever"] / 60

    df["rating_score"] = (df["bayes_rating"] / 100).fillna(0.5).clip(0, 1)
    df["popularity_score"] = robust_minmax(np.log1p(df["review_volume"].fillna(0)))
    df["metacritic_norm"] = percentage_series(df["metacritic_score"])
    df["playtime_score"] = robust_minmax(np.log1p(df["avg_playtime_forever"].fillna(0)))
    df["recency_score"] = robust_minmax(df["year"].fillna(df["year"].median()))
    df["affordability_score"] = robust_minmax(df["price_effective"].fillna(0), invert=True)
    df["discount_score"] = percentage_series(df["discount_pct"].fillna(0))
    df["novelty_score"] = (1 - df["popularity_score"]).clip(0, 1)

    df["quality_score"] = (
        0.34 * df["rating_score"]
        + 0.22 * df["popularity_score"]
        + 0.16 * df["metacritic_norm"]
        + 0.12 * df["playtime_score"]
        + 0.10 * df["recency_score"]
        + 0.06 * df["affordability_score"]
    ).clip(0, 1)
    df["crowd_score"] = (
        0.52 * df["rating_score"]
        + 0.32 * df["popularity_score"]
        + 0.11 * df["metacritic_norm"]
        + 0.05 * df["playtime_score"]
    ).clip(0, 1)
    df["value_score"] = (
        0.48 * df["quality_score"]
        + 0.32 * df["affordability_score"]
        + 0.12 * df["discount_score"]
        + 0.08 * df["rating_score"]
    ).clip(0, 1)
    df["display_score"] = (df["quality_score"] * 100).round(1)
    df["content_text"] = df.apply(weighted_content_text, axis=1)

    return df


@st.cache_data(show_spinner=False)
def load_games_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return prepare_games(raw)


@st.cache_data(show_spinner=False)
def load_games_from_path(path_text: str) -> pd.DataFrame:
    raw = pd.read_csv(path_text)
    return prepare_games(raw)


@st.cache_resource(show_spinner=False)
def build_tfidf(texts: tuple[str, ...]):
    safe_texts = tuple(t if str(t).strip() else "unknown game" for t in texts)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=20000,
        token_pattern=r"(?u)\b[\w\-]+\b",
    )
    matrix = vectorizer.fit_transform(safe_texts)
    return vectorizer, matrix


@st.cache_data(show_spinner=False)
def load_interactions_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return canonicalize_columns(pd.read_csv(io.BytesIO(file_bytes)))


# -----------------------------------------------------------------------------
# Recommendation functions
# -----------------------------------------------------------------------------
def top_values_from_lists(df: pd.DataFrame, list_col: str, limit: int = 80) -> list[str]:
    counter: Counter[str] = Counter()
    if list_col not in df.columns:
        return []
    for values in df[list_col]:
        if isinstance(values, list):
            counter.update(values)
    return [name for name, _ in counter.most_common(limit)]


def normalize_array(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, default, dtype=float)
    clean = arr.copy()
    clean[~finite] = np.nan
    mn = np.nanmin(clean)
    mx = np.nanmax(clean)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.full_like(arr, default if default else 0.5, dtype=float)
    clean = (clean - mn) / (mx - mn)
    clean = np.nan_to_num(clean, nan=default, posinf=1.0, neginf=0.0)
    return np.clip(clean, 0, 1)


def content_scores(
    games: pd.DataFrame,
    matrix,
    vectorizer: TfidfVectorizer,
    favorite_titles: Sequence[str],
    preferred_genres: Sequence[str],
    preferred_tags: Sequence[str],
    mood_terms: Sequence[str],
) -> np.ndarray:
    n = len(games)
    score = np.zeros(n, dtype=float)
    weight_total = 0.0

    if favorite_titles:
        title_lookup = {str(name).lower(): idx for idx, name in games["name"].items()}
        fav_indices = [title_lookup[t.lower()] for t in favorite_titles if t.lower() in title_lookup]
        if fav_indices:
            fav_sim = cosine_similarity(matrix[fav_indices], matrix).mean(axis=0)
            score += 0.72 * np.asarray(fav_sim).ravel()
            weight_total += 0.72

    profile_terms: list[str] = []
    profile_terms.extend(list(preferred_genres) * 4)
    profile_terms.extend(list(preferred_tags) * 5)
    profile_terms.extend(list(mood_terms) * 3)
    if profile_terms:
        query_text = " ".join(profile_terms).lower()
        query_vec = vectorizer.transform([query_text])
        term_sim = cosine_similarity(query_vec, matrix).ravel()
        score += 0.28 * term_sim
        weight_total += 0.28

    if weight_total <= 0:
        return np.zeros(n, dtype=float)
    return np.clip(score / weight_total, 0, 1)


def rule_scores(
    games: pd.DataFrame,
    preferred_genres: Sequence[str],
    preferred_tags: Sequence[str],
    max_price: float,
    min_positivity: float,
    mode: str,
) -> np.ndarray:
    genre_set = {g.lower() for g in preferred_genres}
    tag_set = {t.lower() for t in preferred_tags}
    scores = []
    for _, row in games.iterrows():
        score = 0.0
        score += 0.35 * float(row.get("quality_score", 0.5))
        score += 0.15 * float(row.get("affordability_score", 0.5))

        if genre_set:
            row_genres = {g.lower() for g in row.get("genre_list", [])}
            score += 0.20 * (len(row_genres & genre_set) / max(1, len(genre_set)))
        else:
            score += 0.10

        if tag_set:
            row_tags = {t.lower() for t in row.get("tag_list", [])}
            score += 0.22 * (len(row_tags & tag_set) / max(1, len(tag_set)))
        else:
            score += 0.08

        price = float(row.get("price_effective", np.nan))
        if bool(row.get("is_free", False)) or (np.isfinite(price) and price <= max_price):
            score += 0.06
        pos = float(row.get("positivity", np.nan))
        if np.isfinite(pos) and pos >= min_positivity:
            score += 0.05
        if mode == "singleplayer" and bool(row.get("is_singleplayer", False)):
            score += 0.07
        elif mode == "multiplayer" and bool(row.get("is_multiplayer", False)):
            score += 0.07
        elif mode == "coop" and bool(row.get("is_coop", False)):
            score += 0.07
        elif mode == "any":
            score += 0.04
        scores.append(score)
    return np.clip(np.asarray(scores, dtype=float), 0, 1)


def apply_candidate_filters(
    games: pd.DataFrame,
    max_price: float,
    min_positivity: float,
    min_reviews: int,
    preferred_genres: Sequence[str],
    must_have_tags: Sequence[str],
    mode: str,
    exclude_titles: Sequence[str],
) -> pd.DataFrame:
    res = games.copy()
    price_ok = (res["price_effective"].fillna(np.inf) <= max_price) | res["is_free"].fillna(False)
    res = res[price_ok]
    res = res[res["positivity"].fillna(0) >= min_positivity]
    res = res[res["review_volume"].fillna(0) >= min_reviews]

    if preferred_genres:
        genre_set = {g.lower() for g in preferred_genres}
        res = res[res["genre_list"].apply(lambda xs: bool({x.lower() for x in xs} & genre_set))]

    for tag in must_have_tags:
        res = res[res["tag_list"].apply(lambda xs, t=tag: any(x.lower() == t.lower() for x in xs))]

    if mode == "singleplayer":
        res = res[res["is_singleplayer"]]
    elif mode == "multiplayer":
        res = res[res["is_multiplayer"]]
    elif mode == "coop":
        res = res[res["is_coop"]]

    if exclude_titles:
        exclude = {t.lower() for t in exclude_titles}
        res = res[~res["name"].str.lower().isin(exclude)]

    return res


def build_interaction_cf_scores(
    games: pd.DataFrame,
    interactions: pd.DataFrame | None,
    favorite_titles: Sequence[str],
) -> np.ndarray | None:
    if interactions is None or interactions.empty or sparse is None or not favorite_titles:
        return None

    df_int = canonicalize_columns(interactions)
    if "user_id" not in df_int.columns:
        for candidate in ["user", "uid", "steamid", "steam_id"]:
            if candidate in df_int.columns:
                df_int = df_int.rename(columns={candidate: "user_id"})
                break
    if "user_id" not in df_int.columns:
        return None

    id_col = None
    if "app_id" in df_int.columns and "app_id" in games.columns:
        id_col = "app_id"
    elif "name" in df_int.columns:
        id_col = "name"
    else:
        return None

    if "rating" in df_int.columns:
        values = to_number(df_int["rating"]).fillna(0).clip(lower=0)
    elif "playtime_forever" in df_int.columns:
        values = np.log1p(to_number(df_int["playtime_forever"]).fillna(0))
    elif "liked" in df_int.columns:
        values = to_bool(df_int["liked"]).astype(int)
    else:
        values = pd.Series(1.0, index=df_int.index)

    if id_col == "app_id":
        item_map = pd.Series(games.index.values, index=games["app_id"].astype(str)).to_dict()
        item_idx = df_int["app_id"].astype(str).map(item_map)
    else:
        item_map = pd.Series(games.index.values, index=games["name"].str.lower()).to_dict()
        item_idx = df_int["name"].astype(str).str.lower().map(item_map)

    valid = item_idx.notna() & df_int["user_id"].notna() & values.notna() & (values > 0)
    if valid.sum() < 3:
        return None

    users = pd.factorize(df_int.loc[valid, "user_id"].astype(str))[0]
    items = item_idx.loc[valid].astype(int).to_numpy()
    vals = values.loc[valid].astype(float).to_numpy()
    mat = sparse.csr_matrix((vals, (users, items)), shape=(users.max() + 1, len(games)))

    title_lookup = {str(name).lower(): idx for idx, name in games["name"].items()}
    fav_indices = [title_lookup[t.lower()] for t in favorite_titles if t.lower() in title_lookup]
    if not fav_indices:
        return None
    item_user = mat.T.tocsr()
    sims = cosine_similarity(item_user[fav_indices], item_user).mean(axis=0)
    return np.asarray(sims).ravel()


def mmr_rerank(
    candidates: pd.DataFrame,
    matrix,
    score_col: str,
    top_n: int,
    diversity: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    pool_size = min(len(candidates), max(top_n * 12, 80))
    pool = candidates.sort_values(score_col, ascending=False).head(pool_size).copy()
    if diversity <= 0 or len(pool) <= top_n:
        return pool.head(top_n)

    rel = normalize_array(pool[score_col].to_numpy(), default=0.5)
    idxs = pool.index.to_list()
    selected_positions: list[int] = []
    remaining_positions = list(range(len(idxs)))
    lambda_rel = float(np.clip(1 - diversity, 0.35, 0.95))

    while remaining_positions and len(selected_positions) < top_n:
        if not selected_positions:
            best = max(remaining_positions, key=lambda p: rel[p])
        else:
            remaining_idxs = [idxs[p] for p in remaining_positions]
            selected_idxs = [idxs[p] for p in selected_positions]
            sim_to_selected = cosine_similarity(matrix[remaining_idxs], matrix[selected_idxs]).max(axis=1)
            mmr_values = []
            for local_i, p in enumerate(remaining_positions):
                mmr_values.append(lambda_rel * rel[p] - diversity * float(sim_to_selected[local_i]))
            best = remaining_positions[int(np.argmax(mmr_values))]
        selected_positions.append(best)
        remaining_positions.remove(best)

    selected_indices = [idxs[p] for p in selected_positions]
    return pool.loc[selected_indices]


def recommend_games(
    games: pd.DataFrame,
    matrix,
    vectorizer: TfidfVectorizer,
    engine: str,
    favorite_titles: Sequence[str],
    preferred_genres: Sequence[str],
    preferred_tags: Sequence[str],
    must_have_tags: Sequence[str],
    mood_terms: Sequence[str],
    max_price: float,
    min_positivity: float,
    min_reviews: int,
    mode: str,
    top_n: int,
    diversity: float,
    weights: dict[str, float],
    interactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidate_df = apply_candidate_filters(
        games=games,
        max_price=max_price,
        min_positivity=min_positivity,
        min_reviews=min_reviews,
        preferred_genres=preferred_genres,
        must_have_tags=must_have_tags,
        mode=mode,
        exclude_titles=favorite_titles,
    )
    if candidate_df.empty:
        return candidate_df

    content = content_scores(games, matrix, vectorizer, favorite_titles, preferred_genres, preferred_tags, mood_terms)
    rule = rule_scores(games, preferred_genres, preferred_tags, max_price, min_positivity, mode)
    cf_true = build_interaction_cf_scores(games, interactions, favorite_titles)
    if cf_true is None:
        cf = games["crowd_score"].to_numpy(dtype=float)
        cf_label = "Crowd proxy"
    else:
        cf = normalize_array(cf_true, default=0.0)
        cf_label = "User-item CF"

    scores = pd.DataFrame(
        {
            "content_component": normalize_array(content, default=0.0),
            "rule_component": normalize_array(rule, default=0.5),
            "crowd_component": normalize_array(cf, default=0.5),
            "quality_component": games["quality_score"].to_numpy(dtype=float),
            "value_component": games["value_score"].to_numpy(dtype=float),
            "novelty_component": games["novelty_score"].to_numpy(dtype=float),
        },
        index=games.index,
    )

    if engine == "Content-Based":
        final = 0.78 * scores["content_component"] + 0.14 * scores["quality_component"] + 0.08 * scores["value_component"]
    elif engine == "Rule-Based":
        final = 0.65 * scores["rule_component"] + 0.22 * scores["quality_component"] + 0.13 * scores["value_component"]
    elif engine == "Collaborative / Crowd":
        final = 0.74 * scores["crowd_component"] + 0.16 * scores["quality_component"] + 0.10 * scores["novelty_component"]
    else:
        total_w = max(1e-9, sum(max(0.0, v) for v in weights.values()))
        normalized = {k: max(0.0, v) / total_w for k, v in weights.items()}
        final = (
            normalized.get("content", 0.0) * scores["content_component"]
            + normalized.get("crowd", 0.0) * scores["crowd_component"]
            + normalized.get("rule", 0.0) * scores["rule_component"]
            + normalized.get("value", 0.0) * scores["value_component"]
            + normalized.get("novelty", 0.0) * scores["novelty_component"]
        )

    out = candidate_df.join(scores, how="left")
    out["final_score"] = final.loc[out.index].clip(0, 1)
    out["final_score_pct"] = (out["final_score"] * 100).round(1)
    out["cf_source"] = cf_label
    out = mmr_rerank(out, matrix, "final_score", top_n, diversity)
    return out.sort_values("final_score", ascending=False).head(top_n)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def esc(value: object) -> str:
    return html.escape("" if pd.isna(value) else str(value))


def fmt_int(value: object) -> str:
    try:
        if not np.isfinite(float(value)):
            return "-"
        return f"{int(float(value)):,}"
    except Exception:
        return "-"


def fmt_float(value: object, digits: int = 1, suffix: str = "") -> str:
    try:
        val = float(value)
        if not np.isfinite(val):
            return "-"
        return f"{val:.{digits}f}{suffix}"
    except Exception:
        return "-"


def steam_url(row: pd.Series) -> str:
    try:
        app_id = int(float(row.get("app_id", np.nan)))
        if app_id > 0:
            return f"https://store.steampowered.com/app/{app_id}/"
    except Exception:
        pass
    return ""


def query_value(name: str, default: str = "") -> str:
    """Read one query-param value across Streamlit versions."""
    try:
        value = st.query_params.get(name, default)
        if isinstance(value, list):
            return str(value[0]) if value else default
        return str(value) if value is not None else default
    except Exception:
        return default


def match_known_value(raw: str, options: Sequence[str]) -> str:
    """Return the existing option with matching casing, if available."""
    raw_clean = str(raw or "").strip()
    if not raw_clean:
        return ""
    lookup = {str(option).lower(): str(option) for option in options}
    return lookup.get(raw_clean.lower(), raw_clean)


def app_link(view: str = "Explore", tag: str | None = None, game: str | None = None, anchor: str | None = None) -> str:
    """Create an in-app same-tab navigation link using query params."""
    params = [f"view={quote(str(view), safe='')}"]
    if tag:
        params.append(f"tag={quote(str(tag), safe='')}")
    if game:
        params.append(f"game={quote(str(game), safe='')}")
    suffix = f"#{anchor}" if anchor else ""
    return "?" + "&".join(params) + suffix


def tag_link(tag: str, active_tag: str = "") -> str:
    safe = esc(tag)
    active = " tag-active" if active_tag and active_tag.lower() == str(tag).lower() else ""
    href = app_link("Explore", tag=tag, anchor="content-start")
    return f'<a class="tag{active}" href="{href}" target="_top" title="Show more {safe} games in this page">{safe}</a>'


def clean_game_text(value: object) -> str:
    """Plain-text cleanup for Steam descriptions so raw HTML never appears in cards."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and pd.isna(value):
            return ""
    except Exception:
        pass
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def shorten_text(text: str, limit: int = 170) -> str:
    text = clean_game_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rsplit(" ", 1)[0].rstrip(".,;:") + "…"


def gameplay_description(row: pd.Series, limit: int = 180) -> str:
    """Use dataset short_description first, then generate a small gameplay-like summary."""
    desc = clean_game_text(row.get("short_description", ""))
    if desc:
        return shorten_text(desc, limit)
    title = clean_game_text(row.get("name", "this game")) or "This game"
    genre = clean_game_text(row.get("genre_primary", "game")) or "game"
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    tag_part = ", ".join(str(t) for t in tags[:3])
    if tag_part:
        fallback = f"{title} is a {genre} title built around {tag_part} elements, surfaced here through quality, crowd, value, and content signals."
    else:
        fallback = f"{title} is a {genre} title surfaced through quality, crowd, value, and content signals."
    return shorten_text(fallback, limit)


def game_key(row: pd.Series) -> str:
    try:
        app_id = int(float(row.get("app_id", np.nan)))
        if app_id > 0:
            return str(app_id)
    except Exception:
        pass
    try:
        return f"idx-{int(row.name)}"
    except Exception:
        return quote(str(row.get("name", "game")), safe="")


def detail_link(row: pd.Series, anchor: str | None = "content-start") -> str:
    return app_link("Detail", game=game_key(row), anchor=anchor)


def find_game_by_key(games: pd.DataFrame, key: str) -> pd.Series | None:
    key_clean = str(key or "").strip()
    if not key_clean:
        return None
    try:
        normalized_number = str(int(float(key_clean)))
    except Exception:
        normalized_number = key_clean
    for _, row in games.iterrows():
        if game_key(row) == normalized_number or game_key(row) == key_clean:
            return row
    lower_key = key_clean.lower()
    matched = games[games["name"].astype(str).str.lower() == lower_key]
    if not matched.empty:
        return matched.iloc[0]
    return None


def price_badge(row: pd.Series) -> str:
    if bool(row.get("is_free", False)):
        return "<span class='pill pill-green'>Free</span>"
    price = row.get("price_effective", np.nan)
    if pd.notna(price):
        base = f"<span class='pill pill-blue'>${float(price):.2f}</span>"
    else:
        base = "<span class='pill'>Price n/a</span>"
    discount = row.get("discount_pct", 0)
    try:
        if float(discount) > 0:
            base += f"<span class='pill pill-red'>-{int(float(discount))}%</span>"
    except Exception:
        pass
    return base


def component_bar(label: str, value: float) -> str:
    value = float(np.clip(value if np.isfinite(value) else 0, 0, 1))
    pct = int(round(value * 100))
    return textwrap.dedent(f"""
    <div class='bar-row'>
      <div class='bar-label'><span>{esc(label)}</span><span>{pct}</span></div>
      <div class='bar-track'><div class='bar-fill' style='width:{pct}%'></div></div>
    </div>
    """).strip()


def explain_row(row: pd.Series, games: pd.DataFrame, favorite_titles: Sequence[str], preferred_tags: Sequence[str]) -> str:
    reasons: list[str] = []
    if favorite_titles:
        fav_rows = games[games["name"].isin(favorite_titles)]
        fav_tags = set()
        fav_genres = set()
        for _, fav in fav_rows.iterrows():
            fav_tags.update([x.lower() for x in fav.get("tag_list", [])])
            fav_genres.update([x.lower() for x in fav.get("genre_list", [])])
        row_tags = {x.lower() for x in row.get("tag_list", [])}
        row_genres = {x.lower() for x in row.get("genre_list", [])}
        shared_tags = [t for t in row.get("tag_list", []) if t.lower() in fav_tags][:4]
        shared_genres = [g for g in row.get("genre_list", []) if g.lower() in fav_genres][:3]
        if shared_tags:
            reasons.append("similar tags: " + ", ".join(shared_tags))
        elif shared_genres:
            reasons.append("similar genres: " + ", ".join(shared_genres))
    if preferred_tags:
        matched = [t for t in row.get("tag_list", []) if t.lower() in {x.lower() for x in preferred_tags}][:4]
        if matched:
            reasons.append("matches preference: " + ", ".join(matched))
    try:
        if float(row.get("bayes_rating", 0)) >= 85:
            reasons.append("strong Bayesian crowd rating")
    except Exception:
        pass
    try:
        if float(row.get("value_score", 0)) >= 0.72:
            reasons.append("good value for money")
    except Exception:
        pass
    if bool(row.get("is_free", False)):
        reasons.append("free to play")
    if not reasons:
        reasons.append("high combined recommendation score")
    return "; ".join(reasons[:3])


def game_card_html(
    row: pd.Series,
    games: pd.DataFrame,
    favorite_titles: Sequence[str] = (),
    preferred_tags: Sequence[str] = (),
    show_components: bool = False,
    rank: int | None = None,
    active_tag: str = "",
) -> str:
    title = esc(row.get("name", "Unknown Game"))
    detail_href = esc(detail_link(row))
    img = esc(str(row.get("header_image", "")).strip())
    initials = "".join([part[:1] for part in re.findall(r"[A-Za-z0-9]+", title)[:2]]).upper() or "SV"
    fallback = f'<div class="game-img-fallback"><div><span>{esc(initials)}</span><b>{title}</b></div></div>'
    media_inner = f'{fallback}<img src="{img}" alt="{title} cover" loading="lazy">' if img else fallback
    img_html = f'<a class="cover-link" href="{detail_href}" target="_top" aria-label="Open detail page for {title}">{media_inner}<span class="preview-chip">View details</span></a>'
    genre = esc(row.get("genre_primary", "Unknown"))
    year = fmt_int(row.get("year"))
    score = fmt_float(row.get("final_score_pct", row.get("display_score", 0)), 1)
    pos = fmt_float(row.get("positivity"), 1, "%")
    recs = fmt_int(row.get("review_volume"))
    play = fmt_float(row.get("playtime_h"), 1, "h")
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    tag_html = "".join(tag_link(str(t), active_tag=active_tag) for t in tags[:7])
    why = esc(explain_row(row, games, favorite_titles, preferred_tags))
    desc = esc(gameplay_description(row, limit=185))
    desc_html = f'<p class="game-desc">{desc}</p>' if desc else ""
    comp_html = ""
    if show_components:
        comp_html = (
            component_bar("Content match", float(row.get("content_component", 0)))
            + component_bar("Crowd signal", float(row.get("crowd_component", 0)))
            + component_bar("Rule fit", float(row.get("rule_component", 0)))
            + component_bar("Value", float(row.get("value_component", 0)))
        )
    rank_label = f"#{rank:02d}" if rank is not None else "Featured"
    first_tag = str(tags[0]) if tags else ""
    action_primary = f'<a class="card-action" href="{detail_href}" target="_top">View detail</a>'
    action_secondary = (
        f'<a class="card-action secondary" href="{app_link("Explore", tag=first_tag, anchor="content-start")}" target="_top">More like this</a>'
        if first_tag
        else '<span class="card-action secondary">More like this</span>'
    )
    card_markup = textwrap.dedent(f"""
    <article class="game-card">
      <div class="game-img-wrap">
        {img_html}
        <div class="game-topline">
          <span class="rank-badge">{rank_label}</span>
          <span class="score-badge">Score {score}</span>
        </div>
      </div>
      <div class="game-body">
        <div class="game-title"><a href="{detail_href}" target="_top">{title}</a></div>
        <div class="meta-line">{genre} | {year} | {price_badge(row)}</div>
        <div class="pill-row">
          <span class="pill pill-green">Pos {pos}</span>
          <span class="pill">Reviews {recs}</span>
          <span class="pill">Playtime {play}</span>
        </div>
        {desc_html}
        <div>{tag_html}</div>
        {comp_html}
        <div class="why"><span class="why-label">Why:</span> {why}</div>
        <div class="card-actions">{action_primary}{action_secondary}</div>
      </div>
    </article>
    """).strip()
    return re.sub(r">\s+<", "><", card_markup)

def render_cards(
    rows: pd.DataFrame,
    games: pd.DataFrame,
    favorite_titles: Sequence[str] = (),
    preferred_tags: Sequence[str] = (),
    columns: int = 3,
    show_components: bool = False,
    active_tag: str = "",
) -> None:
    if rows.empty:
        st.info("Tidak ada data yang cocok dengan filter saat ini.")
        return
    columns = int(max(1, min(4, columns)))
    cards_html = []
    for i, (_, row) in enumerate(rows.iterrows()):
        cards_html.append(
            game_card_html(
                row,
                games,
                favorite_titles=favorite_titles,
                preferred_tags=preferred_tags,
                show_components=show_components,
                rank=i + 1,
                active_tag=active_tag,
            )
        )
    render_html(f'<div class="card-grid" style="--cards-per-row:{columns};">{"".join(cards_html)}</div>')



def similar_games_for(row: pd.Series, games: pd.DataFrame, matrix, limit: int = 6) -> pd.DataFrame:
    """Find similar games for the detail page using content similarity plus quality signal."""
    if games.empty:
        return games.head(0)
    try:
        row_idx = int(row.name)
        sims = cosine_similarity(matrix[row_idx], matrix).ravel()
    except Exception:
        sims = np.zeros(len(games), dtype=float)
    out = games.copy()
    if len(sims) == len(out):
        out["detail_similarity"] = sims
    else:
        out["detail_similarity"] = 0.0
    row_tags = {str(t).lower() for t in row.get("tag_list", [])} if isinstance(row.get("tag_list", []), list) else set()
    row_genres = {str(g).lower() for g in row.get("genre_list", [])} if isinstance(row.get("genre_list", []), list) else set()
    def overlap_bonus(other: pd.Series) -> float:
        tags = {str(t).lower() for t in other.get("tag_list", [])} if isinstance(other.get("tag_list", []), list) else set()
        genres = {str(g).lower() for g in other.get("genre_list", [])} if isinstance(other.get("genre_list", []), list) else set()
        tag_score = len(tags & row_tags) / max(1, len(row_tags)) if row_tags else 0.0
        genre_score = len(genres & row_genres) / max(1, len(row_genres)) if row_genres else 0.0
        return 0.7 * tag_score + 0.3 * genre_score
    out["detail_overlap"] = out.apply(overlap_bonus, axis=1)
    out["detail_score"] = (
        0.52 * out["detail_similarity"].fillna(0)
        + 0.24 * out["detail_overlap"].fillna(0)
        + 0.24 * out["quality_score"].fillna(0.5)
    )
    out = out.drop(index=row.name, errors="ignore")
    out = out[out["name"].astype(str) != str(row.get("name", ""))]
    return out.sort_values("detail_score", ascending=False).head(limit)


def detail_panel_html(title: str, body: str) -> str:
    return f'<div class="detail-panel"><b>{esc(title)}</b><p>{esc(body)}</p></div>'


def render_game_detail(row: pd.Series, games: pd.DataFrame, matrix, active_tag: str = "") -> None:
    title = esc(row.get("name", "Unknown Game"))
    img = esc(str(row.get("header_image", "")).strip())
    cover_html = f'<img src="{img}" alt="{title} cover" loading="lazy">' if img else f'<div class="game-img-fallback"><div><span>SV</span><b>{title}</b></div></div>'
    desc = esc(gameplay_description(row, limit=650))
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    tag_html = "".join(tag_link(str(t), active_tag=active_tag) for t in tags[:12])
    genre = esc(row.get("genre_primary", "Unknown"))
    year = fmt_int(row.get("year"))
    pos = fmt_float(row.get("positivity"), 1, "%")
    recs = fmt_int(row.get("review_volume"))
    play = fmt_float(row.get("playtime_h"), 1, "h")
    score = fmt_float(row.get("display_score", 0), 1)
    developer = clean_game_text(row.get("developer", "")) or "Unknown developer"
    publisher = clean_game_text(row.get("publisher", "")) or "Unknown publisher"
    steam = steam_url(row)
    steam_button = f'<a class="detail-cta primary" href="{esc(steam)}" target="_blank" rel="noopener noreferrer">Open Steam page</a>' if steam else '<span class="detail-cta primary">Steam page unavailable</span>'
    first_tag = str(tags[0]) if tags else ""
    more_button = f'<a class="detail-cta secondary" href="{app_link("Explore", tag=first_tag, anchor="content-start")}" target="_top">Explore more {esc(first_tag)}</a>' if first_tag else f'<a class="detail-cta secondary" href="{app_link("Explore", anchor="content-start")}" target="_top">Back to library</a>'
    back_href = app_link("Explore", tag=active_tag if active_tag else None, anchor="content-start")

    mode_bits = []
    if bool(row.get("is_singleplayer", False)):
        mode_bits.append("Singleplayer")
    if bool(row.get("is_multiplayer", False)):
        mode_bits.append("Multiplayer")
    if bool(row.get("is_coop", False)):
        mode_bits.append("Co-op")
    mode_text = ", ".join(mode_bits) if mode_bits else "Mode tidak tertulis eksplisit di metadata."
    core_tags = ", ".join(str(t) for t in tags[:5]) if tags else "Tag belum tersedia."
    why_text = explain_row(row, games, [], [])

    render_html(f"""
    <section class="detail-hero">
      <a class="back-link" href="{back_href}" target="_top">← Back to library</a>
      <div class="detail-grid">
        <div class="detail-cover">{cover_html}</div>
        <div class="detail-copy">
          <span class="detail-kicker">{genre} / {year}</span>
          <h1 class="detail-title">{title}</h1>
          <p class="detail-description">{desc}</p>
          <div class="pill-row">
            <span class="pill pill-blue">Quality {score}</span>
            <span class="pill pill-green">Pos {pos}</span>
            <span class="pill">Reviews {recs}</span>
            <span class="pill">Playtime {play}</span>
            {price_badge(row)}
          </div>
          <div>{tag_html}</div>
          <div class="detail-actions">{steam_button}{more_button}</div>
        </div>
      </div>
    </section>
    """)

    panels = "".join([
        detail_panel_html("Gameplay identity", f"{genre} game dengan fokus metadata: {core_tags}. Mode: {mode_text}"),
        detail_panel_html("Studio signal", f"Developer: {developer}. Publisher: {publisher}."),
        detail_panel_html("Why this game", why_text),
    ])
    render_html(f'<div class="detail-panels">{panels}</div>')

    render_html(section_header("If you're interested in this", "similar games from tags, genre, description, and quality signal"))
    similar = similar_games_for(row, games, matrix, limit=6)
    render_cards(similar, games, columns=3, active_tag=active_tag)

def apply_global_filters(
    games: pd.DataFrame,
    year_range: tuple[int, int],
    max_price: float,
    min_pos: float,
    genres: Sequence[str],
    tags: Sequence[str],
    mode: str,
    search: str,
) -> pd.DataFrame:
    df = games.copy()
    if df["year"].notna().any():
        df = df[df["year"].fillna(0).between(year_range[0], year_range[1])]
    df = df[(df["price_effective"].fillna(np.inf) <= max_price) | df["is_free"]]
    df = df[df["positivity"].fillna(0) >= min_pos]
    if genres:
        genre_set = {g.lower() for g in genres}
        df = df[df["genre_list"].apply(lambda xs: bool({x.lower() for x in xs} & genre_set))]
    for tag in tags:
        df = df[df["tag_list"].apply(lambda xs, t=tag: any(x.lower() == t.lower() for x in xs))]
    if mode == "singleplayer":
        df = df[df["is_singleplayer"]]
    elif mode == "multiplayer":
        df = df[df["is_multiplayer"]]
    elif mode == "coop":
        df = df[df["is_coop"]]
    if search.strip():
        pat = re.escape(search.strip())
        df = df[df["name"].str.contains(pat, case=False, regex=True, na=False)]
    return df


def clean_plotly(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,19,52,0.16)",
        font=dict(color="#C7DCE2", family="Inter"),
        margin=dict(l=12, r=12, t=55, b=12),
        height=height,
    )
    return fig


def safe_top_tags(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for values in df.get("tag_list", []):
        if isinstance(values, list):
            counter.update(values)
    return pd.DataFrame(counter.most_common(n), columns=["tag", "count"])



def top_unique_games(df: pd.DataFrame, sort_col: str, used_names: set[str], n: int = 3) -> pd.DataFrame:
    """Pick top games while avoiding repeated titles across quick-pick panels."""
    if df.empty or sort_col not in df.columns:
        return df.head(0)
    ranked = df.sort_values(sort_col, ascending=False, na_position="last")
    fresh = ranked[~ranked["name"].astype(str).isin(used_names)].head(n)
    if len(fresh) < n:
        fallback = ranked[~ranked.index.isin(fresh.index)].head(n - len(fresh))
        fresh = pd.concat([fresh, fallback], axis=0)
    used_names.update(fresh["name"].astype(str).tolist())
    return fresh




def render_sidebar_brand() -> None:
    render_html(
        """
        <div class="brand-card">
            <div class="brand-mark">SV</div>
            <h2>SteamVault Pro</h2>
            <p>Premium discovery console untuk analisis, eksplorasi, dan rekomendasi game Steam.</p>
        </div>
        <div class="sidebar-note">
            Atur library filter di sini. Gunakan tab Rekomendasi untuk mencari game berdasarkan preferensi, genre, tag, budget, dan sinyal crowd.
        </div>
        """
    )


def hero_section(total_games: int, filtered_games: int, data_source: str) -> str:
    explore_href = app_link("Explore", anchor="content-start")
    recommend_href = app_link("Recommend", anchor="content-start")
    return f"""
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">SteamVault Pro / cinematic discovery engine</div>
          <h1><span class="ghost-word">Enter the</span> <span class="accent">Vault</span> of games.</h1>
          <p class="hero-subtitle">
            Platform eksplorasi game Steam dengan tampilan cinematic, filter library,
            dan rekomendasi hybrid yang menjelaskan alasan tiap game dipilih.
          </p>
          <div class="hero-proof-row">
            <span class="hero-proof">Hybrid recommender</span>
            <span class="hero-proof">Same-page tag filter</span>
            <span class="hero-proof">Game detail pages</span>
            <span class="hero-proof">AAA-style UI</span>
          </div>
          <div class="hero-actions">
            <a class="cta cta-primary" href="{recommend_href}" target="_top">Cari rekomendasi</a>
            <a class="cta cta-secondary" href="{explore_href}" target="_top">Browse library</a>
          </div>
          <div class="hero-action-note">
            <b>Rekomendasi</b> = isi preferensi lalu sistem memilih game yang paling cocok. \n
            <b>Library</b> = browse semua game, sorting, filter, dan klik tag tanpa buka tab baru.
          </div>
          <div class="hero-stats">
            <div class="hero-stat"><strong>{total_games:,}</strong><span>Total games indexed</span></div>
            <div class="hero-stat"><strong>{filtered_games:,}</strong><span>Live results after filters</span></div>
            <div class="hero-stat"><strong>{esc(data_source)}</strong><span>Active dataset</span></div>
          </div>
        </div>
        <div class="hero-panel">
          <div class="launcher-screen">
            <div class="signature-orb"></div>
            <span class="particle p1"></span><span class="particle p2"></span><span class="particle p3"></span><span class="particle p4"></span>
            <div class="mock-row one">
              <div class="mock-img"></div>
              <div class="mock-line"><b>Quality signal</b><span>Bayesian rating + popularity depth</span></div>
              <div class="mock-score">92</div>
            </div>
            <div class="mock-row two">
              <div class="mock-img"></div>
              <div class="mock-line"><b>Content match</b><span>TF-IDF genre, tag, description</span></div>
              <div class="mock-score">88</div>
            </div>
            <div class="mock-row three">
              <div class="mock-img"></div>
              <div class="mock-line"><b>Hybrid engine</b><span>Rule + crowd + value + novelty</span></div>
              <div class="mock-score">95</div>
            </div>
          </div>
        </div>
      </div>
    </section>
    """

def feature_strip() -> str:
    items = [
        ("Discover", "Browse game cards with cinematic covers, gameplay descriptions, badges, value signals, and detail pages."),
        ("Analyze", "Read market-level insight from genre, price, positivity, and quality distributions."),
        ("Recommend", "Run smart hybrid recommendation using content, rule, crowd, value, and novelty weights."),
        ("Explain", "Every recommendation includes a readable reason and component score breakdown."),
    ]
    cards = "".join(f'<div class="feature-card"><b>{esc(title)}</b><span>{esc(desc)}</span></div>' for title, desc in items)
    return f'<div class="feature-grid">{cards}</div>'


def section_header(title: str, subtitle: str = "") -> str:
    return f'<div class="section-title"><h3>{esc(title)}</h3><span>{esc(subtitle)}</span></div>'

def top_navigation(active_view: str, active_tag: str = "") -> str:
    items = [
        ("Overview", "Overview"),
        ("Explore Library", "Explore"),
        ("Recommender", "Recommend"),
        ("Evaluation", "Evaluation"),
        ("Methodology", "Methodology"),
    ]
    links: list[str] = []
    for label, view in items:
        active = " active" if active_view == view else ""
        tag = active_tag if view == "Explore" and active_tag else None
        links.append(
            f'<a class="top-nav-link{active}" href="{app_link(view, tag=tag, anchor="content-start")}" target="_top">{esc(label)}</a>'
        )
    if active_view == "Detail":
        tag_note = 'Game detail page'
    else:
        tag_note = f'Tag aktif: {esc(active_tag)}' if active_tag else 'Same-tab navigation'
    return (
        '<nav class="top-nav-shell" aria-label="Main navigation">'
        '<a class="top-nav-brand" href="' + app_link('Overview') + '" target="_top"><span>SV</span>SteamVault Pro</a>'
        f'<div class="top-nav-links">{"".join(links)}<div class="top-nav-meta">{tag_note}</div></div>'
        '</nav>'
    )

# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
inject_css()

render_sidebar_brand()
uploaded_games = st.sidebar.file_uploader("Upload CSV dataset game", type=["csv"], help="Opsional. Jika kosong, aplikasi membaca steam_top_games_2026.csv di folder yang sama.")
uploaded_interactions = st.sidebar.file_uploader(
    "Opsional: upload interaksi user-item",
    type=["csv"],
    help="Untuk true collaborative filtering. Kolom minimal: user_id dan app_id/name; rating/playtime/liked opsional.",
)

try:
    if uploaded_games is not None:
        games = load_games_from_bytes(uploaded_games.getvalue())
        data_source = uploaded_games.name
    elif DEFAULT_CSV.exists():
        games = load_games_from_path(str(DEFAULT_CSV))
        data_source = DEFAULT_CSV.name
    else:
        st.error("CSV belum ditemukan. Upload dataset melalui sidebar atau letakkan steam_top_games_2026.csv di folder app.")
        st.stop()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    st.stop()

interactions = None
if uploaded_interactions is not None:
    try:
        interactions = load_interactions_from_bytes(uploaded_interactions.getvalue())
    except Exception as exc:
        st.sidebar.warning(f"Interaksi gagal dibaca: {exc}")

vectorizer, tfidf_matrix = build_tfidf(tuple(games["content_text"].tolist()))
all_titles = sorted(games["name"].dropna().astype(str).unique().tolist())
all_genres = sorted([g for g in games["genre_primary"].dropna().unique().tolist() if g and g != "Unknown"])
all_tags = top_values_from_lists(games, "tag_list", limit=120)

NAV_OPTIONS = ["Overview", "Explore", "Recommend", "Evaluation", "Methodology", "Detail"]
active_view = match_known_value(query_value("view", "Overview"), NAV_OPTIONS)
if active_view not in NAV_OPTIONS:
    active_view = "Overview"
active_tag = match_known_value(query_value("tag", ""), all_tags)
active_tag_default = [active_tag] if active_tag in all_tags else []

# Sidebar global filters
st.sidebar.markdown("---")
st.sidebar.markdown("### Filter global")
years = games["year"].dropna()
if years.empty:
    min_year, max_year = 1990, 2030
else:
    min_year, max_year = int(years.min()), int(years.max())
year_range = st.sidebar.slider("Tahun rilis", min_year, max_year, (min_year, max_year))
price_limit_global = float(np.nanquantile(games["price_effective"].fillna(0), 0.98)) if len(games) else 100.0
price_limit_global = max(10.0, min(200.0, price_limit_global))
global_price = st.sidebar.slider("Harga maksimum global ($)", 0.0, float(math.ceil(price_limit_global)), min(60.0, float(math.ceil(price_limit_global))), 1.0)
global_min_pos = st.sidebar.slider("Minimal positivity global (%)", 0, 100, 0)
global_genres = st.sidebar.multiselect("Genre global", all_genres, max_selections=5)
global_tags = st.sidebar.multiselect(
    "Tag wajib global",
    all_tags,
    default=active_tag_default,
    max_selections=5,
    key=f"global_tags_{active_tag or 'all'}",
)
global_mode = st.sidebar.selectbox("Mode global", ["any", "singleplayer", "multiplayer", "coop"])
global_search = st.sidebar.text_input("Cari judul")
filtered = apply_global_filters(games, year_range, global_price, global_min_pos, global_genres, global_tags, global_mode, global_search)

render_html(top_navigation(active_view, active_tag))
nav_view = active_view

if nav_view == "Detail":
    render_html('<span id="content-start"></span>')
    detail_row = find_game_by_key(games, query_value("game", ""))
    if detail_row is None:
        st.warning("Game detail tidak ditemukan. Kembali ke Library lalu pilih game lagi.")
        st.stop()
    render_game_detail(detail_row, games, tfidf_matrix, active_tag=active_tag)
    st.stop()

render_html(hero_section(len(games), len(filtered), data_source))
render_html(
    """
    <div id="content-start" class="nav-intro">
      <span>Navigation ready</span>
      <b>Poster dan judul membuka halaman detail di tab yang sama. Tombol Steam tersedia di halaman detail.</b>
    </div>
    """
)

if active_tag:
    render_html(
        f"""
        <div class="active-filter-card">
          <span>Tag mode aktif: <b>{esc(active_tag)}</b>. Library sekarang menampilkan game dengan tag ini.</span>
          <a href="{app_link('Explore', anchor='content-start')}" target="_top">Clear tag</a>
        </div>
        """
    )

st.caption(f"Data source: {data_source} | Jumlah data: {len(games):,} game | Setelah filter: {len(filtered):,} game")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Library size", f"{len(games):,}")
kpi2.metric("Active results", f"{len(filtered):,}")
kpi3.metric("Free titles", f"{int(filtered['is_free'].sum()):,}" if not filtered.empty else "0")
kpi4.metric("Avg positivity", fmt_float(filtered["positivity"].mean() if not filtered.empty else np.nan, 1, "%"))
kpi5.metric("Quality index", fmt_float((filtered["quality_score"].mean() * 100) if not filtered.empty else np.nan, 1))

if nav_view == "Overview":
    render_html(feature_strip())
    render_html(section_header("Library intelligence", "overview dataset"))
    if filtered.empty:
        st.warning("Tidak ada data pada filter global saat ini.")
    else:
        c1, c2 = st.columns([1.12, 0.88])
        with c1:
            genre_count = filtered.groupby("genre_primary", as_index=False).size().sort_values("size", ascending=False).head(14)
            fig = px.bar(genre_count, x="size", y="genre_primary", orientation="h", title="Top genre berdasarkan jumlah game", labels={"size": "Jumlah", "genre_primary": "Genre"})
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(clean_plotly(fig, height=430), width="stretch")
        with c2:
            top_tags = safe_top_tags(filtered, 14)
            if not top_tags.empty:
                fig = px.bar(top_tags, x="count", y="tag", orientation="h", title="Top tag paling sering muncul", labels={"count": "Jumlah", "tag": "Tag"})
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(clean_plotly(fig, height=430), width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            price_df = filtered[filtered["price_effective"].notna()].copy()
            fig = px.histogram(price_df, x="price_effective", nbins=30, title="Distribusi harga", labels={"price_effective": "Harga efektif ($)", "count": "Jumlah"})
            st.plotly_chart(clean_plotly(fig, height=340), width="stretch")
        with c4:
            scatter = filtered.copy()
            scatter["review_volume_log"] = np.log10(scatter["review_volume"].fillna(0) + 1)
            fig = px.scatter(
                scatter,
                x="positivity",
                y="display_score",
                size="review_volume_log",
                color="genre_primary",
                hover_name="name",
                title="Positivity vs quality score",
                labels={"positivity": "Positivity (%)", "display_score": "Quality score", "review_volume_log": "Log reviews", "genre_primary": "Genre"},
            )
            st.plotly_chart(clean_plotly(fig, height=340), width="stretch")

        render_html(section_header("Fast picks", "quality, value, and crowd favorites"))
        pick_cols = st.columns(3)
        used_quick_names: set[str] = set()
        quick_sets = [
            ("Best Quality", top_unique_games(filtered, "quality_score", used_quick_names, 3)),
            ("Best Value", top_unique_games(filtered, "value_score", used_quick_names, 3)),
            ("Crowd Favorite", top_unique_games(filtered, "crowd_score", used_quick_names, 3)),
        ]
        for col, (label, data) in zip(pick_cols, quick_sets):
            with col:
                render_html(f"<div class='glass-panel'><b>{esc(label)}</b></div>")
                render_cards(data, games, columns=1, active_tag=active_tag)

elif nav_view == "Explore":
    render_html('<span id="explore"></span>' + section_header("Game explorer", "browse, filter, and discover similar games"))
    render_html("<div class='mini-note'>Klik poster atau judul game untuk masuk ke halaman detail di tab yang sama. Tombol Steam tersedia di halaman detail game.</div>")
    if filtered.empty:
        st.warning("Tidak ada data pada filter global saat ini.")
    else:
        e1, e2, e3 = st.columns([1.5, 1, 1])
        sort_col = e1.selectbox(
            "Urutkan berdasarkan",
            ["quality_score", "value_score", "crowd_score", "display_score", "positivity", "review_volume", "year", "price_effective", "metacritic_score"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        sort_asc = e2.toggle("Ascending", value=False)
        n_show = e3.slider("Jumlah kartu", 6, 60, 18, 3)
        browse = filtered.sort_values(sort_col, ascending=sort_asc, na_position="last").head(n_show)
        render_cards(browse, games, columns=3, active_tag=active_tag)
        st.markdown("### Tabel data")
        display_cols = [
            "name",
            "genre_primary",
            "year",
            "price_effective",
            "is_free",
            "positivity",
            "review_volume",
            "display_score",
            "metacritic_score",
            "playtime_h",
            "developer",
            "publisher",
        ]
        st.dataframe(
            filtered[display_cols].rename(columns={"display_score": "quality_score", "price_effective": "price_usd"}),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download hasil filter CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="steamvault_filtered_games.csv",
            mime="text/csv",
        )

elif nav_view == "Recommend":
    render_html('<span id="recommender"></span>' + section_header("Smart recommender", "hybrid, explainable, configurable"))
    render_html(
        "<div class='mini-note'>Tips: pilih 1-5 game favorit atau beberapa tag/genre. Jika tidak ada input favorit, sistem otomatis menjadi cold-start recommender berbasis rule, value, dan crowd signal.</div>"
    )

    MOODS = {
        "Tanpa preset": [],
        "Story rich & singleplayer": ["Story Rich", "Singleplayer", "RPG", "Adventure", "Atmospheric"],
        "Competitive multiplayer": ["Multiplayer", "PvP", "Competitive", "Shooter", "eSports"],
        "Cozy casual": ["Casual", "Relaxing", "Cozy", "Cute", "Family Friendly"],
        "Strategy deep dive": ["Strategy", "Simulation", "Turn-Based", "Management", "Tactical"],
        "Budget friendly": ["Free to Play", "Indie", "Casual", "Co-op"],
    }

    r1, r2 = st.columns([1.05, 0.95])
    with r1:
        engine = st.selectbox(
            "Engine rekomendasi",
            ["Smart Hybrid", "Content-Based", "Rule-Based", "Collaborative / Crowd"],
            help="Smart Hybrid memakai weighted hybrid. Collaborative akan menjadi true user-item CF jika file interaksi diupload; jika tidak, memakai crowd wisdom proxy.",
        )
        favorite_titles = st.multiselect("Game favorit / referensi", all_titles, max_selections=5)
        preferred_genres = st.multiselect("Genre preferensi", all_genres, max_selections=5)
        preferred_tags = st.multiselect("Tag preferensi", all_tags, max_selections=10)
        mood_name = st.selectbox("Mood preset", list(MOODS.keys()))
        mood_terms = MOODS[mood_name]
    with r2:
        max_price = st.slider("Maksimum harga rekomendasi ($)", 0.0, float(math.ceil(price_limit_global)), min(45.0, float(math.ceil(price_limit_global))), 1.0)
        min_pos = st.slider("Minimal positivity rekomendasi (%)", 0, 100, 65)
        min_reviews = st.slider("Minimal review/recommendation", 0, 100000, 250, 250)
        mode = st.selectbox("Mode bermain", ["any", "singleplayer", "multiplayer", "coop"])
        must_have_tags = st.multiselect("Tag wajib", all_tags, max_selections=4)
        top_n = st.slider("Jumlah rekomendasi", 5, 30, 12)
        diversity = st.slider("Diversity penalty", 0.0, 0.60, 0.18, 0.02, help="Lebih tinggi = hasil lebih beragam, mengurangi game yang terlalu mirip satu sama lain.")

    weights = {"content": 0.42, "crowd": 0.27, "rule": 0.16, "value": 0.10, "novelty": 0.05}
    if engine == "Smart Hybrid":
        with st.expander("Atur bobot hybrid", expanded=False):
            w1, w2, w3, w4, w5 = st.columns(5)
            weights["content"] = w1.slider("Content", 0.0, 1.0, weights["content"], 0.05)
            weights["crowd"] = w2.slider("Crowd/CF", 0.0, 1.0, weights["crowd"], 0.05)
            weights["rule"] = w3.slider("Rule", 0.0, 1.0, weights["rule"], 0.05)
            weights["value"] = w4.slider("Value", 0.0, 1.0, weights["value"], 0.05)
            weights["novelty"] = w5.slider("Novelty", 0.0, 1.0, weights["novelty"], 0.05)

    recs = recommend_games(
        games=games,
        matrix=tfidf_matrix,
        vectorizer=vectorizer,
        engine=engine,
        favorite_titles=favorite_titles,
        preferred_genres=preferred_genres,
        preferred_tags=preferred_tags,
        must_have_tags=must_have_tags,
        mood_terms=mood_terms,
        max_price=max_price,
        min_positivity=float(min_pos),
        min_reviews=int(min_reviews),
        mode=mode,
        top_n=int(top_n),
        diversity=float(diversity),
        weights=weights,
        interactions=interactions,
    )

    if recs.empty:
        st.warning("Tidak ada rekomendasi yang cocok. Turunkan minimal positivity, review, harga, atau tag wajib.")
    else:
        source_label = recs["cf_source"].iloc[0] if "cf_source" in recs.columns else "Crowd proxy"
        render_html(
            f"<div class='mini-note'><b>Engine aktif:</b> {esc(engine)} | <b>Sinyal kolaboratif:</b> {esc(source_label)} | Hasil sudah direrank dengan diversity penalty.</div>"
        )
        render_cards(recs, games, favorite_titles, preferred_tags, columns=3, show_components=True, active_tag=active_tag)

        st.markdown("### Score breakdown")
        chart_df = recs.head(10)[["name", "content_component", "crowd_component", "rule_component", "value_component", "novelty_component", "final_score"]].copy()
        chart_long = chart_df.melt(id_vars="name", var_name="component", value_name="score")
        fig = px.bar(chart_long, x="score", y="name", color="component", orientation="h", barmode="group", title="Komponen skor top recommendation", labels={"score": "Skor 0-1", "name": "Game"})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(clean_plotly(fig, height=470), width="stretch")

        export_cols = [
            "name",
            "genre_primary",
            "year",
            "price_effective",
            "positivity",
            "review_volume",
            "final_score_pct",
            "content_component",
            "crowd_component",
            "rule_component",
            "value_component",
            "novelty_component",
            "developer",
            "publisher",
            "short_description",
        ]
        st.download_button(
            "Download rekomendasi CSV",
            recs[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="steamvault_recommendations.csv",
            mime="text/csv",
        )

elif nav_view == "Evaluation":
    render_html(section_header("Recommendation evaluation", "quality, diversity, coverage"))
    st.write("Tab ini mengevaluasi hasil rekomendasi terakhir dari konfigurasi pada tab Rekomendasi.")
    try:
        eval_recs = recs.copy()
    except NameError:
        eval_recs = pd.DataFrame()

    if eval_recs.empty:
        st.info("Buat rekomendasi terlebih dahulu di tab Rekomendasi.")
    else:
        # Intra-list diversity from TF-IDF item vectors.
        idxs = eval_recs.index.to_list()
        if len(idxs) > 1:
            sim = cosine_similarity(tfidf_matrix[idxs], tfidf_matrix[idxs])
            tri = sim[np.triu_indices_from(sim, k=1)]
            diversity_metric = 1 - float(np.mean(tri))
        else:
            diversity_metric = np.nan
        genre_coverage = eval_recs["genre_primary"].nunique()
        tag_counter = Counter()
        for xs in eval_recs["tag_list"]:
            if isinstance(xs, list):
                tag_counter.update(xs)
        tag_coverage = len(tag_counter)
        avg_price = eval_recs["price_effective"].mean()
        avg_pos = eval_recs["positivity"].mean()
        avg_final = eval_recs["final_score_pct"].mean()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Avg final score", fmt_float(avg_final, 1))
        m2.metric("Avg positivity", fmt_float(avg_pos, 1, "%"))
        m3.metric("Genre coverage", f"{genre_coverage}")
        m4.metric("Tag coverage", f"{tag_coverage}")
        m5.metric("Intra-list diversity", fmt_float(diversity_metric, 2))

        e1, e2 = st.columns(2)
        with e1:
            genre_eval = eval_recs.groupby("genre_primary", as_index=False).size().sort_values("size", ascending=False)
            fig = px.pie(genre_eval, values="size", names="genre_primary", title="Sebaran genre pada hasil rekomendasi")
            st.plotly_chart(clean_plotly(fig, height=380), width="stretch")
        with e2:
            top_tag_eval = pd.DataFrame(tag_counter.most_common(12), columns=["tag", "count"])
            if not top_tag_eval.empty:
                fig = px.bar(top_tag_eval, x="count", y="tag", orientation="h", title="Top tag pada hasil rekomendasi")
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(clean_plotly(fig, height=380), width="stretch")

        st.markdown("### Interpretasi evaluasi")
        st.markdown(
            f"""
            - **Final score rata-rata** menunjukkan kekuatan rekomendasi berdasarkan engine yang dipilih.
            - **Intra-list diversity** mendekati 1 berarti rekomendasi lebih bervariasi; mendekati 0 berarti hasil sangat mirip satu sama lain.
            - **Genre/tag coverage** membantu melihat apakah sistem terlalu sempit atau sudah cukup beragam.
            - **Harga rata-rata** saat ini sekitar **${avg_price:.2f}**, sehingga bisa dipakai untuk membahas aspek value-for-money.
            """
        )

elif nav_view == "Methodology":
    render_html(section_header("Recommendation methodology", "ready for dashboard explanation"))
    st.markdown(
        """
        Dashboard ini memakai empat pendekatan utama agar sesuai dengan topik recommendation system.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        render_html(
            """
            <div class='method-card'>
            <h4>1. Rule-Based Recommendation</h4>
            <p>Rekomendasi dipilih menggunakan aturan eksplisit seperti genre, harga, minimal positivity, minimal review, mode bermain, dan tag wajib.</p>
            <p><b>Kelebihan:</b> mudah dijelaskan dan cocok untuk cold-start user.</p>
            </div>
            """
        )
        render_html(
            """
            <div class='method-card'>
            <h4>2. Content-Based Recommendation</h4>
            <p>Item profile dibangun dari genre, tag, kategori, developer, publisher, dan deskripsi singkat. Teks dikonversi menjadi TF-IDF, lalu dihitung kemiripannya dengan cosine similarity.</p>
            <p><b>Formula:</b> similarity(user, item) = cosine(TF-IDF user profile, TF-IDF item profile).</p>
            </div>
            """
        )
    with c2:
        render_html(
            """
            <div class='method-card'>
            <h4>3. Collaborative / Crowd Signal</h4>
            <p>Jika file interaksi user-item diupload, sistem memakai item-based collaborative filtering. Jika tidak, dashboard memakai proxy crowd wisdom dari Bayesian rating, volume review, dan popularity.</p>
            <p><b>Catatan ilmiah:</b> proxy crowd wisdom bukan pure CF, tetapi aman untuk dataset agregat yang tidak punya user_id.</p>
            </div>
            """
        )
        render_html(
            """
            <div class='method-card'>
            <h4>4. Weighted Hybrid Recommendation</h4>
            <p>Skor akhir menggabungkan content match, crowd/collaborative signal, rule fit, value, dan novelty.</p>
            <p><b>Formula:</b> S = w1*C_content + w2*C_crowd + w3*C_rule + w4*C_value + w5*C_novelty.</p>
            </div>
            """
        )

    st.markdown("### Rumus penting")
    st.latex(r"WR = \frac{v}{v+m}R + \frac{m}{v+m}C")
    st.markdown(
        """
        Keterangan: `R` adalah positivity item, `v` adalah jumlah review/recommendation, `C` adalah rata-rata positivity seluruh item, dan `m` adalah ambang minimum berbasis kuantil. Rumus ini membuat game dengan review sedikit tidak langsung menang hanya karena positivity tinggi.
        """
    )
    st.markdown("### Keterbatasan")
    st.markdown(
        """
        - Dataset Steam top games biasanya bersifat agregat, sehingga tidak selalu memiliki matriks `user_id x item`. Karena itu, true collaborative filtering hanya aktif jika file interaksi user-item ditambahkan.
        - Content-based recommendation sangat bergantung pada kualitas metadata seperti tag, genre, dan deskripsi.
        - Hybrid recommendation lebih robust, tetapi bobotnya perlu divalidasi dengan data interaksi nyata atau A/B testing jika digunakan di lingkungan produksi.
        """
    )
