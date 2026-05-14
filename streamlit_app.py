# SteamVault Pro - Steam Game Discovery & Hybrid Recommendation Dashboard
# Put this file in the same folder as steam_top_games_2026.csv, or upload the CSV from the sidebar.

from __future__ import annotations

import html
import io
import math
import re
import textwrap
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote
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
    """Render custom HTML/CSS safely as HTML, not as visible code text."""
    cleaned = textwrap.dedent(str(markup)).strip()
    if not cleaned:
        return
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
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 16px;
        }
        .hero-stat {
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


        /* Sidebar navigation replaces the too-high top navigation. */
        .sidebar-nav-hint {
            margin: 2px 0 10px;
            padding: 10px 12px;
            border-radius: 16px;
            color: var(--muted) !important;
            background: rgba(165,197,204,.06);
            border: 1px solid rgba(165,197,204,.12);
            font-size: .80rem;
            line-height: 1.45;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            width: 100% !important;
            border-radius: 16px !important;
            margin-bottom: 6px !important;
            padding: 10px 13px !important;
            border: 1px solid rgba(165,197,204,.13) !important;
            background: rgba(1,42,97,.20) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: rgba(253,199,135,.30) !important;
            background: rgba(253,199,135,.08) !important;
        }
        .content-anchor { height: 1px; width: 1px; overflow: hidden; }

        /* More editorial/media-card layout inspired by compact discovery apps. */
        .game-card { min-height: 100%; }
        .game-body { padding: 16px 18px 18px; }
        .game-title {
            font-size: clamp(1.06rem, 1.15vw, 1.34rem) !important;
            letter-spacing: -.035em;
            margin-bottom: 7px !important;
        }
        .game-title a { text-decoration: none !important; }
        .game-title a:hover { color: var(--gold) !important; }
        .game-desc {
            margin: 10px 0 12px;
            color: rgba(238,248,250,.78) !important;
            font-size: .88rem;
            line-height: 1.55;
            min-height: 4.05em;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .why {
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 16px;
            background: rgba(253,199,135,.07) !important;
            border: 1px solid rgba(253,199,135,.14) !important;
            color: rgba(238,248,250,.82) !important;
            font-size: .78rem;
            line-height: 1.45;
        }
        .card-actions {
            display: flex;
            gap: 9px;
            flex-wrap: wrap;
            margin-top: 13px;
            position: relative;
            z-index: 3;
        }
        .card-action {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 38px;
            padding: 0 13px;
            border-radius: 999px;
            text-decoration: none !important;
            font-size: .76rem;
            font-weight: 950;
            color: var(--ink) !important;
            background: linear-gradient(135deg, var(--gold), var(--mist));
            border: 1px solid rgba(253,199,135,.38);
            box-shadow: 0 12px 28px rgba(0,0,0,.24), 0 0 22px rgba(253,199,135,.12);
            transition: transform .16s ease, filter .16s ease;
        }
        .card-action:hover { transform: translateY(-2px); filter: brightness(1.06); }
        .card-action.secondary {
            color: var(--text) !important;
            background: rgba(165,197,204,.08);
            border-color: rgba(165,197,204,.18);
            box-shadow: none;
        }
        .card-action.ghost {
            color: var(--mist) !important;
            background: transparent;
            border-color: rgba(165,197,204,.18);
            box-shadow: none;
        }
        .preview-chip { color: var(--ink) !important; }

        .detail-hero {
            position: relative;
            overflow: hidden;
            border-radius: 34px;
            margin: 8px 0 24px;
            min-height: 430px;
            border: 1px solid rgba(253,199,135,.24);
            box-shadow: 0 36px 110px rgba(0,0,0,.44), inset 0 1px 0 rgba(255,255,255,.08);
            background: linear-gradient(135deg, rgba(1,42,97,.46), rgba(2,19,52,.92));
        }
        .detail-backdrop {
            position: absolute;
            inset: 0;
            background-size: cover;
            background-position: center;
            opacity: .42;
            filter: saturate(1.2) contrast(1.12);
            transform: scale(1.03);
        }
        .detail-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 18% 22%, rgba(253,199,135,.23), transparent 17rem),
                linear-gradient(90deg, rgba(2,19,52,.95), rgba(2,19,52,.74) 46%, rgba(2,19,52,.30)),
                linear-gradient(0deg, rgba(2,8,23,.96), transparent 48%);
            z-index: 1;
        }
        .detail-content {
            position: relative;
            z-index: 2;
            max-width: 860px;
            padding: clamp(26px, 4vw, 54px);
        }
        .detail-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 32px;
            padding: 0 12px;
            border-radius: 999px;
            font-weight: 950;
            font-size: .76rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            color: var(--gold) !important;
            border: 1px solid rgba(253,199,135,.26);
            background: rgba(253,199,135,.08);
        }
        .detail-title {
            margin: 18px 0 12px;
            font-size: clamp(2.5rem, 7vw, 5.7rem);
            line-height: .90;
            letter-spacing: -.07em;
            color: #fff !important;
            text-shadow: 0 24px 66px rgba(0,0,0,.56);
        }
        .detail-desc {
            max-width: 790px;
            color: rgba(238,248,250,.84) !important;
            font-size: clamp(1rem, 1.3vw, 1.18rem);
            line-height: 1.75;
            margin: 0 0 22px;
        }
        .detail-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 18px;
        }
        .detail-action {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 50px;
            padding: 0 20px;
            border-radius: 999px;
            font-weight: 950;
            text-decoration: none !important;
            color: var(--ink) !important;
            background: linear-gradient(135deg, var(--gold), var(--mist));
            box-shadow: 0 22px 52px rgba(253,199,135,.20);
        }
        .detail-action.secondary {
            color: var(--text) !important;
            background: rgba(165,197,204,.08);
            border: 1px solid rgba(165,197,204,.20);
            box-shadow: none;
        }
        .detail-stat {
            padding: 17px 16px;
            border-radius: 22px;
            border: 1px solid rgba(165,197,204,.14);
            background: linear-gradient(180deg, rgba(1,42,97,.30), rgba(2,19,52,.68));
            box-shadow: 0 20px 56px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .detail-stat b {
            display: block;
            color: var(--text) !important;
            font-size: 1.35rem;
            letter-spacing: -.03em;
        }
        .detail-stat span {
            display: block;
            margin-top: 4px;
            color: var(--muted) !important;
            font-size: .80rem;
        }
        .detail-section {
            margin: 20px 0 26px;
            padding: clamp(18px, 2vw, 24px);
            border-radius: 28px;
            border: 1px solid rgba(165,197,204,.14);
            background: linear-gradient(180deg, rgba(1,42,97,.24), rgba(2,19,52,.58));
            box-shadow: 0 24px 78px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .detail-section h3 { margin-top: 0; }
        .detail-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        @media (max-width: 900px) {
            .detail-hero { min-height: auto; }
            .detail-content { padding: 28px 20px; }
        }
        @media (max-width: 580px) {
            .card-actions, .detail-actions { flex-direction: column; }
            .card-action, .detail-action { width: 100%; }
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


def app_link(view: str = "Library", tag: str | None = None, anchor: str | None = None, game: str | None = None) -> str:
    """Create an in-app same-tab navigation link using query params."""
    view_map = {
        "Overview": "Discover",
        "Explore": "Library",
        "Recommend": "Recommender",
        "Evaluation": "Analytics",
    }
    normalized_view = view_map.get(str(view), str(view))
    params = [f"view={quote(normalized_view, safe='')}"]
    if tag:
        params.append(f"tag={quote(str(tag), safe='')}")
    if game:
        params.append(f"game={quote(str(game), safe='')}")
    suffix = f"#{anchor}" if anchor else ""
    return "?" + "&".join(params) + suffix


def game_key(row: pd.Series) -> str:
    try:
        app_id = int(float(row.get("app_id", np.nan)))
        if app_id > 0:
            return str(app_id)
    except Exception:
        pass
    return re.sub(r"[^a-z0-9]+", "-", str(row.get("name", "game")).lower()).strip("-") or "game"


def game_detail_link(row: pd.Series) -> str:
    return app_link("Detail", game=game_key(row), anchor="content-start")


def find_game_from_param(games: pd.DataFrame, raw: str) -> pd.Series | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if "app_id" in games.columns:
        app_ids = games["app_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        matches = games[app_ids == value]
        if not matches.empty:
            return matches.iloc[0]
    slugged = games["name"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True).str.strip("-")
    matches = games[slugged == value.lower()]
    if not matches.empty:
        return matches.iloc[0]
    name_matches = games[games["name"].astype(str).str.lower() == value.lower()]
    if not name_matches.empty:
        return name_matches.iloc[0]
    return None


def tag_link(tag: str, active_tag: str = "") -> str:
    safe = esc(tag)
    active = " tag-active" if active_tag and active_tag.lower() == str(tag).lower() else ""
    href = app_link("Library", tag, "content-start")
    return f'<a class="tag{active}" href="{href}" target="_self" title="Show more {safe} games in this page">{safe}</a>'


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



def plain_description(row: pd.Series, max_chars: int = 190) -> str:
    """Clean and shorten game descriptions for card/detail copy."""
    raw = str(row.get("short_description", "") or "").strip()
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
        genre = str(row.get("genre_primary", "game"))
        if tags:
            raw = f"A {genre} title with {', '.join(tags[:4])} elements, recommended through quality, crowd, value, and content signals."
        else:
            raw = f"A {genre} title discovered from Steam metadata, scored with hybrid recommendation signals."
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"


def similar_games_for(row: pd.Series, games: pd.DataFrame, matrix, n: int = 6) -> pd.DataFrame:
    """Find related games using TF-IDF content similarity, with quality as a tie breaker."""
    if games.empty:
        return games.head(0)
    try:
        idx = int(row.name)
        sims = cosine_similarity(matrix[idx], matrix).ravel()
        out = games.copy()
        out["similarity_score"] = sims
        out = out.drop(index=idx, errors="ignore")
    except Exception:
        out = games.copy()
        tags = {str(t).lower() for t in row.get("tag_list", []) if str(t).strip()} if isinstance(row.get("tag_list", []), list) else set()
        genre = str(row.get("genre_primary", "")).lower()
        out["similarity_score"] = out.apply(
            lambda r: (0.55 if str(r.get("genre_primary", "")).lower() == genre else 0.0)
            + 0.45 * (len(tags & {str(t).lower() for t in r.get("tag_list", [])}) / max(1, len(tags)))
            if isinstance(r.get("tag_list", []), list)
            else 0.0,
            axis=1,
        )
        out = out[out["name"].astype(str) != str(row.get("name", ""))]
    out["related_rank_score"] = 0.70 * out["similarity_score"].fillna(0) + 0.30 * out["quality_score"].fillna(0)
    return out.sort_values("related_rank_score", ascending=False).head(n)


def query_value(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def clean_description(value: object, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 1].rsplit(" ", 1)[0].strip() + "..."
    return text


def game_key(row: pd.Series) -> str:
    value = row.get("app_id", "")
    try:
        val = float(value)
        if np.isfinite(val) and val > 0:
            return str(int(val))
    except Exception:
        pass
    return "name:" + str(row.get("name", "Unknown Game"))


def detail_href(row: pd.Series) -> str:
    return f"?view=detail&game={quote(game_key(row), safe='')}"


def tag_href(tag: str) -> str:
    return f"?view=library&tag={quote(str(tag), safe='')}"


def view_href(view: str) -> str:
    return f"?view={quote(str(view), safe='')}"


def find_game(games: pd.DataFrame, key: str) -> pd.Series | None:
    raw = unquote(str(key or "")).strip()
    if not raw:
        return None
    if raw.startswith("name:"):
        target = raw[5:].lower()
        match = games[games["name"].astype(str).str.lower() == target]
        return match.iloc[0] if not match.empty else None
    try:
        app_id = int(float(raw))
        match = games[pd.to_numeric(games["app_id"], errors="coerce").fillna(-1).astype(int) == app_id]
        if not match.empty:
            return match.iloc[0]
    except Exception:
        pass
    match = games[games["name"].astype(str).str.lower() == raw.lower()]
    return match.iloc[0] if not match.empty else None


def gameplay_blurb(row: pd.Series) -> str:
    genre = row.get("genre_primary", "game") or "game"
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    cats = row.get("category_list", []) if isinstance(row.get("category_list", []), list) else []
    tag_text = ", ".join(tags[:4]) if tags else "Steam-style discovery"
    cat_text = ", ".join(cats[:3]) if cats else "player-focused features"
    return f"A {genre} experience built around {tag_text}. Expect {cat_text.lower()} with a recommendation profile shaped by quality, popularity, value, and content similarity."


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
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe"),
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
    if df.empty or sort_col not in df.columns:
        return df.head(0)
    ranked = df.sort_values(sort_col, ascending=False, na_position="last")
    fresh = ranked[~ranked["name"].astype(str).isin(used_names)].head(n)
    if len(fresh) < n:
        fallback = ranked[~ranked.index.isin(fresh.index)].head(n - len(fresh))
        fresh = pd.concat([fresh, fallback], axis=0)
    used_names.update(fresh["name"].astype(str).tolist())
    return fresh


def similar_games(row: pd.Series, games: pd.DataFrame, matrix, n: int = 6) -> pd.DataFrame:
    if games.empty:
        return games.head(0)
    try:
        idx = int(row.name)
        sims = cosine_similarity(matrix[idx], matrix).ravel()
    except Exception:
        title = str(row.get("name", "")).lower()
        matches = games.index[games["name"].astype(str).str.lower() == title].tolist()
        if not matches:
            return games.sort_values("quality_score", ascending=False).head(n)
        idx = matches[0]
        sims = cosine_similarity(matrix[idx], matrix).ravel()
    out = games.copy()
    out["similarity_component"] = sims
    out = out[out.index != idx]
    out["final_score"] = (0.68 * normalize_array(out["similarity_component"].to_numpy(), default=0.0) + 0.32 * out["quality_score"].to_numpy(dtype=float)).clip(0, 1)
    out["final_score_pct"] = (out["final_score"] * 100).round(1)
    return out.sort_values("final_score", ascending=False).head(n)


def inject_extra_css() -> None:
    st.markdown(
        """
        <style>
        .sidebar-brand {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(253, 199, 135, 0.22);
            border-radius: 24px;
            padding: 18px 16px;
            margin: 0 0 16px;
            background:
                radial-gradient(circle at top right, rgba(253,199,135,.16), transparent 9rem),
                linear-gradient(145deg, rgba(1,42,97,.42), rgba(2,19,52,.84));
            box-shadow: 0 22px 60px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.08);
        }
        .sidebar-brand .logo {
            width: 46px; height: 46px; display: grid; place-items: center;
            border-radius: 16px; margin-bottom: 12px; color: #021334 !important;
            font-weight: 950; letter-spacing: -0.08em;
            background: linear-gradient(135deg, #FDC787, #A5C5CC);
            box-shadow: 0 0 35px rgba(253,199,135,.28);
        }
        .sidebar-brand h2 { margin: 0; font-size: 1.2rem; line-height: 1.05; }
        .sidebar-brand p { margin: 7px 0 0; color: rgba(165,197,204,.74) !important; font-size: .82rem; line-height: 1.45; }
        .side-nav { display: grid; gap: 8px; margin: 10px 0 18px; }
        .side-tab {
            display: flex; align-items: center; justify-content: space-between; gap: 10px;
            padding: 12px 13px; border-radius: 16px; text-decoration: none !important;
            border: 1px solid rgba(165,197,204,.14);
            color: #C7DCE2 !important;
            background: rgba(165,197,204,.055);
            transition: transform .18s ease, border-color .18s ease, background .18s ease, box-shadow .18s ease;
        }
        .side-tab:hover { transform: translateX(4px); border-color: rgba(253,199,135,.34); background: rgba(253,199,135,.10); }
        .side-tab.active { color: #021334 !important; font-weight: 900; background: linear-gradient(135deg, #FDC787, #A5C5CC); box-shadow: 0 16px 42px rgba(253,199,135,.18); }
        .side-tab small { opacity: .78; font-weight: 700; }
        .hero2 {
            position: relative; overflow: hidden; border-radius: 34px; padding: clamp(28px, 5vw, 58px);
            border: 1px solid rgba(253,199,135,.28);
            background:
                linear-gradient(110deg, rgba(2,19,52,.94) 0%, rgba(1,42,97,.78) 54%, rgba(2,19,52,.90) 100%),
                radial-gradient(circle at 78% 28%, rgba(253,199,135,.24), transparent 18rem);
            box-shadow: 0 34px 110px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.10);
            margin: 0 0 20px;
        }
        .hero2::before {
            content: ""; position: absolute; inset: 0; pointer-events: none;
            background-image: linear-gradient(115deg, rgba(165,197,204,.08) 1px, transparent 1px);
            background-size: 44px 44px; mask-image: linear-gradient(to right, rgba(0,0,0,.6), rgba(0,0,0,.08));
        }
        .hero2 .kicker { position: relative; display: inline-flex; padding: 8px 12px; border-radius: 999px; background: rgba(253,199,135,.12); border: 1px solid rgba(253,199,135,.28); color: #FDC787 !important; font-size: .78rem; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; }
        .hero2 h1 { position: relative; max-width: 880px; margin: 18px 0 0; color: #fff !important; font-size: clamp(2.8rem, 7vw, 6.5rem); line-height: .86; letter-spacing: -.075em; text-shadow: 0 24px 70px rgba(0,0,0,.48); }
        .hero2 p { position: relative; max-width: 760px; margin: 18px 0 0; color: #C7DCE2 !important; font-size: clamp(1rem, 1.5vw, 1.18rem); line-height: 1.75; }
        .hero-actions { position: relative; display: flex; flex-wrap: wrap; gap: 14px; margin-top: 26px; }
        .cta2, .card-action, .steam-button, .back-link {
            display: inline-flex; align-items: center; justify-content: center; gap: 9px;
            border-radius: 999px; padding: 12px 18px; text-decoration: none !important;
            font-weight: 900; border: 1px solid rgba(165,197,204,.20); transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
        }
        .cta2.primary, .card-action.primary, .steam-button { color: #021334 !important; background: linear-gradient(135deg, #FDC787, #A5C5CC); box-shadow: 0 16px 48px rgba(253,199,135,.24); }
        .cta2.secondary, .card-action.secondary, .back-link { color: #EAF7FA !important; background: rgba(165,197,204,.08); border-color: rgba(165,197,204,.20); }
        .cta2:hover, .card-action:hover, .steam-button:hover, .back-link:hover { transform: translateY(-2px); }
        .section-title { margin: 18px 0 14px; display: flex; align-items: end; justify-content: space-between; gap: 16px; }
        .section-title h3 { margin: 0; font-size: clamp(1.4rem, 2.6vw, 2.25rem); color: #fff !important; }
        .section-title span { color: rgba(165,197,204,.72) !important; font-weight: 700; }
        .game-card {
            position: relative; overflow: hidden; min-height: 100%; border-radius: 28px !important;
            border: 1px solid rgba(165,197,204,.17) !important;
            background: linear-gradient(180deg, rgba(1,42,97,.48), rgba(2,19,52,.94)) !important;
            box-shadow: 0 28px 75px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.07) !important;
        }
        .game-card::after { content: ""; position: absolute; inset: -1px; pointer-events: none; opacity: 0; background: radial-gradient(circle at 50% 0%, rgba(253,199,135,.22), transparent 15rem); transition: opacity .22s ease; }
        .game-card:hover { transform: translateY(-6px) scale(1.01) !important; border-color: rgba(253,199,135,.42) !important; box-shadow: 0 35px 110px rgba(0,0,0,.52), 0 0 0 1px rgba(253,199,135,.08) !important; }
        .game-card:hover::after { opacity: 1; }
        .poster-link { display: block; text-decoration: none !important; }
        .game-img-wrap { position: relative; aspect-ratio: 16/7; overflow: hidden; background: linear-gradient(135deg, rgba(39,90,145,.35), rgba(253,199,135,.12)); }
        .game-img-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .32s ease, filter .32s ease; }
        .game-card:hover .game-img-wrap img { transform: scale(1.06); filter: saturate(1.12) contrast(1.04); }
        .game-img-wrap::after { content: ""; position: absolute; inset: 0; background: linear-gradient(to top, rgba(2,19,52,.65), transparent 52%); }
        .game-body { padding: 17px 17px 19px !important; position: relative; z-index: 2; }
        .game-title { margin: 0 0 8px; font-size: 1.12rem !important; line-height: 1.18; font-weight: 950 !important; letter-spacing: -.03em; }
        .game-title a { color: #fff !important; text-decoration: none !important; }
        .game-title a:hover { color: #FDC787 !important; }
        .meta-line { color: rgba(165,197,204,.78) !important; font-weight: 700; }
        .game-desc { margin: 11px 0 0; color: #D9EAEE !important; font-size: .86rem; line-height: 1.55; min-height: 4.05em; }
        .tag-link { text-decoration: none !important; cursor: pointer; }
        .tag-link:hover { background: rgba(253,199,135,.18) !important; border-color: rgba(253,199,135,.42) !important; color: #FDE1B7 !important; }
        .card-actions { display: flex; gap: 9px; flex-wrap: wrap; margin-top: 14px; }
        .card-action { padding: 9px 12px; font-size: .78rem; }
        .detail-hero {
            position: relative; overflow: hidden; border-radius: 34px; border: 1px solid rgba(253,199,135,.26);
            background: linear-gradient(135deg, rgba(2,19,52,.94), rgba(1,42,97,.70));
            box-shadow: 0 34px 110px rgba(0,0,0,.45); margin-bottom: 20px;
        }
        .detail-hero .cover { width: 100%; max-height: 430px; object-fit: cover; display: block; filter: saturate(1.1) contrast(1.04); }
        .detail-hero .overlay { position: absolute; inset: 0; background: linear-gradient(to top, rgba(2,19,52,.98) 3%, rgba(2,19,52,.68) 44%, rgba(2,19,52,.08)); }
        .detail-hero .content { position: absolute; left: clamp(22px, 5vw, 56px); right: clamp(22px, 5vw, 56px); bottom: clamp(20px, 5vw, 48px); }
        .detail-hero h1 { margin: 0; color: #fff !important; font-size: clamp(2rem, 5vw, 5rem); line-height: .92; }
        .detail-hero p { max-width: 900px; color: #D9EAEE !important; font-size: 1.02rem; line-height: 1.72; }
        .detail-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(300px, .8fr); gap: 18px; }
        .detail-panel, .mini-panel {
            border: 1px solid rgba(165,197,204,.17); border-radius: 26px; padding: 20px;
            background: linear-gradient(180deg, rgba(1,42,97,.38), rgba(2,19,52,.86));
            box-shadow: 0 22px 70px rgba(0,0,0,.28);
        }
        .detail-panel h3, .mini-panel h3 { margin-top: 0; color: #fff !important; }
        .stat-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
        .stat-box { padding: 14px; border-radius: 18px; background: rgba(165,197,204,.08); border: 1px solid rgba(165,197,204,.14); }
        .stat-box b { display: block; color: #fff !important; font-size: 1.25rem; }
        .stat-box span { color: rgba(165,197,204,.74) !important; font-size: .78rem; font-weight: 800; }
        @media (max-width: 900px) {
            .detail-grid { grid-template-columns: 1fr; }
            .detail-hero .content { position: relative; left: auto; right: auto; bottom: auto; padding: 22px; }
            .detail-hero .overlay { display: none; }
            .hero-actions { flex-direction: column; }
            .cta2 { width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="logo">SV</div>
          <h2>SteamVault Pro</h2>
          <p>Cinematic Steam discovery, analysis, and hybrid recommendation system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav(current_view: str) -> None:
    active_view = "library" if current_view == "detail" else current_view
    items = [
        ("home", "Discover", "start"),
        ("library", "Library", "browse"),
        ("recommend", "Recommender", "hybrid"),
        ("analytics", "Analytics", "insight"),
        ("methodology", "Methodology", "explain"),
    ]
    links = []
    for view, label, sub in items:
        cls = "side-tab active" if view == active_view else "side-tab"
        links.append(f'<a class="{cls}" href="{view_href(view)}"><span>{esc(label)}</span><small>{esc(sub)}</small></a>')
    st.sidebar.markdown('<div class="side-nav">' + ''.join(links) + '</div>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title"><h3>{esc(title)}</h3><span>{esc(subtitle)}</span></div>', unsafe_allow_html=True)


def game_card_html(
    row: pd.Series,
    games: pd.DataFrame,
    favorite_titles: Sequence[str] = (),
    preferred_tags: Sequence[str] = (),
    show_components: bool = False,
) -> str:
    title = esc(row.get("name", "Unknown Game"))
    detail_url = esc(detail_href(row))
    img = esc(str(row.get("header_image", "")).strip())
    img_html = f'<img src="{img}" alt="{title} cover" loading="lazy" onerror="this.style.display=\'none\'">' if img else ""
    genre = esc(row.get("genre_primary", "Unknown"))
    year = fmt_int(row.get("year"))
    score = fmt_float(row.get("final_score_pct", row.get("display_score", 0)), 1)
    pos = fmt_float(row.get("positivity"), 1, "%")
    recs = fmt_int(row.get("review_volume"))
    play = fmt_float(row.get("playtime_h"), 1, "h")
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    tag_html = "".join(f'<a class="tag tag-link" href="{esc(tag_href(t))}">{esc(t)}</a>' for t in tags[:7])
    desc = clean_description(row.get("short_description", ""), 185) or gameplay_blurb(row)
    why = esc(explain_row(row, games, favorite_titles, preferred_tags))
    comp_html = ""
    if show_components:
        comp_html = (
            component_bar("Content match", float(row.get("content_component", 0)))
            + component_bar("Crowd signal", float(row.get("crowd_component", 0)))
            + component_bar("Rule fit", float(row.get("rule_component", 0)))
            + component_bar("Value", float(row.get("value_component", 0)))
        )
    return f"""<article class="game-card"><a class="poster-link" href="{detail_url}"><div class="game-img-wrap">{img_html}</div></a><div class="game-body"><div class="game-title"><a href="{detail_url}">{title}</a></div><div class="meta-line">{genre} | {year} | {price_badge(row)}</div><div class="pill-row"><span class="pill pill-blue">Score {score}</span><span class="pill pill-green">Pos {pos}</span><span class="pill">Reviews {recs}</span><span class="pill">Playtime {play}</span></div><p class="game-desc">{esc(desc)}</p><div>{tag_html}</div>{comp_html}<div class="why"><b>Why:</b> {why}</div><div class="card-actions"><a class="card-action primary" href="{detail_url}">Open detail</a><a class="card-action secondary" href="{esc(view_href('recommend'))}">Recommend</a></div></div></article>"""


def render_cards(
    rows: pd.DataFrame,
    games: pd.DataFrame,
    favorite_titles: Sequence[str] = (),
    preferred_tags: Sequence[str] = (),
    columns: int = 3,
    show_components: bool = False,
) -> None:
    if rows.empty:
        st.info("Tidak ada data yang cocok dengan filter saat ini.")
        return
    columns = max(1, int(columns))
    cards = st.columns(columns)
    for i, (_, row) in enumerate(rows.iterrows()):
        with cards[i % columns]:
            st.markdown(game_card_html(row, games, favorite_titles, preferred_tags, show_components), unsafe_allow_html=True)


def home_hero(total_games: int, filtered_games: int, data_source: str) -> None:
    st.markdown(
        f"""
        <section class="hero2">
          <div class="kicker">Steam intelligence platform</div>
          <h1>Find the next game worth your night.</h1>
          <p>Explore a cinematic Steam library, open any game into a detail page, read its gameplay summary, then discover similar titles through a hybrid recommendation engine.</p>
          <div class="hero-actions">
            <a class="cta2 primary" href="{view_href('library')}">Browse library</a>
            <a class="cta2 secondary" href="{view_href('recommend')}">Build recommendation</a>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total games indexed", f"{total_games:,}")
    c2.metric("Live results after filters", f"{filtered_games:,}")
    c3.metric("Active dataset", data_source)


def render_home(games: pd.DataFrame, filtered: pd.DataFrame, data_source: str) -> None:
    home_hero(len(games), len(filtered), data_source)
    if filtered.empty:
        st.warning("Tidak ada game yang cocok dengan filter global saat ini.")
        return
    section_header("Tonight's featured picks", "click any title to open detail in this tab")
    used: set[str] = set()
    c1, c2, c3 = st.columns(3)
    quick_sets = [
        ("Best Quality", top_unique_games(filtered, "quality_score", used, 2)),
        ("Best Value", top_unique_games(filtered, "value_score", used, 2)),
        ("Crowd Favorite", top_unique_games(filtered, "crowd_score", used, 2)),
    ]
    for col, (label, data) in zip([c1, c2, c3], quick_sets):
        with col:
            st.markdown(f"<div class='glass-panel'><b>{esc(label)}</b></div>", unsafe_allow_html=True)
            render_cards(data, games, columns=1)


def render_library(games: pd.DataFrame, filtered: pd.DataFrame, selected_tag: str = "") -> None:
    section_header("Game library", "poster/title opens detail, tag chips filter this same tab")
    if selected_tag:
        st.markdown(f"<div class='mini-note'><b>Active tag:</b> {esc(selected_tag)}. Klik tag lain di kartu untuk ganti filter tanpa tab baru.</div>", unsafe_allow_html=True)
    if filtered.empty:
        st.warning("Tidak ada data pada filter global saat ini.")
        return
    e1, e2, e3 = st.columns([1.55, 0.9, 0.9])
    sort_col = e1.selectbox(
        "Urutkan berdasarkan",
        ["quality_score", "value_score", "crowd_score", "display_score", "positivity", "review_volume", "year", "price_effective", "metacritic_score"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    sort_asc = e2.toggle("Ascending", value=False)
    n_show = e3.slider("Jumlah kartu", 6, 60, 18, 3)
    browse = filtered.sort_values(sort_col, ascending=sort_asc, na_position="last").head(n_show)
    render_cards(browse, games, columns=3)
    with st.expander("Tampilkan tabel data dan download CSV", expanded=False):
        display_cols = [
            "name", "genre_primary", "year", "price_effective", "is_free", "positivity", "review_volume", "display_score", "metacritic_score", "playtime_h", "developer", "publisher", "short_description",
        ]
        st.dataframe(filtered[display_cols].rename(columns={"display_score": "quality_score", "price_effective": "price_usd"}), width="stretch", hide_index=True)
        st.download_button("Download hasil filter CSV", filtered.to_csv(index=False).encode("utf-8"), file_name="steamvault_filtered_games.csv", mime="text/csv")


def render_detail(game_key_value: str, games: pd.DataFrame, matrix) -> None:
    row = find_game(games, game_key_value)
    if row is None:
        st.warning("Game detail tidak ditemukan. Balik ke library dan pilih game lain.")
        st.markdown(f'<a class="back-link" href="{view_href("library")}">Back to library</a>', unsafe_allow_html=True)
        return
    title = esc(row.get("name", "Unknown Game"))
    img = esc(str(row.get("header_image", "")).strip())
    desc = clean_description(row.get("short_description", "")) or gameplay_blurb(row)
    steam = steam_url(row)
    steam_html = f'<a class="steam-button" href="{esc(steam)}" target="_blank" rel="noopener noreferrer">Open Steam page</a>' if steam else ""
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    tag_html = "".join(f'<a class="tag tag-link" href="{esc(tag_href(t))}">{esc(t)}</a>' for t in tags[:12])
    genre = esc(row.get("genre_primary", "Unknown"))
    hero_img = f'<img class="cover" src="{img}" alt="{title} cover">' if img else ""
    st.markdown(
        f"""<a class="back-link" href="{view_href('library')}">Back to library</a><div class="detail-hero">{hero_img}<div class="overlay"></div><div class="content"><div class="hero-actions">{steam_html}</div><h1>{title}</h1><p>{esc(desc)}</p><div>{tag_html}</div></div></div>""",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown(
            f"""
            <div class="detail-panel">
              <h3>Gameplay snapshot</h3>
              <p>{esc(gameplay_blurb(row))}</p>
              <p><b>Developer:</b> {esc(row.get('developer', '-'))}<br><b>Publisher:</b> {esc(row.get('publisher', '-'))}<br><b>Genre:</b> {genre}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="mini-panel">
              <h3>Signals</h3>
              <div class="stat-grid">
                <div class="stat-box"><b>{fmt_float(row.get('display_score'), 1)}</b><span>Quality index</span></div>
                <div class="stat-box"><b>{fmt_float(row.get('positivity'), 1, '%')}</b><span>Positive signal</span></div>
                <div class="stat-box"><b>{fmt_int(row.get('review_volume'))}</b><span>Reviews</span></div>
                <div class="stat-box"><b>{'Free' if bool(row.get('is_free', False)) else ('$' + fmt_float(row.get('price_effective'), 2))}</b><span>Price</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    section_header("If you're interested in this", "similar titles based on tags, genre, description, and quality")
    render_cards(similar_games(row, games, matrix, n=6), games, columns=3)


def render_analytics(filtered: pd.DataFrame) -> None:
    section_header("Library analytics", "market overview after global filters")
    if filtered.empty:
        st.warning("Tidak ada data pada filter global saat ini.")
        return
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
        fig = px.scatter(scatter, x="positivity", y="display_score", size="review_volume_log", color="genre_primary", hover_name="name", title="Positivity vs quality score", labels={"positivity": "Positivity (%)", "display_score": "Quality score", "review_volume_log": "Log reviews", "genre_primary": "Genre"})
        st.plotly_chart(clean_plotly(fig, height=340), width="stretch")


def render_recommender(games: pd.DataFrame, vectorizer: TfidfVectorizer, tfidf_matrix, interactions: pd.DataFrame | None, all_titles: list[str], all_genres: list[str], all_tags: list[str], price_limit_global: float) -> None:
    section_header("Smart recommender", "hybrid, explainable, configurable")
    st.markdown("<div class='mini-note'>Pilih game referensi, genre, atau tag. Hasilnya bisa langsung dibuka ke halaman detail tanpa pindah tab.</div>", unsafe_allow_html=True)
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
        engine = st.selectbox("Engine rekomendasi", ["Smart Hybrid", "Content-Based", "Rule-Based", "Collaborative / Crowd"])
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
        diversity = st.slider("Diversity penalty", 0.0, 0.60, 0.18, 0.02)
    weights = {"content": 0.42, "crowd": 0.27, "rule": 0.16, "value": 0.10, "novelty": 0.05}
    if engine == "Smart Hybrid":
        with st.expander("Atur bobot hybrid", expanded=False):
            w1, w2, w3, w4, w5 = st.columns(5)
            weights["content"] = w1.slider("Content", 0.0, 1.0, weights["content"], 0.05)
            weights["crowd"] = w2.slider("Crowd/CF", 0.0, 1.0, weights["crowd"], 0.05)
            weights["rule"] = w3.slider("Rule", 0.0, 1.0, weights["rule"], 0.05)
            weights["value"] = w4.slider("Value", 0.0, 1.0, weights["value"], 0.05)
            weights["novelty"] = w5.slider("Novelty", 0.0, 1.0, weights["novelty"], 0.05)
    recs = recommend_games(games, tfidf_matrix, vectorizer, engine, favorite_titles, preferred_genres, preferred_tags, must_have_tags, mood_terms, max_price, float(min_pos), int(min_reviews), mode, int(top_n), float(diversity), weights, interactions)
    if recs.empty:
        st.warning("Tidak ada rekomendasi yang cocok. Turunkan minimal positivity, review, harga, atau tag wajib.")
        return
    source_label = recs["cf_source"].iloc[0] if "cf_source" in recs.columns else "Crowd proxy"
    st.markdown(f"<div class='mini-note'><b>Engine aktif:</b> {esc(engine)} | <b>Sinyal kolaboratif:</b> {esc(source_label)} | Klik kartu untuk detail.</div>", unsafe_allow_html=True)
    render_cards(recs, games, favorite_titles, preferred_tags, columns=3, show_components=True)
    with st.expander("Score breakdown", expanded=False):
        chart_df = recs.head(10)[["name", "content_component", "crowd_component", "rule_component", "value_component", "novelty_component", "final_score"]].copy()
        chart_long = chart_df.melt(id_vars="name", var_name="component", value_name="score")
        fig = px.bar(chart_long, x="score", y="name", color="component", orientation="h", barmode="group", title="Komponen skor top recommendation", labels={"score": "Skor 0-1", "name": "Game"})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(clean_plotly(fig, height=470), width="stretch")
        export_cols = ["name", "genre_primary", "year", "price_effective", "positivity", "review_volume", "final_score_pct", "content_component", "crowd_component", "rule_component", "value_component", "novelty_component", "developer", "publisher", "short_description"]
        st.download_button("Download rekomendasi CSV", recs[export_cols].to_csv(index=False).encode("utf-8"), file_name="steamvault_recommendations.csv", mime="text/csv")


def render_methodology() -> None:
    section_header("Recommendation methodology", "siap dipakai untuk pembahasan dashboard")
    st.markdown("Dashboard ini memakai empat pendekatan utama agar sesuai dengan topik recommendation system.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='method-card'><h4>1. Rule-Based Recommendation</h4><p>Rekomendasi dipilih menggunakan aturan eksplisit seperti genre, harga, minimal positivity, minimal review, mode bermain, dan tag wajib.</p><p><b>Kelebihan:</b> mudah dijelaskan dan cocok untuk cold-start user.</p></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='method-card'><h4>2. Content-Based Recommendation</h4><p>Item profile dibangun dari genre, tag, kategori, developer, publisher, dan deskripsi singkat. Teks dikonversi menjadi TF-IDF, lalu dihitung kemiripannya dengan cosine similarity.</p><p><b>Formula:</b> similarity(user, item) = cosine(TF-IDF user profile, TF-IDF item profile).</p></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='method-card'><h4>3. Collaborative / Crowd Signal</h4><p>Jika file interaksi user-item diupload, sistem memakai item-based collaborative filtering. Jika tidak, dashboard memakai proxy crowd wisdom dari Bayesian rating, volume review, dan popularity.</p><p><b>Catatan ilmiah:</b> proxy crowd wisdom bukan pure CF, tetapi aman untuk dataset agregat yang tidak punya user_id.</p></div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='method-card'><h4>4. Weighted Hybrid Recommendation</h4><p>Skor akhir menggabungkan content match, crowd/collaborative signal, rule fit, value, dan novelty.</p><p><b>Formula:</b> S = w1*C_content + w2*C_crowd + w3*C_rule + w4*C_value + w5*C_novelty.</p></div>
        """, unsafe_allow_html=True)
    st.markdown("### Rumus penting")
    st.latex(r"WR = \frac{v}{v+m}R + \frac{m}{v+m}C")
    st.markdown("""
    Keterangan: `R` adalah positivity item, `v` adalah jumlah review/recommendation, `C` adalah rata-rata positivity seluruh item, dan `m` adalah ambang minimum berbasis kuantil. Rumus ini membuat game dengan review sedikit tidak langsung menang hanya karena positivity tinggi.
    """)
    st.markdown("### Keterbatasan")
    st.markdown("""
    - Dataset Steam top games biasanya bersifat agregat, sehingga tidak selalu memiliki matriks `user_id x item`. Karena itu, true collaborative filtering hanya aktif jika file interaksi user-item ditambahkan.
    - Content-based recommendation sangat bergantung pada kualitas metadata seperti tag, genre, dan deskripsi.
    - Hybrid recommendation lebih robust, tetapi bobotnya perlu divalidasi dengan data interaksi nyata atau A/B testing jika digunakan di lingkungan produksi.
    """)


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
inject_css()
inject_extra_css()

current_view = query_value("view", "home").lower().strip() or "home"
selected_tag = unquote(query_value("tag", "")).strip()
selected_game = query_value("game", "")
if current_view not in {"home", "library", "recommend", "analytics", "methodology", "detail"}:
    current_view = "home"

sidebar_brand()
sidebar_nav(current_view)

uploaded_games = st.sidebar.file_uploader("Upload CSV dataset game", type=["csv"], help="Opsional. Jika kosong, aplikasi membaca steam_top_games_2026.csv di folder yang sama.")
uploaded_interactions = st.sidebar.file_uploader("Opsional: upload interaksi user-item", type=["csv"], help="Untuk true collaborative filtering. Kolom minimal: user_id dan app_id/name; rating/playtime/liked opsional.")

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
default_tag = [selected_tag] if selected_tag in all_tags else []
global_tags = st.sidebar.multiselect("Tag global", all_tags, default=default_tag, max_selections=5)
global_mode = st.sidebar.selectbox("Mode global", ["any", "singleplayer", "multiplayer", "coop"])
global_search = st.sidebar.text_input("Cari judul")
filtered = apply_global_filters(games, year_range, global_price, global_min_pos, global_genres, global_tags, global_mode, global_search)

st.caption(f"Data source: {data_source} | Jumlah data: {len(games):,} game | Setelah filter: {len(filtered):,} game")

if current_view == "detail":
    render_detail(selected_game, games, tfidf_matrix)
elif current_view == "library":
    render_library(games, filtered, selected_tag=selected_tag)
elif current_view == "recommend":
    render_recommender(games, vectorizer, tfidf_matrix, interactions, all_titles, all_genres, all_tags, price_limit_global)
elif current_view == "analytics":
    render_analytics(filtered)
elif current_view == "methodology":
    render_methodology()
else:
    render_home(games, filtered, data_source)
