# SteamVault Pro - Steam Game Discovery & Hybrid Recommendation Dashboard
# Put this file in the same folder as steam_top_games_2026.csv, or upload the CSV from the sidebar.

from __future__ import annotations

import html
import io
import math
import re
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
def render_html(markup: str) -> None:
    """Render HTML/CSS with the safest Streamlit API available."""
    try:
        st.html(markup)
    except Exception:
        st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    render_html(
        """
        <style>
        :root {
            --bg-0: #05070d;
            --bg-1: #08111f;
            --bg-2: #0f172a;
            --panel: rgba(15, 23, 42, 0.88);
            --panel-2: rgba(17, 31, 54, 0.94);
            --panel-3: rgba(30, 41, 59, 0.88);
            --line: rgba(148, 163, 184, 0.20);
            --line-2: rgba(56, 189, 248, 0.38);
            --text: #edf5ff;
            --text-2: #dbeafe;
            --muted: #9fb0c8;
            --soft: #cbd5e1;
            --blue: #38bdf8;
            --cyan: #22d3ee;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #fb7185;
            --purple: #a78bfa;
        }

        html, body, .stApp {
            background: #05070d !important;
            color: var(--text) !important;
        }
        .stApp {
            background:
                radial-gradient(circle at 10% -5%, rgba(56, 189, 248, 0.18), transparent 34rem),
                radial-gradient(circle at 90% 0%, rgba(167, 139, 250, 0.16), transparent 32rem),
                radial-gradient(circle at 50% 105%, rgba(52, 211, 153, 0.08), transparent 42rem),
                linear-gradient(135deg, var(--bg-0), var(--bg-1) 52%, #050816) !important;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li, .stApp label, .stApp span,
        .stApp [data-testid="stMarkdownContainer"] {
            color: var(--text) !important;
        }
        .stApp a { color: #7dd3fc !important; }
        .stApp small, .stApp .muted, .stApp [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
        }
        h1, h2, h3 { letter-spacing: -0.035em; }

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(5, 7, 13, 0.97), rgba(8, 17, 31, 0.95)),
                radial-gradient(circle at top left, rgba(56,189,248,0.11), transparent 18rem) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.17);
        }
        section[data-testid="stSidebar"] > div {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
            color: var(--text-2) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
        }

        /* Force widgets to stay readable in Streamlit light/dark/system themes. */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(15, 23, 42, 0.96) !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
            box-shadow: none !important;
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] input {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            caret-color: var(--blue) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg,
        div[data-baseweb="select"] div {
            color: var(--text) !important;
            fill: var(--text) !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[role="listbox"],
        ul[role="listbox"] {
            background: #0b1120 !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
        }
        div[role="option"], li[role="option"] {
            background: #0b1120 !important;
            color: var(--text) !important;
        }
        div[role="option"]:hover, li[role="option"]:hover {
            background: rgba(56, 189, 248, 0.16) !important;
        }
        div[data-baseweb="tag"] {
            background: rgba(56, 189, 248, 0.18) !important;
            border: 1px solid rgba(56, 189, 248, 0.30) !important;
            color: #dff6ff !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(15, 23, 42, 0.86) !important;
            border: 1px dashed rgba(56, 189, 248, 0.38) !important;
            border-radius: 18px !important;
            color: var(--text) !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--text) !important;
        }
        [data-testid="stFileUploaderDropzone"] button,
        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(135deg, rgba(56,189,248,0.20), rgba(167,139,250,0.18)) !important;
            border: 1px solid rgba(125, 211, 252, 0.36) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
            font-weight: 800 !important;
        }
        [data-testid="stFileUploaderDropzone"] small {
            color: var(--muted) !important;
        }
        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #ffd1dc !important;
            font-weight: 900 !important;
        }
        [data-testid="stSlider"] p {
            color: var(--text-2) !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17, 31, 54, 0.92), rgba(8, 17, 31, 0.94));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 1.25rem;
            padding: 1rem 1rem;
            box-shadow: 0 18px 54px rgba(0,0,0,0.28);
        }
        div[data-testid="stMetric"] label { color: var(--muted) !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line-2);
            border-radius: 30px;
            padding: 34px;
            margin: 0 0 18px 0;
            background:
                linear-gradient(135deg, rgba(56,189,248,0.17), rgba(167,139,250,0.12)),
                linear-gradient(180deg, rgba(17,31,54,0.88), rgba(8,17,31,0.94));
            box-shadow: 0 30px 95px rgba(0, 0, 0, 0.35);
        }
        .hero:after {
            content: "";
            position: absolute;
            right: -120px;
            top: -120px;
            width: 315px;
            height: 315px;
            border-radius: 50%;
            background: rgba(56,189,248,0.13);
            filter: blur(1px);
        }
        .hero-kicker {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(56,189,248,0.14);
            border: 1px solid rgba(56,189,248,0.28);
            color: #bae6fd !important;
            font-size: 0.82rem;
            font-weight: 900;
            margin-bottom: 12px;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.3rem, 5vw, 4.7rem);
            line-height: 0.94;
            color: #ffffff !important;
            text-shadow: 0 18px 50px rgba(0,0,0,0.34);
        }
        .hero p {
            max-width: 880px;
            color: var(--soft) !important;
            font-size: 1.06rem;
            margin-top: 16px;
            margin-bottom: 0;
        }
        .glass-panel {
            background: linear-gradient(180deg, rgba(17,31,54,0.93), rgba(15,23,42,0.90));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 22px;
            padding: 18px;
            box-shadow: 0 18px 54px rgba(0,0,0,0.25);
            margin-bottom: 16px;
            color: var(--text) !important;
        }
        .section-title {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 12px;
            margin: 14px 0 12px 0;
        }
        .section-title h3 { margin: 0; color: #f8fbff !important; }
        .muted { color: var(--muted) !important; }

        .game-card {
            height: 100%;
            min-height: 100%;
            background: linear-gradient(180deg, rgba(17,31,54,0.98), rgba(10,18,32,0.98));
            border: 1px solid rgba(148,163,184,0.19);
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.28);
            transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
            margin-bottom: 18px;
        }
        .game-card:hover {
            transform: translateY(-4px);
            border-color: rgba(56,189,248,0.55);
            box-shadow: 0 26px 82px rgba(0,0,0,0.39);
        }
        .game-img-wrap {
            width: 100%;
            aspect-ratio: 2.16 / 1;
            background: linear-gradient(135deg, rgba(56,189,248,0.16), rgba(167,139,250,0.11));
            overflow: hidden;
            border-bottom: 1px solid rgba(148,163,184,0.16);
        }
        .game-img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .game-body { padding: 15px 16px 17px 16px; }
        .game-title {
            font-size: 1.05rem;
            font-weight: 900;
            color: #f8fbff !important;
            line-height: 1.25;
            margin-bottom: 7px;
        }
        .game-title a { color: #f8fbff !important; text-decoration: none; }
        .game-title a:hover { color: #7dd3fc !important; }
        .meta-line {
            color: var(--muted) !important;
            font-size: 0.83rem;
            margin-bottom: 10px;
        }
        .pill-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 9px 0; }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.01em;
            border: 1px solid rgba(148,163,184,0.18);
            color: #eaf2ff !important;
            background: rgba(148,163,184,0.12);
            white-space: nowrap;
        }
        .pill-blue { background: rgba(56,189,248,0.16); color: #bae6fd !important; border-color: rgba(56,189,248,0.34); }
        .pill-green { background: rgba(52,211,153,0.16); color: #bbf7d0 !important; border-color: rgba(52,211,153,0.34); }
        .pill-amber { background: rgba(251,191,36,0.16); color: #fde68a !important; border-color: rgba(251,191,36,0.34); }
        .pill-red { background: rgba(251,113,133,0.18); color: #fecdd3 !important; border-color: rgba(251,113,133,0.36); }
        .tag {
            display: inline-flex;
            padding: 5px 9px;
            border-radius: 999px;
            background: rgba(148,163,184,0.11);
            border: 1px solid rgba(148,163,184,0.20);
            color: #dbeafe !important;
            font-size: 0.72rem;
            margin: 0 5px 6px 0;
        }
        .why {
            border: 1px solid rgba(56,189,248,0.18);
            background: linear-gradient(180deg, rgba(56,189,248,0.09), rgba(15,23,42,0.24));
            margin-top: 12px;
            padding: 11px 12px;
            border-radius: 15px;
            color: var(--soft) !important;
            font-size: 0.82rem;
            line-height: 1.45;
        }
        .why b, .why-label {
            color: #bae6fd !important;
            font-weight: 900;
        }
        .bar-row { margin: 8px 0; }
        .bar-label {
            display: flex;
            justify-content: space-between;
            color: var(--muted) !important;
            font-size: 0.72rem;
            margin-bottom: 4px;
        }
        .bar-label span { color: var(--muted) !important; }
        .bar-track {
            height: 7px;
            border-radius: 999px;
            background: rgba(148,163,184,0.14);
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--blue), var(--purple));
        }
        .mini-note {
            border-left: 3px solid var(--blue);
            background: rgba(56,189,248,0.09);
            border-radius: 15px;
            padding: 12px 14px;
            color: var(--soft) !important;
            margin: 8px 0 16px 0;
        }
        .method-card {
            background: linear-gradient(180deg, rgba(17,31,54,0.93), rgba(15,23,42,0.91));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 19px;
            padding: 17px;
            height: 100%;
            box-shadow: 0 16px 44px rgba(0,0,0,0.24);
        }
        .method-card h4 { margin-top: 0; color: #f8fbff !important; }
        .method-card p { color: var(--soft) !important; }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(148,163,184,0.08);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 999px;
            padding: 9px 15px;
            color: var(--text-2) !important;
        }
        .stTabs [data-baseweb="tab"] p { color: var(--text-2) !important; }
        .stTabs [aria-selected="true"] {
            background: rgba(56,189,248,0.16) !important;
            border-color: rgba(56,189,248,0.38) !important;
        }
        .stTabs [aria-selected="true"] p { color: #bae6fd !important; font-weight: 900 !important; }

        [data-testid="stExpander"] {
            background: rgba(15,23,42,0.86) !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
            border-radius: 18px !important;
        }
        .stAlert {
            background: rgba(15,23,42,0.90) !important;
            color: var(--text) !important;
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
    return f"""
    <div class='bar-row'>
      <div class='bar-label'><span>{esc(label)}</span><span>{pct}</span></div>
      <div class='bar-track'><div class='bar-fill' style='width:{pct}%'></div></div>
    </div>
    """


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
) -> str:
    title = esc(row.get("name", "Unknown Game"))
    url = esc(steam_url(row))
    title_html = (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
        if url
        else title
    )
    img = esc(str(row.get("header_image", "")).strip())
    img_html = f'<img src="{img}" alt="{title} cover" loading="lazy">' if img else ""
    genre = esc(row.get("genre_primary", "Unknown"))
    year = fmt_int(row.get("year"))
    score = fmt_float(row.get("final_score_pct", row.get("display_score", 0)), 1)
    pos = fmt_float(row.get("positivity"), 1, "%")
    recs = fmt_int(row.get("review_volume"))
    play = fmt_float(row.get("playtime_h"), 1, "h")
    tags = row.get("tag_list", []) if isinstance(row.get("tag_list", []), list) else []
    tag_html = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags[:7])
    why = esc(explain_row(row, games, favorite_titles, preferred_tags))
    comp_html = ""
    if show_components:
        comp_html = (
            component_bar("Content match", float(row.get("content_component", 0)))
            + component_bar("Crowd signal", float(row.get("crowd_component", 0)))
            + component_bar("Rule fit", float(row.get("rule_component", 0)))
            + component_bar("Value", float(row.get("value_component", 0)))
        )
    return f"""
    <article class="game-card">
      <div class="game-img-wrap">{img_html}</div>
      <div class="game-body">
        <div class="game-title">{title_html}</div>
        <div class="meta-line">{genre} | {year} | {price_badge(row)}</div>
        <div class="pill-row">
          <span class="pill pill-blue">Score {score}</span>
          <span class="pill pill-green">Pos {pos}</span>
          <span class="pill">Reviews {recs}</span>
          <span class="pill">Playtime {play}</span>
        </div>
        <div>{tag_html}</div>
        {comp_html}
        <div class="why"><span class="why-label">Why:</span> {why}</div>
      </div>
    </article>
    """

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
    cards = st.columns(columns)
    for i, (_, row) in enumerate(rows.iterrows()):
        with cards[i % columns]:
            render_html(game_card_html(row, games, favorite_titles, preferred_tags, show_components))


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


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
inject_css()

st.sidebar.markdown("## SteamVault Pro")
st.sidebar.caption("Dashboard analisis dan sistem rekomendasi game Steam.")
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
global_tags = st.sidebar.multiselect("Tag wajib global", all_tags, max_selections=5)
global_mode = st.sidebar.selectbox("Mode global", ["any", "singleplayer", "multiplayer", "coop"])
global_search = st.sidebar.text_input("Cari judul")
filtered = apply_global_filters(games, year_range, global_price, global_min_pos, global_genres, global_tags, global_mode, global_search)

render_html(
    f"""
    <div class='hero'>
      <div class='hero-kicker'>Recommendation System Dashboard</div>
      <h1>{APP_TITLE}</h1>
      <p>
        Dashboard ini menggabungkan eksplorasi data, content-based recommendation, rule-based filtering,
        crowd/collaborative signal, dan weighted hybrid recommendation. Fokusnya bukan hanya menampilkan data,
        tetapi juga menjelaskan kenapa sebuah game direkomendasikan.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Data source: {data_source} | Jumlah data: {len(games):,} game | Setelah filter: {len(filtered):,} game")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total games", f"{len(games):,}")
kpi2.metric("Filtered", f"{len(filtered):,}")
kpi3.metric("Free games", f"{int(filtered['is_free'].sum()):,}" if not filtered.empty else "0")
kpi4.metric("Avg positivity", fmt_float(filtered["positivity"].mean() if not filtered.empty else np.nan, 1, "%"))
kpi5.metric("Avg score", fmt_float((filtered["quality_score"].mean() * 100) if not filtered.empty else np.nan, 1))


tab_overview, tab_explore, tab_recommender, tab_evaluation, tab_method = st.tabs(
    ["Ringkasan", "Eksplorasi", "Rekomendasi", "Evaluasi", "Metodologi"]
)

with tab_overview:
    render_html("<div class='section-title'><h3>Insight utama</h3><span class='muted'>overview dataset</span></div>")
    if filtered.empty:
        st.warning("Tidak ada data pada filter global saat ini.")
    else:
        c1, c2 = st.columns([1.12, 0.88])
        with c1:
            genre_count = filtered.groupby("genre_primary", as_index=False).size().sort_values("size", ascending=False).head(14)
            fig = px.bar(genre_count, x="size", y="genre_primary", orientation="h", title="Top genre berdasarkan jumlah game", labels={"size": "Jumlah", "genre_primary": "Genre"})
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(clean_plotly(fig, height=430), use_container_width=True)
        with c2:
            top_tags = safe_top_tags(filtered, 14)
            if not top_tags.empty:
                fig = px.bar(top_tags, x="count", y="tag", orientation="h", title="Top tag paling sering muncul", labels={"count": "Jumlah", "tag": "Tag"})
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(clean_plotly(fig, height=430), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            price_df = filtered[filtered["price_effective"].notna()].copy()
            fig = px.histogram(price_df, x="price_effective", nbins=30, title="Distribusi harga", labels={"price_effective": "Harga efektif ($)", "count": "Jumlah"})
            st.plotly_chart(clean_plotly(fig, height=340), use_container_width=True)
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
            st.plotly_chart(clean_plotly(fig, height=340), use_container_width=True)

        render_html("<div class='section-title'><h3>Top picks cepat</h3><span class='muted'>quality, value, dan popularity</span></div>")
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
                render_cards(data, games, columns=1)

with tab_explore:
    render_html("<div class='section-title'><h3>Game explorer</h3><span class='muted'>browse dan shortlist kandidat</span></div>")
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
        render_cards(browse, games, columns=3)
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
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download hasil filter CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="steamvault_filtered_games.csv",
            mime="text/csv",
        )

with tab_recommender:
    render_html("<div class='section-title'><h3>Smart recommender</h3><span class='muted'>hybrid, explainable, configurable</span></div>")
    st.markdown(
        "<div class='mini-note'>Tips: pilih 1-5 game favorit atau beberapa tag/genre. Jika tidak ada input favorit, sistem otomatis menjadi cold-start recommender berbasis rule, value, dan crowd signal.</div>",
        unsafe_allow_html=True,
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
        st.markdown(
            f"<div class='mini-note'><b>Engine aktif:</b> {esc(engine)} | <b>Sinyal kolaboratif:</b> {esc(source_label)} | Hasil sudah direrank dengan diversity penalty.</div>",
            unsafe_allow_html=True,
        )
        render_cards(recs, games, favorite_titles, preferred_tags, columns=3, show_components=True)

        st.markdown("### Score breakdown")
        chart_df = recs.head(10)[["name", "content_component", "crowd_component", "rule_component", "value_component", "novelty_component", "final_score"]].copy()
        chart_long = chart_df.melt(id_vars="name", var_name="component", value_name="score")
        fig = px.bar(chart_long, x="score", y="name", color="component", orientation="h", barmode="group", title="Komponen skor top recommendation", labels={"score": "Skor 0-1", "name": "Game"})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(clean_plotly(fig, height=470), use_container_width=True)

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

with tab_evaluation:
    render_html("<div class='section-title'><h3>Evaluasi rekomendasi</h3><span class='muted'>quality, diversity, coverage</span></div>")
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
            st.plotly_chart(clean_plotly(fig, height=380), use_container_width=True)
        with e2:
            top_tag_eval = pd.DataFrame(tag_counter.most_common(12), columns=["tag", "count"])
            if not top_tag_eval.empty:
                fig = px.bar(top_tag_eval, x="count", y="tag", orientation="h", title="Top tag pada hasil rekomendasi")
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(clean_plotly(fig, height=380), use_container_width=True)

        st.markdown("### Interpretasi evaluasi")
        st.markdown(
            f"""
            - **Final score rata-rata** menunjukkan kekuatan rekomendasi berdasarkan engine yang dipilih.
            - **Intra-list diversity** mendekati 1 berarti rekomendasi lebih bervariasi; mendekati 0 berarti hasil sangat mirip satu sama lain.
            - **Genre/tag coverage** membantu melihat apakah sistem terlalu sempit atau sudah cukup beragam.
            - **Harga rata-rata** saat ini sekitar **${avg_price:.2f}**, sehingga bisa dipakai untuk membahas aspek value-for-money.
            """
        )

with tab_method:
    render_html("<div class='section-title'><h3>Metodologi sistem rekomendasi</h3><span class='muted'>siap dipakai untuk pembahasan dashboard</span></div>")
    st.markdown(
        """
        Dashboard ini memakai empat pendekatan utama agar sesuai dengan topik recommendation system.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class='method-card'>
            <h4>1. Rule-Based Recommendation</h4>
            <p>Rekomendasi dipilih menggunakan aturan eksplisit seperti genre, harga, minimal positivity, minimal review, mode bermain, dan tag wajib.</p>
            <p><b>Kelebihan:</b> mudah dijelaskan dan cocok untuk cold-start user.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='method-card'>
            <h4>2. Content-Based Recommendation</h4>
            <p>Item profile dibangun dari genre, tag, kategori, developer, publisher, dan deskripsi singkat. Teks dikonversi menjadi TF-IDF, lalu dihitung kemiripannya dengan cosine similarity.</p>
            <p><b>Formula:</b> similarity(user, item) = cosine(TF-IDF user profile, TF-IDF item profile).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class='method-card'>
            <h4>3. Collaborative / Crowd Signal</h4>
            <p>Jika file interaksi user-item diupload, sistem memakai item-based collaborative filtering. Jika tidak, dashboard memakai proxy crowd wisdom dari Bayesian rating, volume review, dan popularity.</p>
            <p><b>Catatan ilmiah:</b> proxy crowd wisdom bukan pure CF, tetapi aman untuk dataset agregat yang tidak punya user_id.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class='method-card'>
            <h4>4. Weighted Hybrid Recommendation</h4>
            <p>Skor akhir menggabungkan content match, crowd/collaborative signal, rule fit, value, dan novelty.</p>
            <p><b>Formula:</b> S = w1*C_content + w2*C_crowd + w3*C_rule + w4*C_value + w5*C_novelty.</p>
            </div>
            """,
            unsafe_allow_html=True,
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
