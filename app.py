from __future__ import annotations

import html as htmllib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

APP_NAME = "AI-Powered Misinformation Detection System"
APP_VERSION = "2.2.0"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = (
    BASE_DIR / "models" / "misinformation_model_package.joblib",
    BASE_DIR / "misinformation_model_package.joblib",
)

MIN_INPUT_LENGTH = 10
MAX_TITLE_LENGTH = 2_000
MAX_BODY_LENGTH = 200_000

EXAMPLES = {
    "real": {
        "title": "Senate committee advances bipartisan infrastructure funding bill",
        "text": (
            "WASHINGTON (Reuters) - A U.S. Senate committee voted on Tuesday to "
            "advance a bipartisan bill that would fund road, bridge and broadband "
            "projects over the next five years, sending the measure to the full "
            "chamber for debate. The vote followed weeks of negotiations between "
            "Republican and Democratic aides, and lawmakers said they expected a "
            "floor vote before the end of the month. A spokesman for the committee "
            "chairman said amendments from both parties had been incorporated."
        ),
    },
    "fake": {
        "title": "SHOCKING: Secret documents PROVE elites are hiding the truth from you!!!",
        "text": (
            "You will NOT believe what an anonymous insider just leaked. The mainstream "
            "media is completely silent while a so-called expert admits everything "
            "we suspected was true. Share this before they delete it! Wake up, people, "
            "they do not want you to see this video that is already being banned "
            "everywhere. Patriots everywhere are demanding answers right now."
        ),
    },
}


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="TruthLens AI · Misinformation Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_html(markup: str) -> None:
    """Render HTML safely (strips indentation so Markdown never treats it as code)."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_json(data: Any) -> None:
    """Dark-themed JSON viewer (replaces st.json)."""

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, str):
            return " ".join(obj.split())  # collapses stray newlines/spaces
        return obj

    pretty = json.dumps(clean(data), indent=2, ensure_ascii=False, default=str)
    st.code(pretty, language="json")


# ============================================================
# Design System
# ============================================================

render_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070b14;
    --glass: rgba(255,255,255,.04);
    --glass-2: rgba(255,255,255,.07);
    --border: rgba(255,255,255,.09);
    --border-2: rgba(129,140,248,.35);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --dim: #64748b;
    --indigo: #6366f1;
    --blue: #3b82f6;
    --cyan: #22d3ee;
    --green: #34d399;
    --red: #fb7185;
    --amber: #fbbf24;
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(900px 500px at 8% -8%, rgba(99,102,241,.22), transparent 60%),
        radial-gradient(800px 480px at 95% 0%, rgba(34,211,238,.13), transparent 60%),
        radial-gradient(700px 500px at 50% 110%, rgba(139,92,246,.12), transparent 60%),
        var(--bg);
}

header[data-testid="stHeader"], footer, [data-testid="stToolbar"], #MainMenu { display: none !important; }

.block-container { max-width: 1180px; padding: 1.4rem 1.6rem 4rem; }

h1,h2,h3,h4 { color: var(--text) !important; letter-spacing: -.02em; }
p, li, span, label { color: inherit; }

/* ---------- Top bar ---------- */
.topbar {
    display:flex; align-items:center; justify-content:space-between; gap:16px;
    padding: 12px 16px; margin-bottom: 18px;
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 18px; backdrop-filter: blur(18px);
}
.brand { display:flex; align-items:center; gap:12px; }
.logo {
    width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
    background: linear-gradient(135deg, #6366f1, #22d3ee); color:#fff; font-size:20px;
    box-shadow: 0 10px 30px rgba(99,102,241,.35);
}
.brand-name { font-weight:800; font-size:17px; letter-spacing:-.02em; color:#fff; }
.brand-tag  { font-size:12px; color: var(--dim); margin-top:1px; }
.status-pill {
    display:flex; align-items:center; gap:8px; padding: 8px 14px; border-radius: 999px;
    background: rgba(52,211,153,.09); border:1px solid rgba(52,211,153,.25);
    color:#a7f3d0; font-size:12.5px; font-weight:700;
}
.dot { width:8px; height:8px; border-radius:50%; background:#34d399; box-shadow:0 0 12px #34d399; }

/* ---------- Tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; padding: 6px; width: fit-content; max-width: 100%;
    background: var(--glass); border: 1px solid var(--border); border-radius: 16px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px; padding: 0 20px; border-radius: 11px; background: transparent;
    color: var(--muted); font-weight: 700; font-size: 15px;
}
.stTabs [data-baseweb="tab"] p { font-size: 15px !important; font-weight: 700 !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    color: #fff !important; box-shadow: 0 8px 24px rgba(99,102,241,.35);
}
.stTabs [aria-selected="true"] p { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 22px; }

/* ---------- Hero ---------- */
.hero {
    position:relative; overflow:hidden; padding: 38px 40px; border-radius: 26px;
    border:1px solid var(--border-2);
    background: linear-gradient(135deg, rgba(30,41,90,.65), rgba(10,16,34,.85));
    box-shadow: 0 30px 80px rgba(0,0,0,.35);
}
.hero::after {
    content:""; position:absolute; right:-120px; top:-140px; width:380px; height:380px; border-radius:50%;
    background: radial-gradient(circle, rgba(34,211,238,.28), transparent 65%);
}
.hero > * { position:relative; z-index:1; }
.eyebrow { color:#7dd3fc; font-size:12px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { margin:10px 0 0; font-size: clamp(32px, 4.6vw, 54px); line-height:1.05; font-weight:800; letter-spacing:-.04em; color:#fff; }
.grad { background: linear-gradient(90deg, #818cf8, #22d3ee); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-copy { max-width: 680px; margin-top:16px; color:#b6c2d6; font-size:16px; line-height:1.7; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }
.chip { padding:7px 13px; border-radius:999px; background:rgba(255,255,255,.07); border:1px solid var(--border); color:#dbeafe; font-size:12.5px; font-weight:700; }

/* ---------- Section headings ---------- */
.sec { margin: 34px 0 14px; }
.sec-k { color:#818cf8; font-size:12px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.sec-t { color:#fff; font-size:26px; font-weight:800; letter-spacing:-.03em; margin-top:4px; }
.sec-c { color: var(--muted); font-size:14.5px; margin-top:4px; line-height:1.6; }

/* ---------- Glass containers (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass); border: 1px solid var(--border) !important; border-radius: 22px !important;
    padding: 8px 10px; backdrop-filter: blur(14px);
}
.card-h { color:#fff; font-size:18px; font-weight:800; letter-spacing:-.02em; }
.card-s { color: var(--muted); font-size:14px; line-height:1.6; margin: 4px 0 6px; }

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] label p, [data-testid="stTextArea"] label p {
    color:#e2e8f0 !important; font-size:14px !important; font-weight:700 !important;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    background: rgba(5,9,20,.7) !important; color:#f8fafc !important;
    border: 1px solid var(--border) !important; border-radius: 14px !important;
    font-size: 15px !important; padding: 12px 14px !important;
}
[data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color:#5b6b85 !important; }
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
    border-color: #818cf8 !important; box-shadow: 0 0 0 3px rgba(129,140,248,.2) !important;
}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] { background: transparent !important; border: 0 !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    width: 100%; min-height: 48px; border-radius: 13px; transition: all .18s ease;
    background: var(--glass-2); border: 1px solid var(--border); color:#e2e8f0;
}
.stButton > button p, .stDownloadButton > button p {
    color: inherit !important; font-size: 15px !important; font-weight: 700 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px); border-color: var(--border-2); background: rgba(255,255,255,.1); color:#fff;
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important; border: 0 !important; color:#fff !important;
    box-shadow: 0 12px 30px rgba(99,102,241,.35);
}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color:#fff !important; }
.stButton > button[kind="primary"]:hover { box-shadow: 0 16px 38px rgba(99,102,241,.5); }

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(99,102,241,.06) !important; border: 1.5px dashed rgba(129,140,248,.4) !important;
    border-radius: 18px !important; padding: 22px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #cbd5e1 !important; }
[data-testid="stFileUploaderDropzone"] small { color: var(--dim) !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important; border:0 !important; border-radius: 11px !important;
}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color:#fff !important; }

/* ---------- Result ---------- */
.verdict { padding: 24px; border-radius: 22px; border:1px solid var(--border); }
.verdict-real { background: linear-gradient(135deg, rgba(16,185,129,.16), rgba(8,14,28,.9)); border-color: rgba(52,211,153,.35); }
.verdict-fake { background: linear-gradient(135deg, rgba(244,63,94,.16), rgba(8,14,28,.9)); border-color: rgba(251,113,133,.35); }
.v-top { display:flex; align-items:center; gap:22px; flex-wrap:wrap; }
.ring {
    width:128px; height:128px; border-radius:50%; display:grid; place-items:center; flex:none;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(148,163,184,.18) 0);
}
.ring-in { width:100px; height:100px; border-radius:50%; background:#0a1020; display:grid; place-items:center; align-content:center; }
.ring-in b { font-size:25px; font-weight:800; color:#fff; line-height:1; }
.ring-in span { font-size:11px; color: var(--muted); margin-top:3px; font-weight:600; }
.v-kicker { color: var(--muted); font-size:12px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.v-title { font-size:34px; font-weight:800; letter-spacing:-.04em; margin-top:4px; }
.v-real { color: var(--green); } .v-fake { color: var(--red); }
.v-copy { color:#b6c2d6; font-size:14px; line-height:1.6; margin-top:6px; max-width: 380px; }

.meter { margin-top: 22px; }
.meter-head { display:flex; justify-content:space-between; color:#cbd5e1; font-size:13px; font-weight:700; margin-bottom:9px; }
.meter-track { position:relative; height:12px; border-radius:999px; background: rgba(148,163,184,.18); }
.meter-fill { height:100%; border-radius:999px; background: linear-gradient(90deg, #34d399, #fbbf24 55%, #fb7185); }
.meter-mask { position:absolute; top:0; right:0; height:100%; border-radius: 0 999px 999px 0; background: rgba(10,16,32,.88); }
.meter-thr { position:absolute; top:-6px; width:3px; height:24px; border-radius:3px; background:#fff; box-shadow:0 0 10px rgba(255,255,255,.6); }
.meter-legend { display:flex; justify-content:space-between; margin-top:9px; color: var(--dim); font-size:12px; font-weight:600; }

.stat-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top: 16px; }
.stat { padding:14px 16px; border-radius:15px; background: var(--glass); border:1px solid var(--border); }
.stat-l { color: var(--dim); font-size:11.5px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.stat-v { color:#fff; font-size:20px; font-weight:800; margin-top:5px; letter-spacing:-.02em; }
.stat-n { color: var(--dim); font-size:12px; margin-top:2px; }

/* ---------- Empty state ---------- */
.empty {
    min-height: 330px; display:grid; place-items:center; text-align:center; padding: 28px;
    border:1.5px dashed rgba(148,163,184,.25); border-radius: 22px; background: rgba(255,255,255,.02);
}
.empty-ic { width:64px; height:64px; margin:0 auto 14px; display:grid; place-items:center; border-radius:20px; font-size:28px;
    background: linear-gradient(135deg, rgba(99,102,241,.25), rgba(34,211,238,.18)); }
.empty-t { color:#fff; font-size:18px; font-weight:800; }
.empty-c { color: var(--muted); font-size:14px; line-height:1.6; max-width: 300px; margin: 6px auto 0; }

/* ---------- Steps / info cards ---------- */
.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
.step { padding:18px; border-radius:18px; background: var(--glass); border:1px solid var(--border); }
.step-n { width:30px; height:30px; border-radius:10px; display:grid; place-items:center; font-weight:800; font-size:13px; color:#fff;
    background: linear-gradient(135deg, #6366f1, #22d3ee); }
.step-t { color:#fff; font-size:15px; font-weight:800; margin-top:12px; }
.step-c { color: var(--muted); font-size:13px; line-height:1.5; margin-top:4px; }

.info-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; }
.info { padding:18px; border-radius:18px; background: var(--glass); border:1px solid var(--border); }
.info-ic { font-size:22px; }
.info-l { color: var(--dim); font-size:11.5px; font-weight:800; letter-spacing:.09em; text-transform:uppercase; margin-top:8px; }
.info-v { color:#fff; font-size:19px; font-weight:800; margin-top:3px; letter-spacing:-.02em; word-break: break-word; }
.info-n { color: var(--muted); font-size:12.5px; margin-top:3px; }

.dist { display:flex; height:14px; border-radius:999px; overflow:hidden; background: rgba(148,163,184,.18); margin-top:10px; }
.dist-real { background: var(--green); } .dist-fake { background: var(--red); } .dist-err { background: var(--amber); }

.note {
    margin-top: 18px; padding: 14px 18px; border-radius: 15px; font-size:13.5px; line-height:1.65; color: var(--muted);
    background: rgba(99,102,241,.08); border:1px solid rgba(129,140,248,.22);
}
.note strong { color:#c7d2fe; }

.kv { display:flex; justify-content:space-between; gap:16px; padding: 14px 4px; border-bottom:1px solid var(--border); }
.kv:last-child { border-bottom:0; }
.kv-k { color: var(--muted); font-size:14px; font-weight:600; }
.kv-v { color:#fff; font-size:14px; font-weight:700; text-align:right; word-break: break-word; }

/* ---------- Expanders (dark fix) ---------- */
[data-testid="stExpander"] {
    border:1px solid var(--border) !important; border-radius:16px !important;
    background: var(--glass) !important; overflow:hidden;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > summary,
[data-testid="stExpanderDetails"] {
    background: transparent !important; color:#e2e8f0 !important;
}
[data-testid="stExpander"] summary:hover { background: rgba(255,255,255,.04) !important; }
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color:#e2e8f0 !important; font-size:14.5px !important; font-weight:700 !important; }
[data-testid="stExpander"] summary svg { color:#cbd5e1 !important; fill:#cbd5e1 !important; }

/* ---------- Dataframes / alerts ---------- */
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:14px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stRadio"] label p { color:#e2e8f0 !important; font-size:14px !important; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--dim) !important; }
[data-testid="stCode"], [data-testid="stCode"] pre {
    background: rgba(5,9,20,.75) !important; border:1px solid var(--border); border-radius:14px !important;
}
[data-testid="stCode"] code { font-size:13px !important; line-height:1.65 !important; white-space: pre-wrap !important; word-break: break-word; }

.footer { margin-top:48px; padding-top:20px; border-top:1px solid var(--border); text-align:center; color: var(--dim); font-size:12.5px; line-height:1.8; }
.footer b { color:#cbd5e1; }

@media (max-width: 800px) {
    .block-container { padding: 1rem .9rem 3rem; }
    .hero { padding: 26px 22px; }
    .topbar { flex-direction: column; align-items: flex-start; }
    .stat-grid { grid-template-columns: 1fr; }
    .v-title { font-size: 28px; }
}
</style>
"""
)


# ============================================================
# Model Loading (unchanged production logic)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_package() -> tuple[dict[str, Any], str]:
    """Load and validate the frozen production package once."""

    model_path = next((path for path in MODEL_PATHS if path.is_file()), None)

    if model_path is None:
        expected = "\n".join(f"- {path}" for path in MODEL_PATHS)
        raise FileNotFoundError(
            "Production model package was not found.\n\n"
            f"Expected one of:\n{expected}"
        )

    try:
        package = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            "The model package could not be loaded. "
            "It may be corrupted or incompatible with the installed "
            "Python/scikit-learn environment."
        ) from exc

    if not isinstance(package, dict):
        raise ValueError("Invalid production package: expected a dictionary.")

    required_keys = {
        "model", "vectorizer", "threshold", "config", "seed",
        "model_name", "feature_policy", "label_mapping", "metadata",
    }
    missing = required_keys.difference(package.keys())
    if missing:
        raise ValueError(
            "Production package is missing required keys: "
            + ", ".join(sorted(missing))
        )

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = package["threshold"]

    if not hasattr(model, "predict_proba"):
        raise TypeError("Saved model does not support predict_proba().")
    if not hasattr(vectorizer, "transform"):
        raise TypeError("Saved vectorizer does not support transform().")
    if not isinstance(threshold, (int, float)):
        raise TypeError("Saved threshold must be numeric.")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("Saved threshold must be between 0 and 1.")

    feature_policy = package["feature_policy"]
    if not isinstance(feature_policy, dict):
        raise TypeError("feature_policy must be a dictionary.")
    if set(feature_policy.get("used", [])) != {"title", "text"}:
        raise ValueError(
            "Production feature policy must use exactly 'title' and 'text'."
        )

    if not isinstance(package["label_mapping"], dict):
        raise TypeError("label_mapping must be a dictionary.")

    return package, str(model_path)


# ============================================================
# Inference (unchanged production logic)
# ============================================================

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def prepare_text(title: str, text: str) -> str:
    """Combine only the production-approved fields. subject is never consumed."""

    title = normalize_text(title)
    text = normalize_text(text)

    if len(title) > MAX_TITLE_LENGTH:
        raise ValueError(f"Headline is too long (max {MAX_TITLE_LENGTH:,} characters).")
    if len(text) > MAX_BODY_LENGTH:
        raise ValueError(f"Article body is too long (max {MAX_BODY_LENGTH:,} characters).")

    parts = [p for p in (title, text) if p]
    combined = " ".join(parts).strip()

    if not combined:
        raise ValueError("Please provide a headline or article body.")
    if len(combined) < MIN_INPUT_LENGTH:
        raise ValueError(
            f"The input is too short. Please provide at least "
            f"{MIN_INPUT_LENGTH} characters."
        )
    return combined


def predict_article(title: str, text: str, package: dict[str, Any]) -> dict[str, Any]:
    """Run inference using the saved production components."""

    combined_text = prepare_text(title, text)

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = float(package["threshold"])
    label_mapping = package["label_mapping"]

    try:
        features = vectorizer.transform([combined_text])
        probabilities = model.predict_proba(features)[0]
    except Exception as exc:
        raise RuntimeError(
            "Inference failed while transforming the input or generating probabilities."
        ) from exc

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError("The saved model does not expose classes_.")

    class_probabilities = {
        int(cls): float(prob) for cls, prob in zip(classes, probabilities)
    }
    if 0 not in class_probabilities or 1 not in class_probabilities:
        raise RuntimeError("The saved model does not contain the expected classes 0 and 1.")

    real_probability = class_probabilities[0]
    fake_probability = class_probabilities[1]

    prediction = int(fake_probability >= threshold)
    label = label_mapping.get(
        prediction, "Fake News" if prediction == 1 else "Real News"
    )
    confidence = fake_probability if prediction == 1 else real_probability

    return {
        "prediction": prediction,
        "label": label,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": confidence,
        "threshold": threshold,
    }


# ============================================================
# Batch Inference (unchanged production logic)
# ============================================================

def find_column(dataframe: pd.DataFrame, target: str) -> Any | None:
    for column in dataframe.columns:
        if str(column).strip().lower() == target:
            return column
    return None


def analyze_batch(dataframe: pd.DataFrame, package: dict[str, Any]) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no rows.")

    title_column = find_column(dataframe, "title")
    text_column = find_column(dataframe, "text")

    if title_column is None and text_column is None:
        raise ValueError(
            "CSV must contain at least one of these columns: 'title' or 'text'."
        )

    results = dataframe.copy()
    predictions, labels = [], []
    real_probabilities, fake_probabilities, confidences = [], [], []

    progress = st.progress(0.0)
    status = st.empty()
    total = len(dataframe)

    for row_number, (_, row) in enumerate(dataframe.iterrows(), start=1):
        title = normalize_text(row[title_column]) if title_column is not None else ""
        text = normalize_text(row[text_column]) if text_column is not None else ""

        try:
            result = predict_article(title=title, text=text, package=package)
            predictions.append(result["prediction"])
            labels.append(result["label"])
            real_probabilities.append(result["real_probability"])
            fake_probabilities.append(result["fake_probability"])
            confidences.append(result["confidence"])
        except Exception:
            predictions.append(None)
            labels.append("Inference Error")
            real_probabilities.append(None)
            fake_probabilities.append(None)
            confidences.append(None)

        if row_number == total or row_number % 25 == 0:
            progress.progress(row_number / total)
            status.caption(f"Processing {row_number:,} / {total:,} rows")

    progress.progress(1.0)
    status.caption(f"Completed {total:,} rows.")

    results["prediction"] = predictions
    results["label"] = labels
    results["real_probability"] = real_probabilities
    results["fake_probability"] = fake_probabilities
    results["confidence"] = confidences
    return results


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def csv_template() -> bytes:
    return dataframe_to_csv(
        pd.DataFrame([{"title": "Example headline", "text": "Example article body"}])
    )


# ============================================================
# UI helpers
# ============================================================

def section(kicker: str, title: str, copy: str = "") -> None:
    copy_html = f'<div class="sec-c">{copy}</div>' if copy else ""
    render_html(
        f"""
        <div class="sec">
            <div class="sec-k">{kicker}</div>
            <div class="sec-t">{title}</div>
            {copy_html}
        </div>
        """
    )


def render_topbar() -> None:
    render_html(
        """
        <div class="topbar">
            <div class="brand">
                <div class="logo">🛡️</div>
                <div>
                    <div class="brand-name">TruthLens AI</div>
                    <div class="brand-tag">Misinformation intelligence</div>
                </div>
            </div>
            <div class="status-pill"><span class="dot"></span>Model online</div>
        </div>
        """
    )


def render_hero(package: dict[str, Any]) -> None:
    threshold = float(package["threshold"])
    render_html(
        f"""
        <div class="hero">
            <div class="eyebrow">AI-powered misinformation detection</div>
            <div class="hero-title">Read the signal.<br><span class="grad">Challenge the story.</span></div>
            <div class="hero-copy">
                Paste a news headline and article, and get an instant classification
                with a clear probability breakdown — powered by a TF-IDF + Linear SVM
                model trained on real and fake news.
            </div>
            <div class="chips">
                <span class="chip">⚡ Instant results</span>
                <span class="chip">📰 Title + Text</span>
                <span class="chip">📊 Probability scores</span>
                <span class="chip">🎯 Threshold {threshold:.2f}</span>
            </div>
        </div>
        """
    )


def render_result(result: dict[str, Any]) -> None:
    label = str(result["label"])
    is_real = label.strip().lower() == "real news"

    real_p = float(result["real_probability"])
    fake_p = float(result["fake_probability"])
    conf = float(result["confidence"])
    thr = float(result["threshold"])

    cls = "verdict-real" if is_real else "verdict-fake"
    txt_cls = "v-real" if is_real else "v-fake"
    color = "#34d399" if is_real else "#fb7185"
    icon = "✅" if is_real else "⚠️"
    verdict = "Likely Real" if is_real else "Likely Fake"
    copy = (
        "The article's language and structure match patterns the model "
        "associates with genuine news reporting."
        if is_real
        else "The article's language and structure match patterns the model "
        "associates with misleading or fabricated news."
    )

    fake_pct = max(0.0, min(100.0, fake_p * 100))
    thr_pct = max(0.0, min(100.0, thr * 100))
    rule = "≥" if fake_p >= thr else "<"

    render_html(
        f"""
        <div class="verdict {cls}">
            <div class="v-top">
                <div class="ring" style="--p:{conf * 100:.1f};--c:{color}">
                    <div class="ring-in"><b>{conf:.0%}</b><span>confidence</span></div>
                </div>
                <div>
                    <div class="v-kicker">Verdict</div>
                    <div class="v-title {txt_cls}">{icon} {verdict}</div>
                    <div class="v-copy">{copy}</div>
                </div>
            </div>
            <div class="meter">
                <div class="meter-head">
                    <span>Fake-news score</span>
                    <span>{fake_p:.1%} {rule} threshold {thr:.1%}</span>
                </div>
                <div class="meter-track">
                    <div class="meter-fill" style="width:100%"></div>
                    <div class="meter-mask" style="width:{100 - fake_pct:.1f}%"></div>
                    <div class="meter-thr" style="left:calc({thr_pct:.1f}% - 1px)"></div>
                </div>
                <div class="meter-legend"><span>0% · Real</span><span>▏ decision threshold</span><span>Fake · 100%</span></div>
            </div>
        </div>
        <div class="stat-grid">
            <div class="stat"><div class="stat-l">Real probability</div><div class="stat-v" style="color:#34d399">{real_p:.1%}</div></div>
            <div class="stat"><div class="stat-l">Fake probability</div><div class="stat-v" style="color:#fb7185">{fake_p:.1%}</div></div>
            <div class="stat"><div class="stat-l">Predicted class</div><div class="stat-v">{result["prediction"]}</div><div class="stat-n">0 = Real · 1 = Fake</div></div>
        </div>
        """
    )

    with st.expander("Technical details"):
        details = {
            "Model": st.session_state.get("_model_name", "Saved production model"),
            "Vectorizer": st.session_state.get("_vectorizer_name", "TF-IDF"),
            "Threshold": f"{thr:.3f}",
            "Input features": "title + text",
            "Excluded features": "subject",
            "Predicted label": label,
        }
        rows = "".join(
            f'<div class="kv"><div class="kv-k">{htmllib.escape(k)}</div>'
            f'<div class="kv-v">{htmllib.escape(str(v))}</div></div>'
            for k, v in details.items()
        )
        render_html(rows)


def render_disclaimer() -> None:
    render_html(
        """
        <div class="note">
            <strong>Responsible AI.</strong> This is an ML-assisted classifier, not proof
            that a claim is true or false. It reads writing patterns, not facts.
            Always verify important information with reliable, authoritative sources.
        </div>
        """
    )


# ============================================================
# Callbacks
# ============================================================

def load_example(kind: str) -> None:
    st.session_state["analyze_title"] = EXAMPLES[kind]["title"]
    st.session_state["analyze_text"] = EXAMPLES[kind]["text"]
    st.session_state.pop("latest_prediction", None)


def clear_inputs() -> None:
    st.session_state["analyze_title"] = ""
    st.session_state["analyze_text"] = ""
    st.session_state.pop("latest_prediction", None)


# ============================================================
# Pages
# ============================================================

def render_analyze_page(package: dict[str, Any]) -> None:
    render_hero(package)

    section(
        "Step 1 · Verify",
        "Analyze a news article",
        "Paste a headline and the article text, or try one of the examples.",
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with st.container(border=True):
            render_html(
                """
                <div class="card-h">📝 Article</div>
                <div class="card-s">More text gives the model more signal to work with.</div>
                """
            )

            e1, e2, e3 = st.columns(3)
            e1.button("📰 Real example", on_click=load_example, args=("real",), key="ex_real")
            e2.button("🚨 Fake example", on_click=load_example, args=("fake",), key="ex_fake")
            e3.button("🧹 Clear", on_click=clear_inputs, key="clear_inputs")

            title = st.text_input(
                "Headline",
                placeholder="e.g. Senate committee advances infrastructure bill",
                key="analyze_title",
            )
            text = st.text_area(
                "Article body",
                placeholder="Paste the full article text here…",
                height=240,
                key="analyze_text",
            )

            chars = len(normalize_text(title)) + len(normalize_text(text))
            st.caption(f"{chars:,} characters · minimum {MIN_INPUT_LENGTH}")

            if st.button(
                "Analyze article  →",
                type="primary",
                use_container_width=True,
                key="run_single_prediction",
            ):
                try:
                    with st.spinner("Analyzing…"):
                        st.session_state["latest_prediction"] = predict_article(
                            title=title, text=text, package=package
                        )
                except ValueError as exc:
                    st.session_state.pop("latest_prediction", None)
                    st.warning(str(exc))
                except Exception as exc:
                    st.session_state.pop("latest_prediction", None)
                    st.error(f"Prediction failed: {exc}")

    with right:
        result = st.session_state.get("latest_prediction")
        if result:
            render_result(result)
        else:
            render_html(
                """
                <div class="empty">
                    <div>
                        <div class="empty-ic">🔎</div>
                        <div class="empty-t">Your result will appear here</div>
                        <div class="empty-c">
                            Enter an article on the left, or load an example,
                            then press <b>Analyze article</b>.
                        </div>
                    </div>
                </div>
                """
            )

    section(
        "Step 2 · Understand",
        "How it works",
        "Every verdict goes through the same four steps.",
    )
    render_html(
        """
        <div class="steps">
            <div class="step"><div class="step-n">1</div><div class="step-t">Combine</div><div class="step-c">Headline and body are merged into a single text.</div></div>
            <div class="step"><div class="step-n">2</div><div class="step-t">Vectorize</div><div class="step-c">TF-IDF turns words into weighted numeric features.</div></div>
            <div class="step"><div class="step-n">3</div><div class="step-t">Score</div><div class="step-c">The Linear SVM outputs real and fake probabilities.</div></div>
            <div class="step"><div class="step-n">4</div><div class="step-t">Decide</div><div class="step-c">The saved threshold converts the score into a verdict.</div></div>
        </div>
        """
    )
    render_disclaimer()


def render_batch_page(package: dict[str, Any]) -> None:
    section(
        "Batch mode",
        "Analyze a whole dataset",
        "Upload a CSV with a title and/or text column and get a prediction for every row.",
    )

    c1, c2 = st.columns([0.42, 0.58], gap="large")
    with c1:
        with st.container(border=True):
            render_html(
                """
                <div class="card-h">⬇️ Start with the template</div>
                <div class="card-s">Download the expected structure and replace the example row with your data.</div>
                """
            )
            st.download_button(
                "Download CSV template",
                data=csv_template(),
                file_name="misinformation_batch_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with c2:
        render_html(
            """
            <div class="info-grid">
                <div class="info"><div class="info-ic">📰</div><div class="info-l">Column</div><div class="info-v">title</div><div class="info-n">Article headline</div></div>
                <div class="info"><div class="info-ic">📄</div><div class="info-l">Column</div><div class="info-v">text</div><div class="info-n">Article body</div></div>
                <div class="info"><div class="info-ic">🚫</div><div class="info-l">Ignored</div><div class="info-v">subject</div><div class="info-n">Not used by the model</div></div>
            </div>
            """
        )

    section("Upload", "Add your CSV", "Preview your data before running the analysis.")

    uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is None:
        render_disclaimer()
        return

    try:
        dataframe = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    render_html(
        f"""
        <div class="info-grid" style="margin-top:14px">
            <div class="info"><div class="info-l">Rows</div><div class="info-v">{len(dataframe):,}</div><div class="info-n">Uploaded records</div></div>
            <div class="info"><div class="info-l">Columns</div><div class="info-v">{len(dataframe.columns):,}</div><div class="info-n">Detected fields</div></div>
            <div class="info"><div class="info-l">Threshold</div><div class="info-v">{float(package["threshold"]):.3f}</div><div class="info-n">Saved decision rule</div></div>
        </div>
        """
    )

    with st.expander("Preview uploaded data", expanded=True):
        st.dataframe(dataframe.head(10), use_container_width=True, hide_index=True)

    st.write("")
    if st.button(
        "Run batch analysis  →",
        type="primary",
        use_container_width=True,
        key="run_batch_prediction",
    ):
        with st.spinner("Analyzing the dataset…"):
            try:
                st.session_state["batch_results"] = analyze_batch(
                    dataframe=dataframe, package=package
                )
            except Exception as exc:
                st.error(f"Batch analysis failed: {exc}")

    results_df = st.session_state.get("batch_results")

    if results_df is not None:
        section("Results", "Batch report", "Predictions from the saved production model.")

        labels = results_df["label"].astype(str).str.lower()
        total = max(len(results_df), 1)
        real_count = int((labels == "real news").sum())
        fake_count = int((labels == "fake news").sum())
        error_count = int(labels.str.contains("error").sum())

        render_html(
            f"""
            <div class="info-grid">
                <div class="info"><div class="info-l">Processed</div><div class="info-v">{len(results_df):,}</div><div class="info-n">Rows returned</div></div>
                <div class="info"><div class="info-l">Real news</div><div class="info-v" style="color:#34d399">{real_count:,}</div><div class="info-n">{real_count / total:.1%} of rows</div></div>
                <div class="info"><div class="info-l">Fake news</div><div class="info-v" style="color:#fb7185">{fake_count:,}</div><div class="info-n">{fake_count / total:.1%} of rows</div></div>
                <div class="info"><div class="info-l">Errors</div><div class="info-v" style="color:#fbbf24">{error_count:,}</div><div class="info-n">Rows to review</div></div>
            </div>
            <div class="dist">
                <div class="dist-real" style="width:{real_count / total * 100:.2f}%"></div>
                <div class="dist-fake" style="width:{fake_count / total * 100:.2f}%"></div>
                <div class="dist-err" style="width:{error_count / total * 100:.2f}%"></div>
            </div>
            """
        )

        st.write("")
        view = st.radio(
            "Show",
            ["All", "Real", "Fake", "Errors"],
            horizontal=True,
            key="batch_filter",
        )
        shown = results_df
        if view == "Real":
            shown = results_df[labels == "real news"]
        elif view == "Fake":
            shown = results_df[labels == "fake news"]
        elif view == "Errors":
            shown = results_df[labels.str.contains("error")]

        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "fake_probability": st.column_config.ProgressColumn(
                    "Fake probability", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "real_probability": st.column_config.ProgressColumn(
                    "Real probability", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )

        st.download_button(
            "Download predictions CSV",
            data=dataframe_to_csv(results_df),
            file_name="misinformation_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    render_disclaimer()


def render_model_overview(package: dict[str, Any]) -> None:
    threshold = float(package["threshold"])
    model_name = htmllib.escape(str(package.get("model_name", "Saved production model")))
    vectorizer_name = htmllib.escape(type(package["vectorizer"]).__name__)

    section(
        "Model overview",
        "One inference path. No surprises.",
        "The app is an inference layer over a saved model package — it never retrains, "
        "retunes, or simulates predictions.",
    )

    render_html(
        f"""
        <div class="info-grid">
            <div class="info"><div class="info-ic">🧠</div><div class="info-l">Classifier</div><div class="info-v">{model_name}</div><div class="info-n">Saved package</div></div>
            <div class="info"><div class="info-ic">🔤</div><div class="info-l">Vectorizer</div><div class="info-v">{vectorizer_name}</div><div class="info-n">Saved package</div></div>
            <div class="info"><div class="info-ic">🎯</div><div class="info-l">Threshold</div><div class="info-v">{threshold:.3f}</div><div class="info-n">Fake if score ≥ threshold</div></div>
            <div class="info"><div class="info-ic">🔒</div><div class="info-l">Mode</div><div class="info-v">Inference only</div><div class="info-n">No training or tuning</div></div>
        </div>
        """
    )

    section("Pipeline", "From text to verdict")
    render_html(
        """
        <div class="steps">
            <div class="step"><div class="step-n">1</div><div class="step-t">Title + Text</div><div class="step-c">Only these two fields are used.</div></div>
            <div class="step"><div class="step-n">2</div><div class="step-t">Combine</div><div class="step-c">Merged into one document.</div></div>
            <div class="step"><div class="step-n">3</div><div class="step-t">TF-IDF</div><div class="step-c">Saved vectorizer creates features.</div></div>
            <div class="step"><div class="step-n">4</div><div class="step-t">Linear SVM</div><div class="step-c">Calibrated classifier scores it.</div></div>
            <div class="step"><div class="step-n">5</div><div class="step-t">predict_proba()</div><div class="step-c">Real and fake probabilities.</div></div>
            <div class="step"><div class="step-n">6</div><div class="step-t">Threshold</div><div class="step-c">Saved rule → Real / Fake.</div></div>
        </div>
        """
    )

    section("Contract", "Configuration & feature policy")
    with st.container(border=True):
        rows = [
            ("Model package", "misinformation_model_package.joblib"),
            ("Inference features", "title + text"),
            ("Excluded feature", "subject"),
            ("Decision method", "predict_proba() + saved threshold"),
            ("Execution mode", "inference only"),
            ("Application version", APP_VERSION),
        ]
        render_html(
            "".join(
                f'<div class="kv"><div class="kv-k">{k}</div><div class="kv-v">{htmllib.escape(v)}</div></div>'
                for k, v in rows
            )
        )

    st.write("")
    config = package.get("config", {})
    if config:
        with st.expander("View saved configuration"):
            render_json(config)

    with st.expander("View package metadata"):
        metadata = package.get("metadata", {})
        if metadata:
            render_json(metadata)
        else:
            st.info("No additional metadata is exposed by the production package.")

    render_html(
        """
        <div class="note">
            <strong>Limitations.</strong> The model learns writing style and vocabulary
            from its training data. It can be wrong on satire, breaking news, opinion
            pieces, or topics far from what it was trained on. It does not fact-check claims.
        </div>
        """
    )
    render_disclaimer()


# ============================================================
# Main
# ============================================================

def main() -> None:
    try:
        package, model_path = load_model_package()
    except FileNotFoundError as exc:
        st.error("Production Model Not Found")
        st.code(str(exc))
        st.info(
            "Place misinformation_model_package.joblib inside the models/ "
            "directory or project root."
        )
        st.stop()
    except (ValueError, TypeError, RuntimeError) as exc:
        st.error("Production Model Error")
        st.code(str(exc))
        st.stop()
    except Exception as exc:
        st.error("Unexpected production model loading error")
        st.code(f"{type(exc).__name__}: {exc}")
        st.stop()

    st.session_state["_model_name"] = str(package.get("model_name", "Saved production model"))
    st.session_state["_vectorizer_name"] = type(package["vectorizer"]).__name__

    render_topbar()

    tab_analyze, tab_batch, tab_model = st.tabs(
        ["🔍  Analyze", "📂  Batch", "🧠  Model"]
    )

    with tab_analyze:
        render_analyze_page(package)
    with tab_batch:
        render_batch_page(package)
    with tab_model:
        render_model_overview(package)

    render_html(
        f"""
        <div class="footer">
            <b>TruthLens AI</b> · {APP_NAME}<br>
            Production NLP · Responsible AI · v{APP_VERSION}<br>
            Model package: {htmllib.escape(Path(model_path).name)}
        </div>
        """
    )


if __name__ == "__main__":
    main()
