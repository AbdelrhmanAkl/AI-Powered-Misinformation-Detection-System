from __future__ import annotations

from io import BytesIO
from pathlib import Path
import textwrap
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

APP_NAME = "AI-Powered Misinformation Detection System"
APP_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = (
    BASE_DIR / "models" / "misinformation_model_package.joblib",
    BASE_DIR / "misinformation_model_package.joblib",
)

MIN_INPUT_LENGTH = 10
MAX_TITLE_LENGTH = 2_000
MAX_BODY_LENGTH = 200_000


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Professional Portfolio UI
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --bg: #07111f;
            --surface: #0c1a2b;
            --surface-2: #101f33;
            --surface-3: #14263d;
            --border: rgba(148, 163, 184, 0.14);
            --text: #f8fafc;
            --muted: #94a3b8;
            --blue: #3b82f6;
            --blue-soft: #93c5fd;
            --green: #34d399;
            --red: #fb7185;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 8% 0%,
                    rgba(37, 99, 235, 0.16),
                    transparent 27%
                ),
                radial-gradient(
                    circle at 94% 8%,
                    rgba(14, 165, 233, 0.10),
                    transparent 24%
                ),
                var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 1280px;
            padding: 2rem 2.25rem 4rem;
        }

        #MainMenu,
        header,
        footer {
            visibility: hidden;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #081525;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .block-container {
            padding: 1.7rem 1.1rem;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #cbd5e1 !important;
        }

        /* Typography */
        h1, h2, h3 {
            color: var(--text) !important;
            letter-spacing: -0.035em !important;
        }

        p {
            color: var(--muted);
        }

        /* Brand */
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 2rem;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            font-size: 0.9rem;
            font-weight: 900;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
        }

        .brand-name {
            color: #f8fafc;
            font-weight: 850;
            font-size: 0.95rem;
        }

        .brand-subtitle {
            color: #64748b;
            font-size: 0.68rem;
            margin-top: 2px;
        }

        /* Hero */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.7rem 2.8rem;
            border: 1px solid rgba(96, 165, 250, 0.20);
            border-radius: 24px;
            background:
                linear-gradient(
                    135deg,
                    rgba(10, 25, 44, 0.98),
                    rgba(15, 43, 78, 0.95)
                );
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.8rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 330px;
            height: 330px;
            right: -150px;
            top: -180px;
            border-radius: 50%;
            border: 1px solid rgba(147, 197, 253, 0.12);
            box-shadow:
                0 0 0 45px rgba(147, 197, 253, 0.025),
                0 0 0 90px rgba(147, 197, 253, 0.018);
        }

        .hero-kicker {
            color: #93c5fd;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            color: #ffffff;
            font-size: clamp(2.15rem, 5vw, 3.55rem);
            line-height: 1.02;
            font-weight: 900;
            letter-spacing: -0.055em;
            max-width: 850px;
            margin: 0;
        }

        .hero-description {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.75;
            max-width: 790px;
            margin: 1rem 0 0;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 1.35rem;
        }

        .hero-badge {
            padding: 7px 11px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.11);
            color: #dbeafe;
            font-size: 0.72rem;
            font-weight: 750;
        }

        /* Cards */
        .card {
            background: rgba(12, 26, 43, 0.86);
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 1.3rem;
            box-shadow: 0 12px 38px rgba(0, 0, 0, 0.16);
        }

        .card-title {
            color: #f8fafc;
            font-size: 0.9rem;
            font-weight: 850;
            margin-bottom: 0.3rem;
        }

        .card-value {
            color: #dbeafe;
            font-size: 1.35rem;
            font-weight: 900;
        }

        .card-caption {
            color: #64748b;
            font-size: 0.72rem;
            line-height: 1.5;
            margin-top: 0.25rem;
        }

        .section-title {
            color: #f8fafc;
            font-size: 1.25rem;
            font-weight: 850;
            margin: 1.2rem 0 0.2rem;
        }

        .section-caption {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 1rem;
        }

        /* Inputs */
        .stTextInput input,
        .stTextArea textarea,
        .stFileUploader section {
            background: #0b192a !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 11px !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
        }

        .stTextInput label,
        .stTextArea label {
            color: #cbd5e1 !important;
            font-weight: 750 !important;
            font-size: 0.78rem !important;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button {
            min-height: 46px;
            border-radius: 10px;
            font-weight: 800;
            border: 1px solid rgba(96, 165, 250, 0.30);
            transition: 0.18s ease;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: 0;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.20);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: rgba(12, 26, 43, 0.88);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: #64748b !important;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }

        /* Result */
        .result {
            padding: 1.6rem;
            border-radius: 19px;
            background: rgba(12, 26, 43, 0.94);
            border: 1px solid var(--border);
            box-shadow: 0 15px 45px rgba(0, 0, 0, 0.20);
        }

        .result-real {
            border-color: rgba(52, 211, 153, 0.28);
        }

        .result-fake {
            border-color: rgba(251, 113, 133, 0.28);
        }

        .result-kicker {
            color: #64748b;
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .result-label {
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin: 0.35rem 0;
        }

        .real-label {
            color: var(--green);
        }

        .fake-label {
            color: var(--red);
        }

        .result-description {
            color: #94a3b8;
            font-size: 0.8rem;
            line-height: 1.6;
        }

        /* Probability */
        .probability-card {
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 13px;
            background: #091727;
            border: 1px solid rgba(148, 163, 184, 0.10);
        }

        .probability-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: #cbd5e1;
            font-size: 0.76rem;
            font-weight: 750;
            margin-bottom: 0.45rem;
        }

        .probability-value {
            color: #f8fafc;
            font-weight: 900;
        }

        .bar {
            width: 100%;
            height: 8px;
            background: #1e293b;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 0.85rem;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2563eb, #60a5fa);
        }

        /* Pipeline */
        .pipeline {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 9px;
            margin: 1.2rem 0;
        }

        .pipeline-step {
            padding: 0.72rem 0.9rem;
            border-radius: 10px;
            background: #0d1d30;
            border: 1px solid rgba(96, 165, 250, 0.15);
            color: #dbeafe;
            font-size: 0.72rem;
            font-weight: 750;
            text-align: center;
        }

        .pipeline-arrow {
            color: #475569;
            font-weight: 800;
        }

        /* Workflow */
        .workflow {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 0.7rem;
        }

        .workflow-step {
            padding: 1rem;
            border-radius: 14px;
            background: rgba(12, 26, 43, 0.72);
            border: 1px solid var(--border);
        }

        .workflow-number {
            color: #60a5fa;
            font-size: 0.65rem;
            font-weight: 900;
            letter-spacing: 0.10em;
        }

        .workflow-title {
            color: #f8fafc;
            font-size: 0.82rem;
            font-weight: 850;
            margin: 0.35rem 0 0.2rem;
        }

        .workflow-text {
            color: #64748b;
            font-size: 0.7rem;
            line-height: 1.55;
        }

        /* Notice */
        .notice {
            padding: 1rem 1.1rem;
            border-radius: 12px;
            background: rgba(37, 99, 235, 0.07);
            border: 1px solid rgba(96, 165, 250, 0.15);
            color: #94a3b8;
            font-size: 0.76rem;
            line-height: 1.65;
        }

        /* Footer */
        .footer {
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: #475569;
            font-size: 0.68rem;
            line-height: 1.7;
        }

        /* Tables */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        /* Mobile */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.9rem 3rem;
            }

            .hero {
                padding: 1.7rem 1.35rem;
                border-radius: 18px;
            }

            .hero-title {
                font-size: 2rem;
            }

            .workflow {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_package() -> tuple[dict[str, Any], str]:
    """Load and validate the frozen production package once."""

    model_path = next(
        (path for path in MODEL_PATHS if path.is_file()),
        None,
    )

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
        raise ValueError(
            "Invalid production package: expected a dictionary."
        )

    required_keys = {
        "model",
        "vectorizer",
        "threshold",
        "config",
        "seed",
        "model_name",
        "feature_policy",
        "label_mapping",
        "metadata",
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
        raise TypeError(
            "Saved model does not support predict_proba()."
        )

    if not hasattr(vectorizer, "transform"):
        raise TypeError(
            "Saved vectorizer does not support transform()."
        )

    if not isinstance(threshold, (int, float)):
        raise TypeError("Saved threshold must be numeric.")

    if not 0 <= float(threshold) <= 1:
        raise ValueError(
            "Saved threshold must be between 0 and 1."
        )

    feature_policy = package["feature_policy"]

    if not isinstance(feature_policy, dict):
        raise TypeError("feature_policy must be a dictionary.")

    if set(feature_policy.get("used", [])) != {"title", "text"}:
        raise ValueError(
            "Production feature policy must use exactly "
            "'title' and 'text'."
        )

    label_mapping = package["label_mapping"]

    if not isinstance(label_mapping, dict):
        raise TypeError("label_mapping must be a dictionary.")

    return package, str(model_path)


# ============================================================
# Inference
# ============================================================

def normalize_text(value: Any) -> str:
    """Safely normalize an input value to stripped text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def prepare_text(title: str, text: str) -> str:
    """
    Combine only the production-approved fields.

    subject is intentionally never consumed.
    """

    title = normalize_text(title)
    text = normalize_text(text)

    parts = []

    if title:
        parts.append(title)

    if text:
        parts.append(text)

    combined = " ".join(parts).strip()

    if not combined:
        raise ValueError(
            "Please provide a headline or article body."
        )

    if len(combined) < MIN_INPUT_LENGTH:
        raise ValueError(
            f"The input is too short. Please provide at least "
            f"{MIN_INPUT_LENGTH} characters."
        )

    return combined


def predict_article(
    title: str,
    text: str,
    package: dict[str, Any],
) -> dict[str, Any]:
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
            "Inference failed while transforming the input or "
            "generating probabilities."
        ) from exc

    classes = getattr(model, "classes_", None)

    if classes is None:
        raise RuntimeError(
            "The saved model does not expose classes_."
        )

    class_probabilities = {
        int(cls): float(probability)
        for cls, probability in zip(classes, probabilities)
    }

    if 0 not in class_probabilities or 1 not in class_probabilities:
        raise RuntimeError(
            "The saved model does not contain the expected classes "
            "0 and 1."
        )

    real_probability = class_probabilities[0]
    fake_probability = class_probabilities[1]

    prediction = int(fake_probability >= threshold)

    label = label_mapping.get(
        prediction,
        "Fake News" if prediction == 1 else "Real News",
    )

    confidence = (
        fake_probability
        if prediction == 1
        else real_probability
    )

    return {
        "prediction": prediction,
        "label": label,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "confidence": confidence,
        "threshold": threshold,
    }


# ============================================================
# Batch Inference
# ============================================================

def find_column(
    dataframe: pd.DataFrame,
    target: str,
) -> Any | None:
    """Find a CSV column case-insensitively."""

    for column in dataframe.columns:
        if str(column).strip().lower() == target:
            return column

    return None


def analyze_batch(
    dataframe: pd.DataFrame,
    package: dict[str, Any],
) -> pd.DataFrame:
    """Run frozen-model inference on every CSV row."""

    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no rows.")

    title_column = find_column(dataframe, "title")
    text_column = find_column(dataframe, "text")

    if title_column is None and text_column is None:
        raise ValueError(
            "CSV must contain at least one of these columns: "
            "'title' or 'text'."
        )

    results = dataframe.copy()

    predictions = []
    labels = []
    real_probabilities = []
    fake_probabilities = []
    confidences = []

    progress = st.progress(0.0)
    status = st.empty()

    total = len(dataframe)

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        title = (
            normalize_text(row[title_column])
            if title_column is not None
            else ""
        )

        text = (
            normalize_text(row[text_column])
            if text_column is not None
            else ""
        )

        try:
            result = predict_article(
                title=title,
                text=text,
                package=package,
            )

            predictions.append(result["prediction"])
            labels.append(result["label"])
            real_probabilities.append(
                result["real_probability"]
            )
            fake_probabilities.append(
                result["fake_probability"]
            )
            confidences.append(result["confidence"])

        except Exception:
            predictions.append(None)
            labels.append("Inference Error")
            real_probabilities.append(None)
            fake_probabilities.append(None)
            confidences.append(None)

        if row_number == total or row_number % 25 == 0:
            progress.progress(row_number / total)
            status.caption(
                f"Processing {row_number:,} / {total:,} rows"
            )

    progress.progress(1.0)
    status.caption(f"Completed {total:,} rows.")

    results["prediction"] = predictions
    results["label"] = labels
    results["real_probability"] = real_probabilities
    results["fake_probability"] = fake_probabilities
    results["confidence"] = confidences

    return results


def dataframe_to_csv(dataframe: pd.DataFrame) -> bytes:
    """Convert a dataframe into UTF-8 CSV bytes."""

    return dataframe.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


def csv_template() -> bytes:
    """Return the official CSV input template."""

    dataframe = pd.DataFrame(
        [
            {
                "title": "Example headline",
                "text": "Example article body",
            }
        ]
    )

    return dataframe_to_csv(dataframe)


# ============================================================

# ============================================================
# Premium TruthLens UI
# UI/UX layer only. Production inference core remains untouched.
# ============================================================

st.markdown(
    """
    <style>
        :root {
            --bg: #060914;
            --surface: #0b1020;
            --surface-2: #10172a;
            --surface-3: #151d33;
            --border: rgba(148, 163, 184, 0.13);
            --border-strong: rgba(96, 165, 250, 0.24);
            --text: #f8fafc;
            --muted: #94a3b8;
            --muted-2: #64748b;
            --blue: #60a5fa;
            --cyan: #22d3ee;
            --violet: #8b5cf6;
            --green: #34d399;
            --red: #fb7185;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% -10%, rgba(59,130,246,.16), transparent 30%),
                radial-gradient(circle at 94% 2%, rgba(139,92,246,.12), transparent 26%),
                radial-gradient(circle at 50% 105%, rgba(34,211,238,.06), transparent 30%),
                var(--bg);
            color: var(--text);
        }

        header, footer {
            visibility: hidden;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            max-width: 1320px;
            padding: 22px 38px 65px;
        }

        /* ---------------- Navigation ---------------- */

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 13px 15px;
            margin-bottom: 25px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(8, 12, 24, .78);
            backdrop-filter: blur(22px);
            box-shadow: 0 18px 55px rgba(0,0,0,.22);
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 11px;
        }

        .brand-mark {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            color: white;
            font-weight: 950;
            font-size: 16px;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            box-shadow: 0 9px 28px rgba(59,130,246,.26);
        }

        .brand-name {
            color: #fff;
            font-size: 13px;
            font-weight: 900;
            letter-spacing: -.02em;
        }

        .brand-sub {
            margin-top: 2px;
            color: #64748b;
            font-size: 8px;
            font-weight: 800;
            letter-spacing: .13em;
            text-transform: uppercase;
        }

        .online {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            border: 1px solid rgba(52,211,153,.18);
            border-radius: 999px;
            background: rgba(16,185,129,.055);
            color: #a7f3d0;
            font-size: 8px;
            font-weight: 900;
            letter-spacing: .10em;
            text-transform: uppercase;
        }

        .online-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #34d399;
            box-shadow: 0 0 13px rgba(52,211,153,.8);
        }

        /* ---------------- Buttons ---------------- */

        div.stButton > button,
        div.stDownloadButton > button {
            min-height: 42px !important;
            border-radius: 11px !important;
            font-size: 10px !important;
            font-weight: 850 !important;
            transition: .18s ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-1px);
        }

        /* ---------------- Hero ---------------- */

        .hero {
            position: relative;
            overflow: hidden;
            padding: 42px 44px;
            border: 1px solid rgba(96,165,250,.18);
            border-radius: 28px;
            background:
                linear-gradient(125deg, rgba(13,23,43,.98), rgba(9,14,28,.95));
            box-shadow: 0 30px 90px rgba(0,0,0,.28);
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 430px;
            height: 430px;
            right: -180px;
            top: -240px;
            border-radius: 50%;
            background: rgba(59,130,246,.12);
            filter: blur(12px);
        }

        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1.35fr .65fr;
            gap: 38px;
            align-items: end;
        }

        .eyebrow {
            color: #7dd3fc;
            font-size: 8px;
            font-weight: 950;
            letter-spacing: .20em;
            text-transform: uppercase;
        }

        .hero-title {
            margin-top: 11px;
            max-width: 790px;
            color: #fff;
            font-size: clamp(36px, 5vw, 65px);
            line-height: .96;
            font-weight: 950;
            letter-spacing: -.058em;
        }

        .hero-title span {
            color: #60a5fa;
        }

        .hero-copy {
            max-width: 720px;
            margin-top: 18px;
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.75;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 22px;
        }

        .chip {
            padding: 7px 10px;
            border: 1px solid rgba(148,163,184,.12);
            border-radius: 999px;
            background: rgba(255,255,255,.035);
            color: #cbd5e1;
            font-size: 8px;
            font-weight: 850;
            letter-spacing: .04em;
        }

        .hero-status {
            padding: 19px;
            border: 1px solid rgba(148,163,184,.11);
            border-radius: 19px;
            background: rgba(255,255,255,.035);
        }

        .status-label {
            color: #64748b;
            font-size: 8px;
            font-weight: 900;
            letter-spacing: .13em;
            text-transform: uppercase;
        }

        .status-value {
            margin-top: 6px;
            color: #fff;
            font-size: 27px;
            font-weight: 950;
            letter-spacing: -.035em;
        }

        .status-copy {
            margin-top: 7px;
            color: #64748b;
            font-size: 9px;
            line-height: 1.55;
        }

        /* ---------------- Sections ---------------- */

        .section {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 25px;
            margin: 31px 0 14px;
        }

        .section-kicker {
            color: #64748b;
            font-size: 8px;
            font-weight: 950;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .section-title {
            margin-top: 5px;
            color: #f8fafc;
            font-size: 21px;
            font-weight: 900;
            letter-spacing: -.03em;
        }

        .section-copy {
            max-width: 500px;
            color: #64748b;
            font-size: 9px;
            line-height: 1.55;
            text-align: right;
        }

        /* ---------------- Workspace ---------------- */

        .workspace {
            padding: 23px;
            border: 1px solid var(--border);
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(15,23,42,.88), rgba(8,13,25,.93));
            box-shadow: 0 18px 55px rgba(0,0,0,.18);
        }

        .workspace-label {
            color: #7dd3fc;
            font-size: 8px;
            font-weight: 950;
            letter-spacing: .15em;
            text-transform: uppercase;
        }

        .workspace-title {
            margin-top: 6px;
            color: #f8fafc;
            font-size: 19px;
            font-weight: 900;
            letter-spacing: -.025em;
        }

        .workspace-copy {
            margin-top: 5px;
            color: #64748b;
            font-size: 9px;
            line-height: 1.6;
        }

        .feature-note,
        .notice {
            margin-top: 11px;
            padding: 11px 13px;
            border: 1px solid rgba(96,165,250,.13);
            border-radius: 13px;
            background: rgba(59,130,246,.045);
            color: #64748b;
            font-size: 9px;
            line-height: 1.55;
        }

        .feature-note strong,
        .notice strong {
            color: #bfdbfe;
        }

        /* ---------------- Inputs ---------------- */

        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label,
        [data-testid="stFileUploader"] label {
            color: #cbd5e1 !important;
            font-size: 9px !important;
            font-weight: 850 !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            color: #f8fafc !important;
            background: #080f1d !important;
            border: 1px solid rgba(148,163,184,.15) !important;
            border-radius: 13px !important;
            font-size: 11px !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: rgba(96,165,250,.60) !important;
            box-shadow: 0 0 0 1px rgba(96,165,250,.20) !important;
        }

        /* ---------------- Intelligence rail ---------------- */

        .rail {
            padding: 16px;
            margin-bottom: 9px;
            border: 1px solid var(--border);
            border-radius: 17px;
            background: rgba(15,23,42,.72);
        }

        .rail-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }

        .rail-label {
            color: #64748b;
            font-size: 8px;
            font-weight: 850;
            letter-spacing: .11em;
            text-transform: uppercase;
        }

        .rail-icon {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 9px;
            color: #60a5fa;
            background: rgba(96,165,250,.08);
            font-size: 10px;
            font-weight: 950;
        }

        .rail-value {
            margin-top: 6px;
            color: #f8fafc;
            font-size: 17px;
            font-weight: 900;
        }

        .rail-copy {
            margin-top: 4px;
            color: #64748b;
            font-size: 8px;
            line-height: 1.5;
        }

        /* ---------------- Verdict ---------------- */

        .verdict {
            padding: 21px;
            margin-top: 15px;
            border: 1px solid var(--border);
            border-radius: 20px;
        }

        .verdict-real {
            background: linear-gradient(135deg, rgba(6,78,59,.32), rgba(15,23,42,.92));
            border-color: rgba(52,211,153,.20);
        }

        .verdict-fake {
            background: linear-gradient(135deg, rgba(127,29,29,.30), rgba(15,23,42,.92));
            border-color: rgba(251,113,133,.20);
        }

        .verdict-kicker {
            color: #94a3b8;
            font-size: 8px;
            font-weight: 900;
            letter-spacing: .15em;
            text-transform: uppercase;
        }

        .verdict-title {
            margin-top: 5px;
            color: #fff;
            font-size: 29px;
            font-weight: 950;
            letter-spacing: -.045em;
        }

        .verdict-copy {
            margin-top: 5px;
            color: #94a3b8;
            font-size: 9px;
            line-height: 1.55;
        }

        .prob-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 15px;
        }

        .prob-box {
            padding: 12px;
            border: 1px solid rgba(148,163,184,.09);
            border-radius: 13px;
            background: rgba(0,0,0,.12);
        }

        .prob-label {
            color: #64748b;
            font-size: 8px;
        }

        .prob-value {
            margin-top: 4px;
            color: #f8fafc;
            font-size: 17px;
            font-weight: 900;
        }

        .confidence {
            padding: 14px 16px;
            margin-top: 9px;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(15,23,42,.72);
        }

        .confidence-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }

        .confidence-label {
            color: #94a3b8;
            font-size: 8px;
            font-weight: 800;
        }

        .confidence-value {
            color: #fff;
            font-size: 16px;
            font-weight: 900;
        }

        .track {
            height: 5px;
            margin-top: 9px;
            overflow: hidden;
            border-radius: 999px;
            background: #1e293b;
        }

        .fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #3b82f6, #22d3ee);
        }

        /* ---------------- Metrics ---------------- */

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 9px;
            margin-top: 9px;
        }

        .metric {
            padding: 14px;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(15,23,42,.70);
        }

        .metric-label {
            color: #64748b;
            font-size: 8px;
            font-weight: 850;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 6px;
            color: #f8fafc;
            font-size: 16px;
            font-weight: 900;
            letter-spacing: -.02em;
        }

        .metric-note {
            margin-top: 3px;
            color: #475569;
            font-size: 8px;
        }

        /* ---------------- Empty state ---------------- */

        .empty {
            min-height: 255px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px dashed rgba(148,163,184,.17);
            border-radius: 18px;
            background: rgba(3,8,18,.30);
        }

        .empty-icon {
            width: 52px;
            height: 52px;
            margin: 0 auto 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            border: 1px solid rgba(96,165,250,.13);
            background: linear-gradient(135deg, rgba(59,130,246,.10), rgba(139,92,246,.10));
            color: #93c5fd;
            font-size: 21px;
            font-weight: 950;
        }

        .empty-title {
            color: #e2e8f0;
            font-size: 12px;
            font-weight: 850;
        }

        .empty-copy {
            max-width: 255px;
            margin: 6px auto 0;
            color: #64748b;
            font-size: 9px;
            line-height: 1.55;
        }

        /* ---------------- Pipeline ---------------- */

        .pipeline {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 7px;
        }

        .pipeline-item {
            min-height: 88px;
            padding: 13px;
            border: 1px solid var(--border);
            border-radius: 15px;
            background: rgba(15,23,42,.68);
        }

        .pipeline-num {
            color: #475569;
            font-size: 7px;
            font-weight: 900;
            letter-spacing: .11em;
        }

        .pipeline-name {
            margin-top: 18px;
            color: #cbd5e1;
            font-size: 9px;
            line-height: 1.4;
            font-weight: 850;
        }

        /* ---------------- Batch / Architecture ---------------- */

        .dropzone {
            padding: 42px 20px;
            text-align: center;
            border: 1px dashed rgba(96,165,250,.23);
            border-radius: 20px;
            background: rgba(59,130,246,.025);
        }

        .drop-icon {
            width: 52px;
            height: 52px;
            margin: 0 auto 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            color: #60a5fa;
            background: rgba(96,165,250,.08);
            font-size: 21px;
            font-weight: 950;
        }

        .drop-title {
            color: #e2e8f0;
            font-size: 12px;
            font-weight: 850;
        }

        .drop-copy {
            margin-top: 5px;
            color: #64748b;
            font-size: 9px;
        }

        .architecture {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 7px;
        }

        .arch-node {
            min-height: 95px;
            padding: 15px;
            border: 1px solid var(--border);
            border-radius: 15px;
            background: rgba(15,23,42,.70);
        }

        .arch-num {
            color: #60a5fa;
            font-size: 7px;
            font-weight: 950;
        }

        .arch-title {
            margin-top: 10px;
            color: #e2e8f0;
            font-size: 9px;
            font-weight: 850;
            line-height: 1.45;
        }

        [data-testid="stExpander"] {
            margin-top: 9px;
            border: 1px solid var(--border) !important;
            border-radius: 15px !important;
            background: rgba(15,23,42,.62) !important;
        }

        [data-testid="stExpander"] summary p {
            color: #cbd5e1 !important;
            font-size: 9px !important;
            font-weight: 850 !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 13px;
            overflow: hidden;
        }

        .footer {
            margin-top: 46px;
            padding-top: 18px;
            border-top: 1px solid var(--border);
            text-align: center;
            color: #475569;
            font-size: 8px;
            line-height: 1.7;
        }

        .footer strong {
            color: #94a3b8;
        }

        @media (max-width: 900px) {
            .block-container { padding: 16px 14px 50px; }
            .topbar { align-items: flex-start; flex-direction: column; }
            .hero { padding: 28px 22px; }
            .hero-grid { grid-template-columns: 1fr; }
            .hero-title { font-size: 41px; }
            .section { align-items: flex-start; flex-direction: column; }
            .section-copy { text-align: left; }
            .pipeline { grid-template-columns: repeat(2, 1fr); }
            .architecture { grid-template-columns: repeat(2, 1fr); }
            .metric-grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_brand_nav() -> str:
    """Render application navigation without changing inference behavior."""
    st.markdown(
        """
        <div class="topbar">
            <div class="brand-wrap">
                <div class="brand-mark">T</div>
                <div>
                    <div class="brand-name">TRUTHLENS AI</div>
                    <div class="brand-sub">Misinformation Intelligence</div>
                </div>
            </div>
            <div class="online">
                <span class="online-dot"></span>
                Production inference online
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Analyze News"

    cols = st.columns([1.0, 1.0, 1.0, 1.55], gap="small")
    pages = [
        ("Analyze News", "01 · Analyze"),
        ("Batch Analysis", "02 · Batch"),
        ("Model Overview", "03 · Model"),
    ]

    for col, (page_name, label) in zip(cols[:3], pages):
        with col:
            if st.button(
                label,
                key=f"nav_{page_name.lower().replace(' ', '_')}",
                use_container_width=True,
                type="primary" if st.session_state.active_page == page_name else "secondary",
            ):
                st.session_state.active_page = page_name
                st.rerun()

    with cols[3]:
        st.markdown(
            f"""
            <div style="height:42px;display:flex;align-items:center;justify-content:flex-end;
                        color:#64748b;font-size:8px;font-weight:850;">
                v{APP_VERSION} &nbsp;·&nbsp; INFERENCE ONLY &nbsp;·&nbsp; TITLE + TEXT
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.active_page


def render_page_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section">
            <div>
                <div class="section-kicker">{kicker}</div>
                <div class="section-title">{title}</div>
            </div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(package: dict[str, Any]) -> None:
    threshold = float(package["threshold"])
    model_name = str(package.get("model_name", "Calibrated Linear SVM"))

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">AI-powered misinformation detection</div>
                    <div class="hero-title">
                        Read the signal.<br><span>Challenge the story.</span>
                    </div>
                    <div class="hero-copy">
                        A production-grade interface for news classification using the
                        frozen TF-IDF + Linear SVM inference pipeline. Every verdict is
                        generated by the saved model package — no retraining, tuning,
                        or simulated predictions.
                    </div>
                    <div class="chips">
                        <span class="chip">TITLE + TEXT</span>
                        <span class="chip">TF-IDF</span>
                        <span class="chip">LINEAR SVM</span>
                        <span class="chip">PREDICT_PROBA</span>
                        <span class="chip">THRESHOLD {threshold:.3f}</span>
                    </div>
                </div>
                <div class="hero-status">
                    <div class="status-label">Production engine</div>
                    <div class="status-value">READY</div>
                    <div class="status-copy">
                        {model_name}<br>
                        Saved vectorizer · saved threshold<br>
                        Inference-only execution
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, Any]) -> None:
    # IMPORTANT:
    # These keys match the real production predict_article() contract.
    label = str(result["label"])
    is_real = label.strip().lower() == "real news"

    verdict = "REAL NEWS" if is_real else "FAKE NEWS"
    verdict_class = "verdict-real" if is_real else "verdict-fake"

    real_probability = float(result["real_probability"])
    fake_probability = float(result["fake_probability"])
    confidence = float(result["confidence"])
    threshold = float(result["threshold"])

    description = (
        "The production classifier placed this article on the real-news side "
        "of the saved decision rule."
        if is_real
        else
        "The production classifier placed this article on the fake-news side "
        "of the saved decision rule."
    )

    st.markdown(
        f"""
        <div class="verdict {verdict_class}">
            <div class="verdict-kicker">Production verdict</div>
            <div class="verdict-title">{verdict}</div>
            <div class="verdict-copy">{description}</div>
            <div class="prob-grid">
                <div class="prob-box">
                    <div class="prob-label">Real probability</div>
                    <div class="prob-value">{real_probability:.2%}</div>
                </div>
                <div class="prob-box">
                    <div class="prob-label">Fake probability</div>
                    <div class="prob-value">{fake_probability:.2%}</div>
                </div>
            </div>
        </div>
        <div class="confidence">
            <div class="confidence-row">
                <div class="confidence-label">Model confidence · selected class probability</div>
                <div class="confidence-value">{confidence:.2%}</div>
            </div>
            <div class="track">
                <div class="fill" style="width:{max(0, min(100, confidence * 100)):.1f}%;"></div>
            </div>
        </div>
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Predicted class</div>
                <div class="metric-value">{result["prediction"]}</div>
                <div class="metric-note">Saved class decision</div>
            </div>
            <div class="metric">
                <div class="metric-label">Decision threshold</div>
                <div class="metric-value">{threshold:.3f}</div>
                <div class="metric-note">Frozen production rule</div>
            </div>
            <div class="metric">
                <div class="metric-label">Features used</div>
                <div class="metric-value">Title + Text</div>
                <div class="metric-note">Subject excluded</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical inference details"):
        render_technical_details(result)


def render_technical_details(result: dict[str, Any]) -> None:
    model_name = str(st.session_state.get("_model_name", "Saved production model"))

    details = {
        "Model": model_name,
        "Vectorizer": str(st.session_state.get("_vectorizer_name", "Saved TF-IDF vectorizer")),
        "Threshold": result.get("threshold"),
        "Input features": "title + text",
        "Excluded features": "subject",
        "Predicted class": result.get("prediction"),
        "Predicted label": result.get("label"),
    }

    st.dataframe(
        pd.DataFrame(
            [{"Parameter": key, "Value": value} for key, value in details.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="notice">
            <strong>Responsible AI.</strong>
            This is an ML-assisted classification system, not definitive proof
            that a claim is true or false. Important information should be
            independently verified through reliable and authoritative sources.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline() -> None:
    steps = [
        "Title + Text",
        "Text combination",
        "Saved TF-IDF",
        "Linear SVM",
        "predict_proba()",
        "Saved threshold",
        "Real / Fake",
    ]

    items = []
    for index, step in enumerate(steps, 1):
        items.append(
            f'<div class="pipeline-item">'
            f'<div class="pipeline-num">STEP {index:02d}</div>'
            f'<div class="pipeline-name">{step}</div>'
            f'</div>'
        )

    html = '<div class="pipeline">' + ''.join(items) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_analyze_page(package: dict[str, Any]) -> None:
    render_hero(package)

    render_page_header(
        "01 · Verification workspace",
        "Analyze a news article",
        "Provide the headline and body, then run the real production inference pipeline.",
    )

    left, right = st.columns([1.12, .88], gap="large")

    with left:
        st.markdown(
            """
            <div class="workspace">
                <div class="workspace-label">Article input</div>
                <div class="workspace-title">What are we checking?</div>
                <div class="workspace-copy">
                    The saved production policy combines only the headline and
                    article body before vectorization and classification.
                </div>
            """,
            unsafe_allow_html=True,
        )

        title = st.text_input(
            "News headline",
            placeholder="Enter the full article headline…",
            key="analyze_title",
        )

        text = st.text_area(
            "Article body",
            placeholder="Paste the full news article text here…",
            height=270,
            key="analyze_text",
        )

        st.markdown(
            """
            <div class="feature-note">
                <strong>Production feature policy:</strong>
                title + text are used for inference.
                The <strong>subject</strong> field is intentionally excluded.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.button(
            "Run verification  →",
            type="primary",
            use_container_width=True,
            key="run_single_prediction",
        ):
            try:
                # Exact original production function signature.
                result = predict_article(
                    title=title,
                    text=text,
                    package=package,
                )
                st.session_state["latest_prediction"] = result
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with right:
        st.markdown(
            """
            <div class="rail">
                <div class="rail-head">
                    <div class="rail-label">Model status</div>
                    <div class="rail-icon">✓</div>
                </div>
                <div class="rail-value">Production Ready</div>
                <div class="rail-copy">Saved model package loaded for inference.</div>
            </div>

            <div class="rail">
                <div class="rail-head">
                    <div class="rail-label">Input policy</div>
                    <div class="rail-icon">01</div>
                </div>
                <div class="rail-value">Title + Text</div>
                <div class="rail-copy">Subject is not part of production inference.</div>
            </div>

            <div class="rail">
                <div class="rail-head">
                    <div class="rail-label">Decision engine</div>
                    <div class="rail-icon">AI</div>
                </div>
                <div class="rail-value">TF-IDF + SVM</div>
                <div class="rail-copy">predict_proba() followed by the saved threshold.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result = st.session_state.get("latest_prediction")

        if result:
            render_result(result)
        else:
            st.markdown(
                """
                <div class="empty">
                    <div>
                        <div class="empty-icon">⌁</div>
                        <div class="empty-title">Awaiting an article</div>
                        <div class="empty-copy">
                            Add a headline and article body, then run verification
                            to reveal the production verdict.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_page_header(
        "02 · Transparent inference",
        "How the verdict is produced",
        "The interface visualizes the frozen production path without modifying model behavior.",
    )

    render_pipeline()
    render_disclaimer()


def render_batch_page(package: dict[str, Any]) -> None:
    render_page_header(
        "Batch intelligence",
        "Analyze an entire news dataset",
        "Apply the exact same saved model, vectorizer, probabilities, and threshold to every row.",
    )

    st.markdown(
        """
        <div class="hero" style="padding:29px 32px;">
            <div class="hero-grid" style="grid-template-columns:1.15fr .85fr;">
                <div>
                    <div class="eyebrow">Production batch mode</div>
                    <div class="hero-title" style="font-size:37px;">
                        From one article<br><span>to a full feed.</span>
                    </div>
                    <div class="hero-copy" style="margin-top:13px;">
                        Process CSV inputs with the same frozen inference pipeline
                        used by the single-article analyzer.
                    </div>
                </div>
                <div class="hero-status">
                    <div class="status-label">Expected schema</div>
                    <div class="status-value" style="font-size:22px;">title + text</div>
                    <div class="status-copy">
                        Subject is not used by the production pipeline.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns([.44, .56], gap="large")

    with a:
        st.markdown(
            """
            <div class="workspace">
                <div class="workspace-label">Quick start</div>
                <div class="workspace-title">Use the CSV template</div>
                <div class="workspace-copy">
                    Download the expected structure and replace the example row
                    with your own data.
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "Download CSV template",
            data=csv_template(),
            file_name="misinformation_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown(
            """
            <div class="workspace">
                <div class="workspace-label">Schema contract</div>
                <div class="metric-grid" style="margin-top:14px;">
                    <div class="metric">
                        <div class="metric-label">Required</div>
                        <div class="metric-value">title</div>
                        <div class="metric-note">Article headline</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Required</div>
                        <div class="metric-value">text</div>
                        <div class="metric-note">Article body</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Excluded</div>
                        <div class="metric-value">subject</div>
                        <div class="metric-note">Not used in inference</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_page_header(
        "Upload",
        "Dataset ingestion",
        "Preview the data before executing production inference.",
    )

    uploaded_file = st.file_uploader(
        "CSV file",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="dropzone">
                <div class="drop-icon">↑</div>
                <div class="drop-title">Upload your CSV dataset</div>
                <div class="drop-copy">Required columns: title and text</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_disclaimer()
        return

    try:
        dataframe = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Rows</div>
                <div class="metric-value">{len(dataframe):,}</div>
                <div class="metric-note">Uploaded records</div>
            </div>
            <div class="metric">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{len(dataframe.columns):,}</div>
                <div class="metric-note">Detected fields</div>
            </div>
            <div class="metric">
                <div class="metric-label">Threshold</div>
                <div class="metric-value">{float(package["threshold"]):.3f}</div>
                <div class="metric-note">Saved production rule</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Preview uploaded data", expanded=True):
        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "Run batch verification  →",
        type="primary",
        use_container_width=True,
        key="run_batch_prediction",
    ):
        with st.spinner("Running production inference across the dataset…"):
            try:
                # Exact original production function signature.
                results_df = analyze_batch(
                    dataframe=dataframe,
                    package=package,
                )
                st.session_state["batch_results"] = results_df
            except Exception as exc:
                st.error(f"Batch analysis failed: {exc}")

    results_df = st.session_state.get("batch_results")

    if results_df is not None:
        render_page_header(
            "Results",
            "Batch intelligence report",
            "Predictions returned directly from the saved production model package.",
        )

        label_series = (
            results_df["label"]
            if "label" in results_df.columns
            else pd.Series(dtype=str)
        )

        real_count = int(
            (label_series.astype(str).str.lower() == "real news").sum()
        )
        fake_count = int(
            (label_series.astype(str).str.lower() == "fake news").sum()
        )
        error_count = int(
            label_series.astype(str).str.lower().str.contains("error").sum()
        )

        cols = st.columns(4)
        values = [
            ("Processed", f"{len(results_df):,}", "Rows returned"),
            ("Real News", f"{real_count:,}", "Production verdict"),
            ("Fake News", f"{fake_count:,}", "Production verdict"),
            ("Errors", f"{error_count:,}", "Rows requiring review"),
        ]

        for col, (label, value, note) in zip(cols, values):
            with col:
                render_metric_card(label, value, note)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
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
    model_name = str(package.get("model_name", "Saved production model"))
    vectorizer_name = type(package["vectorizer"]).__name__

    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid" style="grid-template-columns:1fr;">
                <div>
                    <div class="eyebrow">Frozen production architecture</div>
                    <div class="hero-title" style="font-size:40px;">
                        One inference path.<br><span>No surprises.</span>
                    </div>
                    <div class="hero-copy" style="max-width:850px;">
                        The application is an inference layer over a saved model package.
                        It does not retrain, retune, replace, or simulate the production
                        classifier.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    architecture = [
        ("01", "Title + Text"),
        ("02", "Text combination"),
        ("03", "Saved TF-IDF"),
        ("04", "Linear SVM"),
        ("05", "predict_proba()"),
        ("06", "Saved threshold → verdict"),
    ]

    nodes = []
    for number, title in architecture:
        nodes.append(
            f'<div class="arch-node">'
            f'<div class="arch-num">{number}</div>'
            f'<div class="arch-title">{title}</div>'
            f'</div>'
        )

    html = '<div class="architecture">' + ''.join(nodes) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = [
        ("Classifier", model_name, "Saved package"),
        ("Vectorizer", vectorizer_name, "Saved package"),
        ("Threshold", f"{threshold:.3f}", "Frozen decision rule"),
        ("Mode", "Inference only", "No training or tuning"),
    ]

    for col, (label, value, note) in zip(cols, metrics):
        with col:
            render_metric_card(label, value, note)

    render_page_header(
        "Production contract",
        "Configuration and feature policy",
        "These values describe the exact inference contract used by the application.",
    )

    config_rows = [
        ("Model package", "misinformation_model_package.joblib"),
        ("Inference features", "title + text"),
        ("Excluded feature", "subject"),
        ("Decision method", "predict_proba() + saved threshold"),
        ("Execution mode", "inference only"),
        ("Application version", APP_VERSION),
    ]

    for key, value in config_rows:
        st.markdown(
            f"""
            <div class="rail" style="display:flex;justify-content:space-between;
                        align-items:center;gap:20px;">
                <div class="rail-label">{key}</div>
                <div style="color:#e2e8f0;font-size:9px;font-weight:850;text-align:right;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    config = package.get("config", {})
    if config:
        with st.expander("View saved configuration"):
            st.json(config)

    with st.expander("View package metadata"):
        metadata = package.get("metadata", {})
        if metadata:
            st.json(metadata)
        else:
            st.info("No additional metadata is exposed by the production package.")

    render_disclaimer()


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Application entry point."""
    try:
        package, model_path = load_model_package()
    except FileNotFoundError as exc:
        st.error("Production Model Not Found")
        st.code(str(exc))
        st.info(
            "Place misinformation_model_package.joblib inside "
            "the models/ directory or project root."
        )
        st.stop()
    except (ValueError, TypeError, RuntimeError) as exc:
        st.error("Production Model Error")
        st.code(str(exc))
        st.stop()
    except Exception as exc:
        # Show the actual exception during debugging instead of hiding
        # the root cause behind a generic message.
        st.error("Unexpected production model loading error")
        st.code(f"{type(exc).__name__}: {exc}")
        st.stop()

    # Visual-only metadata; does not affect inference.
    st.session_state["_model_name"] = str(
        package.get("model_name", "Saved production model")
    )
    st.session_state["_vectorizer_name"] = type(
        package["vectorizer"]
    ).__name__

    page = render_brand_nav()

    if page == "Analyze News":
        render_analyze_page(package)
    elif page == "Batch Analysis":
        render_batch_page(package)
    else:
        render_model_overview(package)

    st.markdown(
        f"""
        <div class="footer">
            <strong>TRUTHLENS AI · {APP_NAME}</strong><br>
            Production NLP · Real Inference · Responsible AI<br>
            Frozen model package: {Path(model_path).name}
            &nbsp;•&nbsp; Version {APP_VERSION}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
