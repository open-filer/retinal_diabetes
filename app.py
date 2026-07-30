"""
Multi-Model AI Prediction System -- Diabetic Retinopathy Detection
Final UI: model picker (any combination of 9 trained models), live prediction
with ensemble verdict, full results dashboard, and PDF report export.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import ensemble_utils as eu
import pdf_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "general_risk"))
import general_risk_utils as gru

import math
import uuid


def render_confidence_ring(probability, color, size=150, stroke=12):
    """Returns an HTML string for an animated SVG ring that fills to the
    given probability (0-1), using a per-instance keyframe so multiple rings
    on the same page animate to their own correct values independently."""
    uid = uuid.uuid4().hex[:8]
    r = (size - stroke) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    target_offset = circumference * (1 - probability)
    percent_text = f"{probability*100:.0f}%"

    return f"""
    <style>
    @keyframes ring-fill-{uid} {{
        from {{ stroke-dashoffset: {circumference:.2f}; }}
        to {{ stroke-dashoffset: {target_offset:.2f}; }}
    }}
    .ring-{uid} {{ animation: ring-fill-{uid} 1.3s cubic-bezier(0.22, 1, 0.36, 1) forwards; }}
    </style>
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(232,163,61,0.15)" stroke-width="{stroke}"/>
      <circle class="ring-{uid}" cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}"
        stroke-width="{stroke}" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{circumference:.2f}"
        stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy+8}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
        font-size="28" font-weight="600" fill="#F5E6D3">{percent_text}</text>
    </svg>
    """

st.set_page_config(
    page_title="Multi-Model AI Prediction System",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Fundus-themed design system: fonts, color tokens, hero banner, restyled
# components. Palette and motif are drawn directly from real retinal fundus
# photography (dark surround, warm amber/red circular field, branching
# vessels) rather than a generic medical blue/white dashboard look.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
  --fundus-black: #0B0705;
  --fundus-deep: #2A0E08;
  --fundus-panel: #1A0F0C;
  --fundus-vessel: #C0392B;
  --fundus-amber: #E8A33D;
  --fundus-teal: #2D9C8F;
  --fundus-pale: #F5E6D3;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* --- Hero banner: REAL video background (eye-scan footage) --- */
.fundus-hero {
  position: relative;
  border-radius: 18px;
  width: 100%;
  height: 390px;
  margin-bottom: 8px;
  overflow: hidden;
  border: 1px solid rgba(232,163,61,0.18);
  background: #000;
}
.fundus-hero video {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain; object-position: center; z-index: 0; opacity: 0.9;
}
.fundus-hero .hero-overlay {
  position: absolute; inset: 0; z-index: 1;
  background: linear-gradient(100deg, rgba(11,7,5,0.97) 0%, rgba(11,7,5,0.88) 32%, rgba(11,7,5,0.15) 68%, rgba(11,7,5,0.35) 100%);
}
.fundus-hero .hero-content { position: relative; z-index: 2; padding: 38px 36px; max-width: 600px; }

.fundus-hero h1 {
  font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.3rem;
  color: var(--fundus-pale); margin: 0; position: relative; z-index: 2;
  letter-spacing: -0.01em;
}
.fundus-hero p.tagline {
  color: #d8c3ad; font-size: 1.02rem; margin-top: 8px; position: relative; z-index: 2;
  max-width: 560px;
}
.fundus-hero .eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
  color: var(--fundus-amber); text-transform: uppercase; margin-bottom: 10px;
  position: relative; z-index: 2;
}

/* --- Dashboard card entrance animation (staggered fade + rise, one-time) --- */
.readout-card {
  background: var(--fundus-panel); border: 1px solid rgba(232,163,61,0.15);
  border-radius: 12px; padding: 16px 18px; transition: transform 0.15s ease, border-color 0.15s ease;
  animation: card-rise 0.55s ease-out backwards;
}
.readout-card:hover { transform: translateY(-2px); border-color: rgba(232,163,61,0.4); }
@keyframes card-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }

/* --- Custom "scanning" loader, used in place of the generic spinner --- */
.scan-loader-wrap { display: flex; align-items: center; gap: 16px; padding: 18px 4px; }
.scan-loader-ring {
  width: 46px; height: 46px; border-radius: 50%;
  border: 3px solid rgba(232,163,61,0.15);
  border-top-color: var(--fundus-amber);
  animation: scan-spin 0.9s linear infinite;
  position: relative; flex-shrink: 0;
}
.scan-loader-ring::after {
  content: ""; position: absolute; inset: 6px; border-radius: 50%;
  border: 2px dashed rgba(45,156,143,0.5);
  animation: scan-spin 2.4s linear infinite reverse;
}
@keyframes scan-spin { to { transform: rotate(360deg); } }
.scan-loader-text { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #d8c3ad; }

/* --- Dashboard video banner --- */
/* --- Reusable video banner (Dashboard + General Screening) --- */
.video-banner {
  position: relative; border-radius: 14px; overflow: hidden; margin-bottom: 18px;
  border: 1px solid rgba(232,163,61,0.18); background: #000;
  width: 100%; height: 260px;
}
.video-banner video {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  object-fit: contain; opacity: 0.95;
}
.video-banner .banner-fade {
  position: absolute; bottom: 0; left: 0; right: 0; height: 25%;
  background: linear-gradient(to top, rgba(11,7,5,0.7), rgba(11,7,5,0));
}



.readout-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
  color: #b89a78; text-transform: uppercase;
}
.readout-value {
  font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.5rem;
  color: var(--fundus-pale); margin-top: 4px;
}
.readout-delta { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--fundus-teal); margin-top: 2px; }

/* --- Verdict boxes, recolored to the fundus palette --- */
.verdict-box-dr { background-color: rgba(192,57,43,0.12); border-left: 6px solid var(--fundus-vessel); padding: 16px; border-radius: 8px; }
.verdict-box-nodr { background-color: rgba(45,156,143,0.12); border-left: 6px solid var(--fundus-teal); padding: 16px; border-radius: 8px; }

/* --- Model chips --- */
.model-chip { display: inline-block; background-color: rgba(232,163,61,0.12); border: 1px solid rgba(232,163,61,0.25); border-radius: 14px; padding: 2px 10px; font-size: 0.78em; margin-right: 6px; color: var(--fundus-pale); }
.framework-keras { border-color: rgba(232,163,61,0.4); }
.framework-pytorch { border-color: rgba(45,156,143,0.4); }
</style>
""", unsafe_allow_html=True)

REGISTRY = eu.MODEL_REGISTRY
ALL_MODEL_NAMES = list(REGISTRY.keys())

# ---------------------------------------------------------------------------
# Sidebar assistant -- a curated guide to every feature, presented as a
# friendly help panel (no external API needed, so it's instant and free).
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Project Assistant")
    st.caption("Hi! I can walk you through what each part of this app does.")

    topic = st.selectbox(
        "What would you like help with?",
        [
            "👋 Quick overview",
            "🔍 How do I run a prediction?",
            "👥 What's General Screening?",
            "📊 What's on the Dashboard?",
            "📷 How does camera capture work?",
            "⬇️ How do I get a PDF report?",
            "🧠 Which models should I pick?",
        ],
    )

    HELP_TEXT = {
        "👋 Quick overview": (
            "This system has **two tiers**:\n\n"
            "1. **Predict tab** — specialist-grade diabetic retinopathy detection "
            "from retina images, using up to 9 deep learning models.\n"
            "2. **General Screening tab** — a simple risk check anyone can use "
            "with everyday health info, no special equipment needed.\n\n"
            "Check the **Dashboard** to see how every model performed."
        ),
        "🔍 How do I run a prediction?": (
            "1. Go to the **Predict** tab\n"
            "2. Pick a model preset (or customize your own combination)\n"
            "3. Upload or capture a retina image\n"
            "4. Click **Run Prediction**\n"
            "5. See the verdict, per-model breakdown, and download a PDF if you want"
        ),
        "👥 What's General Screening?": (
            "A separate, simpler tool for people **without** access to a retinal "
            "camera. You enter basic health info (age, BMI, glucose if known, etc.) "
            "and it estimates general diabetes risk — not diabetic retinopathy "
            "specifically. Think of it as a first-step screening, not a replacement "
            "for the retina-based models."
        ),
        "📊 What's on the Dashboard?": (
            "A full comparison of all **9 trained models** — accuracy, AUC, and "
            "architecture details — so you (or anyone reviewing this project) can "
            "see exactly how each model performed, not just the ones currently selected."
        ),
        "📷 How does camera capture work?": (
            "It opens your device's camera directly in the app. **Important:** it "
            "still needs an actual fundus/retina image as input — e.g. from a "
            "smartphone fundus camera attachment, or by photographing an existing "
            "fundus image. A regular eye selfie won't work; the camera tab is just "
            "a faster way to get a real fundus image in, not a converter."
        ),
        "⬇️ How do I get a PDF report?": (
            "After running a prediction, a **Download PDF Report** button appears "
            "below the results. It includes the ensemble verdict, every model's "
            "individual prediction, and a short methodology note."
        ),
        "🧠 Which models should I pick?": (
            "- **Recommended 5**: best balance of accuracy + architectural diversity\n"
            "- **Top 3 by Accuracy**: highest raw performance\n"
            "- **All 9**: most thorough, slowest\n"
            "- **Fastest Single Model**: quick single-model check\n\n"
            "All 9 models individually exceeded 90% accuracy, so any combination "
            "is reasonable — the presets just optimize for different priorities."
        ),
    }
    st.info(HELP_TEXT[topic])

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_available_models():
    return eu.available_models()


@st.cache_resource(show_spinner=False)
def get_all_results():
    return eu.load_all_results()


@st.cache_resource(show_spinner=False)
def load_models_cached(model_names_tuple):
    """Cached so re-running with the same model selection doesn't reload from disk."""
    return {name: eu.load_model(name) for name in model_names_tuple}


# ---------------------------------------------------------------------------
# Header (hero banner already rendered above)
# ---------------------------------------------------------------------------
tab_predict, tab_general, tab_dashboard, tab_about = st.tabs(
    ["🔍 Predict (Specialist)", "👥 General Screening", "📊 Dashboard", "ℹ️ About"]
)

# ===========================================================================
# TAB 1: PREDICT
# ===========================================================================
with tab_predict:
    st.markdown(f"""
    <div class="fundus-hero">
      <video autoplay loop muted playsinline>
        <source src="app/static/hero_scan.mp4" type="video/mp4">
      </video>
      <div class="hero-overlay"></div>
      <div class="hero-content">
        <div class="eyebrow">RETINAL AI DIAGNOSTICS</div>
        <h1>🩺 Multi-Model AI Prediction System</h1>
        <p class="tagline">Diabetic Retinopathy Detection from retina fundus images, using a configurable ensemble of 9 independently trained models.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    available = get_available_models()
    missing = [m for m in ALL_MODEL_NAMES if m not in available]

    if missing:
        st.warning(
            f"**{len(missing)} model file(s) not found** in the `models/` folder: "
            f"{', '.join(missing)}. Those models won't appear as selectable below. "
            f"Place their `.keras`/`.pt` files in `models/` to enable them.",
            icon="⚠️",
        )

    st.subheader("1. Choose your models")

    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    if "model_multiselect" not in st.session_state:
        st.session_state.model_multiselect = [m for m in eu.DEFAULT_ENSEMBLE if m in available]

    def set_preset(names):
        # Must write to the SAME key the multiselect widget uses below.
        # Writing to a different session_state variable (and passing it only
        # as `default=`) doesn't work -- `default` is only applied on the
        # widget's first-ever render, not on reruns, which was the bug.
        st.session_state.model_multiselect = [n for n in names if n in available]

    with preset_col1:
        if st.button("⭐ Recommended 5", use_container_width=True):
            set_preset(eu.DEFAULT_ENSEMBLE)
    with preset_col2:
        top3 = sorted(available, key=lambda n: REGISTRY[n]["accuracy"], reverse=True)[:3]
        if st.button("🥇 Top 3 by Accuracy", use_container_width=True):
            set_preset(top3)
    with preset_col3:
        if st.button("🌐 Use All 9", use_container_width=True):
            set_preset(ALL_MODEL_NAMES)
    with preset_col4:
        if st.button("🎯 Fastest Single Model", use_container_width=True):
            fastest = min(available, key=lambda n: REGISTRY[n]["params_millions"]) if available else None
            set_preset([fastest] if fastest else [])

    selected_models = st.multiselect(
        "Or customize the exact model combination:",
        options=available,
        format_func=lambda n: f"{n}  ({REGISTRY[n]['accuracy']*100:.1f}% acc, {REGISTRY[n]['framework']})",
        key="model_multiselect",
    )

    if selected_models:
        chip_html = ""
        for n in selected_models:
            fw_class = "framework-keras" if REGISTRY[n]["framework"] == "keras" else "framework-pytorch"
            chip_html += f'<span class="model-chip {fw_class}">{n} · {REGISTRY[n]["framework"]}</span>'
        st.markdown(chip_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("2. Provide a retina image")

    input_method = st.radio(
        "Input method:",
        ["📁 Upload a file", "📷 Capture via camera"],
        horizontal=True,
    )

    if input_method == "📁 Upload a file":
        uploaded_file = st.file_uploader("Choose a fundus image (jpg/png)", type=["jpg", "jpeg", "png"])
    else:
        st.caption(
            "⚠️ This captures whatever your camera sees right now — it must already be a real "
            "fundus/retina image (e.g. from a smartphone fundus camera attachment such as D-EYE "
            "or Peek Retina, or by photographing an existing printed/displayed fundus image). "
            "A regular photo of the outside of an eye does **not** contain retinal information "
            "and cannot be analyzed by these models."
        )
        uploaded_file = st.camera_input("Capture a fundus image")

    col_img, col_results = st.columns([1, 1.4], gap="large")

    with col_img:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image to be analyzed", use_container_width=True)

    with col_results:
        if not uploaded_file:
            st.info("Provide an image and select at least one model to run a prediction.")
        elif not selected_models:
            st.warning("Select at least one model above to run a prediction.")
        else:
            if st.button("▶️ Run Prediction", type="primary", use_container_width=True):
                loader_slot = st.empty()
                loader_slot.markdown(
                    f'<div class="scan-loader-wrap"><div class="scan-loader-ring"></div>'
                    f'<div class="scan-loader-text">Scanning with {len(selected_models)} model(s)...</div></div>',
                    unsafe_allow_html=True,
                )
                loaded = load_models_cached(tuple(selected_models))
                output = eu.run_predictions(selected_models, image, loaded)
                loader_slot.empty()
                st.session_state.last_output = output
                st.session_state.last_selected = selected_models

            if "last_output" in st.session_state:
                output = st.session_state.last_output
                used_models = st.session_state.last_selected

                ensemble_prob = output["ensemble_probability"]
                verdict = output["ensemble_verdict"]

                # --- Risk level, not just binary -- a bit more nuance for the demo ---
                if ensemble_prob < 0.35:
                    risk_label, risk_emoji, risk_color = "Low likelihood of DR", "🟢", "#2D9C8F"
                elif ensemble_prob < 0.65:
                    risk_label, risk_emoji, risk_color = "Borderline / uncertain", "🟡", "#E8A33D"
                else:
                    risk_label, risk_emoji, risk_color = "High likelihood of DR", "🔴", "#C0392B"

                box_class = "verdict-box-dr" if verdict == "Has DR" else "verdict-box-nodr"
                ring_col, verdict_col = st.columns([1, 2.2])
                with ring_col:
                    st.markdown(
                        f'<div style="text-align:center;">{render_confidence_ring(ensemble_prob, risk_color, size=140, stroke=11)}</div>',
                        unsafe_allow_html=True,
                    )
                with verdict_col:
                    st.markdown(
                        f'<div class="{box_class}"><h4>{risk_emoji} Ensemble Verdict: {verdict}</h4>'
                        f'<p>{risk_label} &mdash; confidence: <b>{ensemble_prob*100:.1f}%</b> '
                        f'(based on {len(used_models)} model{"s" if len(used_models)!=1 else ""})</p></div>',
                        unsafe_allow_html=True,
                    )

                # --- Model agreement indicator ---
                agree_count = sum(1 for r in output["per_model"].values() if r["prediction"] == verdict)
                st.caption(f"Model agreement: {agree_count}/{len(used_models)} models agree with the ensemble verdict.")

                st.markdown("##### Per-model breakdown")
                rows = []
                for name in used_models:
                    res = output["per_model"][name]
                    rows.append({
                        "Model": name,
                        "Prediction": res["prediction"],
                        "Confidence": f"{res['probability']*100:.1f}%",
                        "Model Accuracy": f"{REGISTRY[name]['accuracy']*100:.1f}%",
                        "Framework": REGISTRY[name]["framework"],
                    })
                result_df = pd.DataFrame(rows)
                st.dataframe(result_df, hide_index=True, use_container_width=True)

                # --- PDF export ---
                pdf_bytes = pdf_report.generate_report(
                    output["per_model"], ensemble_prob, verdict, used_models, REGISTRY
                )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"dr_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.caption("⚠️ Educational project only — not a certified medical diagnostic tool.")

# ===========================================================================
# TAB 2: GENERAL SCREENING (for the general public, no equipment needed)
# ===========================================================================
with tab_general:
    st.markdown("""
    <div class="video-banner">
      <video autoplay loop muted playsinline>
        <source src="app/static/general_screening_banner.mp4" type="video/mp4">
      </video>
      <div class="banner-fade"></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("👥 General Diabetes Risk Screening")
    st.caption(
        "For anyone without access to a retinal camera. Enter basic health info "
        "you likely already know — this estimates **general diabetes risk**, "
        "not diabetic retinopathy specifically. Trained on the Pima Indians "
        "Diabetes Dataset using the same multi-model approach as the rest of "
        "this project (Logistic Regression, Random Forest, KNN, SVM, XGBoost + ensemble)."
    )
    st.info(
        "💡 This is a screening tool, similar in spirit to public risk "
        "calculators like the ADA Diabetes Risk Test. If your result shows "
        "elevated risk, the right next step is a real medical check-up — "
        "and if it's available, the specialist Predict tab with an actual "
        "retina image.",
        icon="💡",
    )

    with st.form("general_risk_form"):
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            g_age = st.number_input("Age", min_value=1, max_value=120, value=35)
            g_preg = st.number_input("Number of pregnancies (enter 0 if not applicable)", min_value=0, max_value=20, value=0)
            g_bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=24.0, step=0.1)
            g_bp = st.number_input("Blood pressure (mm Hg, if known)", min_value=0, max_value=200, value=75)
        with gcol2:
            g_glucose = st.number_input("Blood glucose level (mg/dL, if known -- 100-125 typical fasting range)", min_value=0, max_value=300, value=110)
            g_insulin = st.number_input("Insulin level (mu U/ml, leave 0 if unknown)", min_value=0, max_value=900, value=0)
            g_skin = st.number_input("Triceps skinfold thickness (mm, leave 0 if unknown)", min_value=0, max_value=100, value=0)
            g_dpf = st.slider(
                "Family history strength (0 = none, 1 = strong family history of diabetes)",
                min_value=0.0, max_value=2.0, value=0.3, step=0.05,
            )

        submitted = st.form_submit_button("🔍 Check My Risk", type="primary", use_container_width=True)

    if submitted:
        result = gru.predict_risk(g_preg, g_glucose, g_bp, g_skin, g_insulin, g_bmi, g_dpf, g_age)
        prob = result["ensemble_probability"]
        verdict = result["ensemble_verdict"]

        if prob < 0.35:
            risk_label, risk_color, risk_emoji = "Lower risk", "#2D9C8F", "🟢"
        elif prob < 0.65:
            risk_label, risk_color, risk_emoji = "Moderate / borderline risk", "#E8A33D", "🟡"
        else:
            risk_label, risk_color, risk_emoji = "Elevated risk", "#C0392B", "🔴"

        gauge_col, info_col = st.columns([1, 1.3])
        with gauge_col:
            st.markdown(
                f'<div style="text-align:center; padding-top:10px;">'
                f'{render_confidence_ring(prob, risk_color, size=180, stroke=14)}'
                f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.75rem; '
                f'color:#b89a78; text-transform:uppercase; letter-spacing:0.08em; margin-top:6px;">'
                f'Estimated Diabetes Risk</div></div>',
                unsafe_allow_html=True,
            )

        with info_col:
            st.markdown(f"### {risk_emoji} {risk_label}")
            st.write(f"Ensemble estimated probability: **{prob*100:.1f}%**")
            st.caption("Based on the combined vote of 5 different machine learning models.")

            with st.expander("See individual model results"):
                rows = []
                for name, res in result["per_model"].items():
                    rows.append({
                        "Model": name,
                        "Prediction": res["prediction"],
                        "Probability": f"{res['probability']*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.caption("⚠️ This is a general screening estimate, not a diagnosis. Please consult a healthcare professional for an accurate assessment.")


with tab_dashboard:
    st.markdown("""
    <div class="video-banner">
      <video autoplay loop muted playsinline>
        <source src="app/static/dashboard_diagnostics.mp4" type="video/mp4">
      </video>
      <div class="banner-fade"></div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 All 9 Trained Models — Performance Comparison")

    all_results = get_all_results()

    # Build the DataFrame explicitly from scalar fields only -- mixing in
    # confusion_matrix (a list-of-lists) here was what caused pandas to treat
    # the whole table as "object" dtype instead of numeric, breaking .round().
    records = []
    for name, vals in all_results.items():
        records.append({
            "model": name,
            "accuracy": float(vals["accuracy"]),
            "auc": float(vals["auc"]),
            "loss": float(vals["loss"]),
            "img_size": vals["img_size"],
            "framework": vals["framework"],
        })
    dash_df = pd.DataFrame(records).set_index("model")
    dash_df = dash_df.sort_values("accuracy", ascending=False)

    # --- Shared Plotly template matching the fundus color system ---
    fundus_template = go.layout.Template()
    fundus_template.layout = go.Layout(
        paper_bgcolor="#1A0F0C", plot_bgcolor="#1A0F0C",
        font=dict(family="Inter, sans-serif", color="#F5E6D3"),
        xaxis=dict(gridcolor="rgba(232,163,61,0.12)", zerolinecolor="rgba(232,163,61,0.2)"),
        yaxis=dict(gridcolor="rgba(232,163,61,0.12)", zerolinecolor="rgba(232,163,61,0.2)"),
        title=dict(font=dict(family="Fraunces, serif", size=18, color="#F5E6D3")),
    )

    # --- Headline readout cards (custom HTML, instrument-panel style) ---
    best_model = dash_df.index[0]
    r1, r2, r3, r4 = st.columns(4)
    readouts = [
        (r1, "BEST MODEL", best_model, f"↑ {dash_df.loc[best_model,'accuracy']*100:.2f}%"),
        (r2, "AVERAGE ACCURACY", f"{dash_df['accuracy'].mean()*100:.2f}%", "across 9 models"),
        (r3, "MODELS ABOVE 90%", f"{(dash_df['accuracy'] >= 0.90).sum()} / {len(dash_df)}", "full clearance"),
        (r4, "FRAMEWORKS USED", str(dash_df["framework"].nunique()), "Keras + PyTorch"),
    ]
    for col, label, value, delta in readouts:
        col.markdown(f"""
        <div class="readout-card">
          <div class="readout-label">{label}</div>
          <div class="readout-value">{value}</div>
          <div class="readout-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_acc = px.bar(
            dash_df.reset_index(), x="accuracy", y="model", orientation="h",
            color="accuracy", color_continuous_scale=[[0, "#5a3a1e"], [0.5, "#C0392B"], [1, "#E8A33D"]],
            text=dash_df["accuracy"].apply(lambda v: f"{v*100:.2f}%"),
            title="Test Accuracy by Model", template=fundus_template,
        )
        fig_acc.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, coloraxis_showscale=False)
        fig_acc.update_traces(textposition="outside", textfont_color="#F5E6D3")
        st.plotly_chart(fig_acc, use_container_width=True)
    with chart_col2:
        fig_auc = px.bar(
            dash_df.reset_index(), x="auc", y="model", orientation="h",
            color="auc", color_continuous_scale=[[0, "#1d5c54"], [1, "#2D9C8F"]],
            text=dash_df["auc"].apply(lambda v: f"{v:.4f}"),
            title="AUC by Model", template=fundus_template,
        )
        fig_auc.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, coloraxis_showscale=False)
        fig_auc.update_traces(textposition="outside", textfont_color="#F5E6D3")
        st.plotly_chart(fig_auc, use_container_width=True)

    # --- Scatter: accuracy vs model size, framework-colored -- shows the
    # accuracy/efficiency tradeoff at a glance, a nice analytical touch ---
    st.markdown("**Accuracy vs. Model Size** (bubble size = parameter count)")
    scatter_df = dash_df.reset_index()
    scatter_df["params_millions"] = scatter_df["model"].map(lambda n: REGISTRY[n]["params_millions"])
    fig_scatter = px.scatter(
        scatter_df, x="params_millions", y="accuracy", color="framework",
        size="params_millions", hover_name="model", text="model",
        labels={"params_millions": "Parameters (Millions)", "accuracy": "Test Accuracy"},
        color_discrete_map={"keras": "#E8A33D", "pytorch": "#2D9C8F"},
        template=fundus_template,
    )
    fig_scatter.update_traces(textposition="top center", textfont_color="#F5E6D3")
    fig_scatter.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.markdown("**Full comparison table**")
    dash_df_display = dash_df.copy()
    dash_df_display["accuracy"] = (dash_df_display["accuracy"] * 100).round(2).astype(str) + "%"
    dash_df_display["auc"] = dash_df_display["auc"].round(4)
    dash_df_display["loss"] = dash_df_display["loss"].round(4)
    st.dataframe(
        dash_df_display[["accuracy", "auc", "loss", "img_size", "framework"]],
        use_container_width=True,
    )

    st.markdown("**Architecture details**")
    arch_rows = []
    for name in dash_df.index:
        meta = REGISTRY.get(name, {})
        arch_rows.append({
            "Model": name,
            "Parameters (M)": meta.get("params_millions", "—"),
            "Framework": meta.get("framework", "—"),
            "Description": meta.get("description", "—"),
        })
    st.dataframe(pd.DataFrame(arch_rows), hide_index=True, use_container_width=True)

# ===========================================================================
# TAB 3: ABOUT
# ===========================================================================
with tab_about:
    st.subheader("About this project")
    st.markdown("""
    **Multi-Model AI Prediction System** — Diabetic Retinopathy Detection

    This system trains and compares **9 independent deep learning models**
    (7 from TensorFlow/Keras, 2 from PyTorch/timm) on the APTOS 2019 Blindness
    Detection dataset, then combines the strongest of them into a configurable
    ensemble for the final prediction.

    ### Methodology
    - **Dataset:** APTOS 2019 Blindness Detection (public Kaggle competition dataset)
    - **Task:** Binary classification — No DR vs Has DR (diabetic retinopathy severity 1-4 combined)
    - **Split:** 70% train / 15% validation / 15% test, stratified
    - **Training:** Transfer learning — frozen backbone first, then fine-tuning the top layers
    - **Augmentation:** Light augmentation only (rotation, zoom, flips) to avoid over-distorting retina images

    ### A key part of this project's methodology: preprocessing correctness
    Each of the 9 architectures requires **different input preprocessing**,
    confirmed individually against each model's own documentation/config rather
    than assumed by analogy:

    | Model family | Required preprocessing |
    |---|---|
    | EfficientNet (B0/B4/B6/V2S), MobileNetV3 | Raw [0,255], normalizes internally |
    | ResNet50 | RGB→BGR, ImageNet mean-subtracted, no scaling |
    | DenseNet121 | [0,255]→[0,1], then ImageNet mean/std normalize |
    | MobileViT-XXS | Simple [0,1] scaling only, no further normalization |
    | EfficientFormerV2-S1 | [0,1] scaling + standard ImageNet mean/std |

    Getting this wrong was the difference between a model achieving 50% (random
    guessing) versus 90%+ accuracy during development — this system's data
    pipeline accounts for each model's specific requirement individually.

    ### Cross-framework ensemble
    Combining TensorFlow and PyTorch models in a single running application
    required resolving a native library conflict between the two frameworks
    (resolved via specific import ordering) — enabling Keras and PyTorch
    models to genuinely run side-by-side in one ensemble, rather than treating
    them as separate systems.
    """)

    st.divider()
    st.caption("⚠️ Educational project only — not a certified medical diagnostic tool.")
