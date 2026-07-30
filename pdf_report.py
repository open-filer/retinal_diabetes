"""
Generates a downloadable PDF report summarizing a single prediction run:
which models were used, their individual verdicts, and the final ensemble
result. Built with fpdf2 (lightweight, no external system dependencies).
"""

import io
from datetime import datetime
from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "Multi-Model Diabetic Retinopathy Detection Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, "Educational project only -- not a certified medical diagnostic tool.", align="C")


def generate_report(per_model_results, ensemble_probability, ensemble_verdict, selected_models, model_registry):
    """Returns PDF bytes for the given prediction results."""
    pdf = ReportPDF()
    pdf.add_page()

    # --- Final verdict banner ---
    pdf.set_font("Helvetica", "B", 13)
    verdict_color = (200, 60, 60) if ensemble_verdict == "Has DR" else (40, 140, 90)
    pdf.set_text_color(*verdict_color)
    pdf.cell(0, 10, f"Ensemble Verdict: {ensemble_verdict}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, f"Ensemble confidence (probability of DR): {ensemble_probability*100:.1f}%", ln=True)
    pdf.ln(4)

    # --- Per-model breakdown table ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, f"Individual Model Results ({len(selected_models)} models used)", ln=True)
    pdf.ln(2)

    col_widths = [55, 30, 35, 35]
    headers = ["Model", "Prediction", "Confidence", "Model Accuracy"]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(235, 235, 235)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for name in selected_models:
        res = per_model_results[name]
        meta = model_registry[name]
        row = [
            name,
            res["prediction"],
            f"{res['probability']*100:.1f}%",
            f"{meta['accuracy']*100:.2f}%",
        ]
        for w, val in zip(col_widths, row):
            pdf.cell(w, 7, str(val), border=1)
        pdf.ln()

    pdf.ln(6)

    # --- Methodology note ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Methodology", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    methodology_text = (
        "Each model was trained via transfer learning on the APTOS 2019 Blindness "
        "Detection dataset (train 70% / validation 15% / test 15%, stratified split), "
        "with light data augmentation and model-specific input preprocessing confirmed "
        "individually for each architecture. The ensemble verdict is the average "
        "predicted probability across all selected models."
    )
    pdf.multi_cell(0, 6, methodology_text)

    return bytes(pdf.output())
