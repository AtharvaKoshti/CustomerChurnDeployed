import io
import json
import os
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from fpdf import FPDF

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "churn_model.pkl")
NUMERIC_COLS = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = ["Gender", "Contract", "PaymentMethod"]
REQUIRED_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna({
        "Tenure": df["Tenure"].median() if "Tenure" in df else 0,
        "MonthlyCharges": df["MonthlyCharges"].median() if "MonthlyCharges" in df else 0,
        "TotalCharges": df["TotalCharges"].median() if "TotalCharges" in df else 0,
        "SeniorCitizen": 0,
    }, inplace=True)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")

    df["ChargeRatio"] = df["TotalCharges"] / (df["Tenure"] + 1)
    return df


def build_summary(df: pd.DataFrame, proba: np.ndarray) -> dict:
    df = df.copy()
    df["churn_probability"] = proba
    df["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    )

    total = len(df)
    high_risk_df = df[df["risk_tier"] == "High"]
    revenue_at_risk = float(high_risk_df["MonthlyCharges"].sum())
    potential_savings = revenue_at_risk * 0.40  # 40% retention campaign effectiveness

    # Churn rate per contract type
    churn_by_contract = (
        df.groupby("Contract")["churn_probability"]
        .mean()
        .round(3)
        .to_dict()
    )

    # Churn rate per payment method
    churn_by_payment = (
        df.groupby("PaymentMethod")["churn_probability"]
        .mean()
        .round(3)
        .to_dict()
    )

    # Avg churn probability by tenure bucket
    df["tenure_bucket"] = pd.cut(df["Tenure"], bins=[0, 12, 24, 48, 999], labels=["0-12m", "13-24m", "25-48m", "48m+"])
    churn_by_tenure = (
        df.groupby("tenure_bucket", observed=True)["churn_probability"]
        .mean()
        .round(3)
        .to_dict()
    )
    churn_by_tenure = {str(k): v for k, v in churn_by_tenure.items()}

    # Risk tier distribution
    risk_counts = df["risk_tier"].value_counts().to_dict()
    risk_distribution = {
        "High": int(risk_counts.get("High", 0)),
        "Medium": int(risk_counts.get("Medium", 0)),
        "Low": int(risk_counts.get("Low", 0)),
    }

    # Top 10 high-risk customers for the table
    top_at_risk = (
        df.nlargest(10, "churn_probability")[
            ["CustomerID", "Tenure", "MonthlyCharges", "Contract", "PaymentMethod", "churn_probability"]
        ]
        .round({"churn_probability": 3})
        .to_dict(orient="records")
    ) if "CustomerID" in df.columns else (
        df.nlargest(10, "churn_probability")[
            ["Tenure", "MonthlyCharges", "Contract", "PaymentMethod", "churn_probability"]
        ]
        .round({"churn_probability": 3})
        .to_dict(orient="records")
    )

    # Monthly charges distribution by risk
    charges_by_risk = {
        tier: round(float(df[df["risk_tier"] == tier]["MonthlyCharges"].mean()), 2)
        for tier in ["High", "Medium", "Low"]
        if len(df[df["risk_tier"] == tier]) > 0
    }

    return {
        "total_customers": total,
        "high_risk_count": int(risk_distribution["High"]),
        "medium_risk_count": int(risk_distribution["Medium"]),
        "low_risk_count": int(risk_distribution["Low"]),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "potential_savings": round(potential_savings, 2),
        "avg_churn_probability": round(float(proba.mean()), 3),
        "churn_by_contract": churn_by_contract,
        "churn_by_payment": churn_by_payment,
        "churn_by_tenure": churn_by_tenure,
        "risk_distribution": risk_distribution,
        "charges_by_risk": charges_by_risk,
        "top_at_risk": top_at_risk,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file"}), 400

    try:
        df_raw = pd.read_csv(file)

        missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        df_clean = clean_and_enrich(df_raw)
        model = load_model()
        proba = model.predict_proba(df_clean)[:, 1]

        summary = build_summary(df_clean, proba)
        return jsonify(summary)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/report", methods=["POST"])
def generate_report():
    """Generate a PDF executive summary from the analysis payload."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(30, 58, 138)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 14, "Customer Churn Analysis - Executive Summary", ln=True)

    pdf.set_text_color(50, 50, 50)
    pdf.set_xy(10, 36)

    def section(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(240, 244, 255)
        pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
        pdf.ln(2)

    def row(label, value, highlight=False):
        pdf.set_font("Helvetica", "", 10)
        if highlight:
            pdf.set_fill_color(254, 243, 199)
            pdf.cell(95, 7, f"  {label}", border=0, fill=True)
            pdf.cell(85, 7, str(value), ln=True, fill=True)
        else:
            pdf.cell(95, 7, f"  {label}")
            pdf.cell(85, 7, str(value), ln=True)

    section("Portfolio Overview")
    row("Total Customers Analysed", f"{data.get('total_customers', 'N/A'):,}")
    row("Average Churn Probability", f"{data.get('avg_churn_probability', 0)*100:.1f}%")
    row("High-Risk Customers", f"{data.get('high_risk_count', 0):,}", highlight=True)
    row("Medium-Risk Customers", f"{data.get('medium_risk_count', 0):,}")
    row("Low-Risk Customers", f"{data.get('low_risk_count', 0):,}")

    pdf.ln(4)
    section("Business Impact")
    row("Monthly Revenue at Risk (High-Risk)", f"${data.get('revenue_at_risk', 0):,.2f}", highlight=True)
    row("Potential Monthly Savings (40% retention)", f"${data.get('potential_savings', 0):,.2f}", highlight=True)
    annual_risk = data.get('revenue_at_risk', 0) * 12
    annual_save = data.get('potential_savings', 0) * 12
    row("Annualised Revenue at Risk", f"${annual_risk:,.2f}")
    row("Annualised Potential Savings", f"${annual_save:,.2f}")

    pdf.ln(4)
    section("Churn Rate by Contract Type")
    for contract, rate in data.get("churn_by_contract", {}).items():
        row(contract, f"{rate*100:.1f}% avg churn probability",
            highlight=(rate == max(data.get("churn_by_contract", {1: 0}).values())))

    pdf.ln(4)
    section("Churn Rate by Tenure Segment")
    for bucket, rate in data.get("churn_by_tenure", {}).items():
        row(bucket, f"{rate*100:.1f}% avg churn probability")

    pdf.ln(4)
    section("Actionable Recommendations")
    recommendations = [
        "1. Prioritise retention outreach for Month-to-month contract customers - highest churn risk.",
        "2. Offer contract upgrade incentives (annual / two-year) to reduce monthly churn exposure.",
        "3. Review Electronic Check payment customers - friction in payment may be driving churn.",
        "4. Focus on customers with Tenure < 12 months - early disengagement is a leading indicator.",
        "5. A 40% retention rate on high-risk accounts saves the business significant monthly revenue.",
    ]
    pdf.set_font("Helvetica", "", 10)
    for rec in recommendations:
        pdf.multi_cell(0, 7, f"  {rec}")
        pdf.ln(1)

    pdf.ln(4)
    section("Top 5 High-Risk Customers")
    top = data.get("top_at_risk", [])[:5]
    if top:
        headers = ["Tenure", "Monthly Charges", "Contract", "Churn Prob."]
        col_w = [35, 45, 65, 40]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(220, 230, 255)
        for h, w in zip(headers, col_w):
            pdf.cell(w, 7, h, border=1, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for cust in top:
            pdf.cell(35, 7, str(cust.get("Tenure", "")), border=1)
            pdf.cell(45, 7, f"${cust.get('MonthlyCharges', 0):.2f}", border=1)
            pdf.cell(65, 7, str(cust.get("Contract", "")), border=1)
            pdf.cell(40, 7, f"{cust.get('churn_probability', 0)*100:.1f}%", border=1)
            pdf.ln()

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Generated by Customer Churn Deployed - Confidential", align="C")

    pdf_bytes = pdf.output()
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="churn_executive_summary.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
