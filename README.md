# Customer Churn Deployed

An end-to-end customer churn prediction platform combining machine learning with an interactive web dashboard. Upload your customer data CSV and instantly receive churn risk scores, EDA visualisations, a business impact calculator, and a downloadable executive summary report.

---

## Project Structure

```
CustomerChurnDeployed/
├── app.py                          # Flask backend
├── train_model.py                  # Offline model training script
├── churn_analysis.ipynb            # Full EDA + ML notebook
├── requirements.txt
├── README.md
├── data/
│   └── Bank_Churn_Classification_Dataset.csv
├── models/
│   ├── churn_model.pkl             # Trained XGBoost model (generated)
│   └── preprocessor.pkl            # Fitted sklearn pipeline (generated)
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── dashboard.js
└── templates/
    └── index.html
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time step)

```bash
python train_model.py
```

This reads the dataset, trains an XGBoost classifier, and saves `models/churn_model.pkl` and `models/preprocessor.pkl`.

### 3. Run the Flask server

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

### 4. Upload data

Upload any CSV that shares the same column schema as the training dataset. The dashboard will render:

- Churn risk distribution chart
- Key churn driver breakdown
- High-risk customer table with revenue-at-risk estimates
- Business Impact Calculator
- Downloadable Executive Summary PDF

---

## Dataset Schema

The model expects a CSV with the following columns:

| Column | Type | Description |
|---|---|---|
| `CustomerID` | int | Unique customer identifier |
| `Gender` | str | Male / Female |
| `SeniorCitizen` | int | 0 or 1 |
| `Tenure` | int | Months with company |
| `MonthlyCharges` | float | Monthly billing amount |
| `Contract` | str | Month-to-month / One year / Two year |
| `PaymentMethod` | str | Electronic check / Bank transfer / Credit card / Mailed check |
| `TotalCharges` | float | Cumulative charges |
| `Churn` | int | 0 or 1 (optional — omit for prediction-only mode) |

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | ~0.91 |
| F1 (churn class) | ~0.72 |
| Precision | ~0.75 |
| Recall | ~0.69 |

*Exact values depend on the random seed and train/test split.*

---

## Tech Stack

- **ML**: scikit-learn, XGBoost
- **Backend**: Flask
- **Frontend**: Vanilla HTML / CSS / JavaScript, Chart.js
- **PDF Reports**: fpdf2

---

## Business Impact Calculator

The dashboard estimates monthly revenue at risk using:

```
Revenue at Risk = Σ(MonthlyCharges for high-risk customers)
Potential Savings = Revenue at Risk × assumed 40% retention campaign effectiveness
```

Adjust the risk threshold slider to see how targeting precision affects both cost and savings.
