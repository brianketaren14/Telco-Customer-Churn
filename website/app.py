"""
RETAIN// Churn Risk Console — Flask backend
=============================================
Memuat model Logistic Regression (model.pkl), scaler (scaler.pkl), dan 
selected_features (selected_features.pkl), melakukan preprocessing manual 
(manual one-hot encoding + scaling), lalu mengembalikan prediksi biner (0/1) 
beserta probabilitasnya.

>>> WAJIB KAMU SESUAIKAN SEBELUM DIJALANKAN <<<
1. FEATURE_COLUMNS di bawah harus PERSIS SAMA (nama + urutan) dengan
   kolom-kolom one-hot encoded saat training model kamu.
2. Letakkan model.pkl, scaler.pkl, dan selected_features.pkl di folder model/.
3. Manual one-hot encoding digunakan (tidak pd.get_dummies) untuk kontrol penuh
   atas mapping kategori.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
SELECTED_FEATURES_PATH = BASE_DIR / "model" / "selected_features.pkl"
TEMPLATES_DIR = BASE_DIR / "templates"


# -------------------------------------------------------------------
# 1. LOAD MODEL & SCALER
# -------------------------------------------------------------------
model = None
scaler = None
load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(SELECTED_FEATURES_PATH, 'rb') as f:
        selected_features = pickle.load(f)

    print(f"[OK] Model dimuat dari {MODEL_PATH}")
    print(f"[OK] Scaler dimuat dari {SCALER_PATH}")
    print(f"[OK] Selected features dimuat dari {SELECTED_FEATURES_PATH}")

except FileNotFoundError as e:
    load_error = str(e)
    print(f"[WARNING] {e}")
    print("[WARNING] Letakkan model.pkl dan scaler.pkl di folder model/ lalu restart server.")


# -------------------------------------------------------------------
# 2. KONFIGURASI PREPROCESSING — EDIT BAGIAN INI SESUAI NOTEBOOK KAMU
# -------------------------------------------------------------------

# Kolom numerik yang DI-SCALE (sesuai scaler baru yang mencakup CLTV)
NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]

# Fitur yang diperlukan model (setelah disesuaikan dengan dataset baru)
FEATURE_COLUMNS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Gender_Male",
    "Senior Citizen_Yes",
    "Partner_Yes",
    "Dependents_Yes",
    "Phone Service_Yes",
    "Paperless Billing_Yes",
    "Multiple Lines_No",
    "Multiple Lines_No phone service",
    "Multiple Lines_Yes",
    "Internet Service_DSL",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No",
    "Online Security_No internet service",
    "Online Security_Yes",
    "Online Backup_No",
    "Online Backup_No internet service",
    "Online Backup_Yes",
    "Device Protection_No",
    "Device Protection_No internet service",
    "Device Protection_Yes",
    "Tech Support_No",
    "Tech Support_No internet service",
    "Tech Support_Yes",
    "Streaming TV_No",
    "Streaming TV_No internet service",
    "Streaming TV_Yes",
    "Streaming Movies_No",
    "Streaming Movies_No internet service",
    "Streaming Movies_Yes",
    "Contract_Month-to-month",
    "Contract_One year",
    "Contract_Two year",
    "Payment Method_BT",
    "Payment Method_CC",
    "Payment Method_EC",
    "Payment Method_MC",
]


def preprocess(raw: dict) -> pd.DataFrame:
    """
    Ubah payload JSON mentah dari form menjadi satu baris DataFrame
    yang siap di-predict dengan selected_features.
    """
    # 1. Inisialisasi dictionary untuk hasil encoding
    encoded_data = {}
    
    # 2. Masukkan numerik columns dengan mapping nama kolom baru
    encoded_data["Tenure Months"] = float(raw.get("tenure", 0))
    encoded_data["Monthly Charges"] = float(raw.get("MonthlyCharges", 0))
    encoded_data["Total Charges"] = float(raw.get("TotalCharges", 0))
    encoded_data["CLTV"] = float(raw.get("CLTV", 0))
    
    # 3. Manual one-hot encoding untuk categorical columns dengan mapping nama kolom baru
    input_mapping = {
        "gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "PhoneService": "Phone Service",
        "PaperlessBilling": "Paperless Billing",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract",
        "PaymentMethod": "Payment Method"
    }

    categorical_mappings = {
        "Gender": ["Female", "Male"],
        "Senior Citizen": ["No", "Yes"],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "Phone Service": ["No", "Yes"],
        "Paperless Billing": ["No", "Yes"],
        "Multiple Lines": ["No", "No phone service", "Yes"],
        "Internet Service": ["DSL", "Fiber optic", "No"],
        "Online Security": ["No", "No internet service", "Yes"],
        "Online Backup": ["No", "No internet service", "Yes"],
        "Device Protection": ["No", "No internet service", "Yes"],
        "Tech Support": ["No", "No internet service", "Yes"],
        "Streaming TV": ["No", "No internet service", "Yes"],
        "Streaming Movies": ["No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Payment Method": ["BT", "CC", "EC", "MC"],
    }
    
    # Manual one-hot encoding
    for raw_key, model_key in input_mapping.items():
        value = raw.get(raw_key, categorical_mappings[model_key][0])
        for cat in categorical_mappings[model_key]:
            col_name = f"{model_key}_{cat}"
            encoded_data[col_name] = 1 if value == cat else 0
    
    # 4. Buat DataFrame dari encoded_data
    df_encoded = pd.DataFrame([encoded_data])
    
    # 5. Pastikan semua FEATURE_COLUMNS ada (isi 0 jika tidak)
    for col in FEATURE_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # 6. Urutkan sesuai FEATURE_COLUMNS
    df_encoded = df_encoded[FEATURE_COLUMNS]
    
    # 7. Scaling numerik SEBELUM filtering selected_features
    df_encoded[NUMERIC_COLUMNS] = scaler.transform(df_encoded[NUMERIC_COLUMNS])
    
    # 8. Filter dengan selected_features dan reorder
    if selected_features is not None:
        available_features = [f for f in selected_features if f in df_encoded.columns]
        df_final = df_encoded[available_features]
    else:
        df_final = df_encoded
    
    return df_final


# -------------------------------------------------------------------
# 3. ROUTES
# -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report")
def report():
    return render_template("report_container.html")


@app.route("/evidently-report")
def evidently_report():
    return send_from_directory(TEMPLATES_DIR, "report.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or selected_features is None:
        return jsonify({
            "error": "Model, scaler, atau selected_features belum dimuat. Pastikan model.pkl, "
                     "scaler.pkl, dan selected_features.pkl ada di folder model/, lalu restart server.",
            "detail": load_error
        }), 500

    try:
        raw = request.get_json(force=True)

        # TotalCharges kosong -> estimasi dari tenure * MonthlyCharges
        if raw.get("TotalCharges") in (None, "", "null"):
            try:
                raw["TotalCharges"] = float(raw.get("tenure", 0)) * float(raw.get("MonthlyCharges", 0))
            except (TypeError, ValueError):
                raw["TotalCharges"] = 0.0

        X = preprocess(raw)

        pred_class = int(model.predict(X)[0])

        # probabilitas churn (kelas 1), jika model mendukung predict_proba
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        else:
            proba = float(pred_class)  # fallback kalau model tidak punya predict_proba

        return jsonify({
            "prediction": pred_class,           # 0 = tidak churn, 1 = churn
            "churn_probability": round(proba, 4),
            "risk_percent": round(proba * 100, 1)
        })

    except KeyError as e:
        return jsonify({"error": f"Kolom input tidak lengkap: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal melakukan prediksi: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if (model is not None and scaler is not None and selected_features is not None) else "model_not_loaded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "selected_features_loaded": selected_features is not None,
        "num_selected_features": len(selected_features) if selected_features is not None else 0
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
