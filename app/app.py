import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)

import streamlit as st, pandas as pd, joblib
from src.features import add_features

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "best_model.joblib"))
model = load_model()

PLAYBOOK = {
    "New & No Commitment": "Offer 10% off for a 12-month contract. Owner: Retention call team.",
    "Unsupported Fiber Users": "Give free tech support for 3 months and make a check-in call.",
    "High Bill, Low Value": "Suggest a cheaper plan or a value add-on bundle.",
    "Loyal Long-Term": "Send loyalty rewards or an early upgrade.",
    "Steady Others": "No action needed. Keep monitoring.",
}

def segment(r):
    if r["Contract"] == "Month-to-month" and r["tenure"] <= 12: return "New & No Commitment"
    if r["InternetService"] == "Fiber optic" and r["has_support"] == 0: return "Unsupported Fiber Users"
    if r["MonthlyCharges"] > 70 and r["num_services"] <= 2: return "High Bill, Low Value"
    if r["tenure"] > 24 and r["Contract"] != "Month-to-month": return "Loyal Long-Term"
    return "Steady Others"

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Predictor & Retention Advisor")

c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    phone = st.selectbox("Phone service", ["Yes", "No"])
with c2:
    multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    sec = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    protect = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
with c3:
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    pay = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 85.0)

if st.button("Predict", type="primary"):
    if internet == "No":
        sec = backup = protect = support = tv = movies = "No internet service"
    if phone == "No":
        multi = "No phone service"

    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": protect,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": pay,
        "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
    }])
    row = add_features(row)
    p = model.predict_proba(row)[0, 1]
    tier = "High" if p > 0.6 else "Medium" if p > 0.3 else "Low"
    seg = segment(row.iloc[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{p:.0%}")
    m2.metric("Risk tier", tier)
    m3.metric("Segment", seg)
    st.success(f"Recommended action: {PLAYBOOK[seg]}")

    pre, clf = model.named_steps["prep"], model.named_steps["model"]
    Xt = pre.transform(row)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = pre.get_feature_names_out()

    if hasattr(clf, "coef_"):                      # logistic regression
        contrib = pd.Series(clf.coef_[0] * Xt[0], index=names)
    else:                                          # XGBoost / tree models
        import shap
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        if isinstance(sv, list):                   # older sklearn forests
            sv = sv[1]
        sv = sv[0, :, 1] if getattr(sv, "ndim", 2) == 3 else sv[0]
        contrib = pd.Series(sv, index=names)

    drivers = contrib.sort_values(key=abs, ascending=False).head(5)
    st.subheader("Why this score? (top 5 drivers)")
    st.bar_chart(drivers)
    st.caption("Positive = pushes toward churn. Negative = pushes toward staying.")
