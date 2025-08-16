import streamlit as st
import pickle
import pandas as pd

# Load model & scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model_gbc.pkl", "rb"))

# Prediction function
def predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc):
    df = pd.DataFrame([{
        'age': age, 'bp': bp, 'sg': sg, 'al': al,
        'hemo': hemo, 'sc': sc,
        'htn': htn, 'dm': dm, 'cad': cad,
        'appet': appet, 'pc': pc
    }])

    # Encode categorical
    df['htn']   = df['htn'].map({'yes':1, "no":0})
    df['dm']    = df['dm'].map({'yes':1, "no":0})
    df['cad']   = df['cad'].map({'yes':1, "no":0})
    df['appet'] = df['appet'].map({'good':1, "poor":0})
    df['pc']    = df['pc'].map({'normal':1, "abnormal":0})

    # Scale numeric
    numeric_cols = ['age','bp','sg','al','hemo','sc']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predict
    pred = model.predict(df)[0]
    return "✅ Patient has CKD" if pred == 1 else "❌ Patient does NOT have CKD"

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🩺 Chronic Kidney Disease Predictor")
st.write("Enter patient details to check CKD risk")

# Input fields
age   = st.number_input("Age (years)", min_value=1, max_value=100, value=45)
bp    = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=80)
sg    = st.number_input("Specific Gravity", min_value=1.000, max_value=1.030, value=1.020, step=0.005, format="%.3f")
al    = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=1.0)
hemo  = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=15.0)
sc    = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.2)

htn   = st.selectbox("Hypertension", ["yes", "no"])
dm    = st.selectbox("Diabetes Mellitus", ["yes", "no"])
cad   = st.selectbox("Coronary Artery Disease", ["yes", "no"])
appet = st.selectbox("Appetite", ["good", "poor"])
pc    = st.selectbox("Pus Cell", ["normal", "abnormal"])

# Predict button
if st.button("Predict CKD Risk"):
    result = predict_chronic_disease(age, bp, sg, al, hemo, sc, htn, dm, cad, appet, pc)
    st.subheader(result)
