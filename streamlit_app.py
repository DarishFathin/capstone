import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('obesity_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🎯 Prediksi Tingkat Obesitas Berdasarkan Data Pribadi")

st.markdown("Silakan masukkan informasi berikut:")

gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (m)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=65.0)
family_history = st.selectbox("Riwayat keluarga kelebihan berat badan", ["yes", "no"])
favc = st.selectbox("Sering makan makanan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Makan sayur tiap makan (skala 0–3)", 0, 3, 2)
ncp = st.slider("Makan besar per hari", 1, 5, 3)
caec = st.selectbox("Makan camilan di antara waktu makan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Apakah merokok?", ["yes", "no"])
ch2o = st.slider("Air minum per hari (liter)", 0.0, 3.0, 1.5)
scc = st.selectbox("Mengontrol asupan kalori harian?", ["yes", "no"])
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 5.0, 1.5)
tue = st.slider("Waktu pakai teknologi (jam/hari)", 0.0, 5.0, 2.0)
calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Jenis transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

if st.button("Prediksi"):
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })

    # Encode kategorikal ke numerik
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype('category').cat.codes

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)
    st.success(f"Prediksi tingkat obesitas: **{prediction[0]}**")
