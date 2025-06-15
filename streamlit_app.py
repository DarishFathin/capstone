import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, dan urutan fitur
model = joblib.load('obesity_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Mapping label target ke nama kelas (disesuaikan dari training)
label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

st.title("🎯 Prediksi Tingkat Obesitas")

# Form input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (m)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=65.0)
family_history = st.selectbox("Riwayat keluarga obesitas", ["yes", "no"])
favc = st.selectbox("Sering makan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Konsumsi sayur per makan (skala 0-3)", 0, 3, 2)
ncp = st.slider("Jumlah makan besar per hari", 1, 5, 3)
caec = st.selectbox("Camilan antar makan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.slider("Air minum per hari (liter)", 0.0, 5.0, 2.0)
scc = st.selectbox("Pantau kalori?", ["yes", "no"])
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 5.0, 1.5)
tue = st.slider("Durasi pakai teknologi (jam/hari)", 0.0, 10.0, 2.0)
calc = st.selectbox("Minum alkohol?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Buat DataFrame input
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

# Encode input kategorikal (gunakan pendekatan saat training)
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        input_df[col] = input_df[col].astype('category').cat.codes

# 🔧 Susun ulang input agar cocok dengan fitur saat training
input_df = input_df[feature_names]

# Transformasi dengan scaler
input_scaled = scaler.transform(input_df)

# Prediksi & mapping
prediction = model.predict(input_scaled)[0]
predicted_class = label_mapping.get(prediction, "Tidak diketahui")

st.success(f"Tingkat obesitas Anda diprediksi sebagai: **{predicted_class}**")
