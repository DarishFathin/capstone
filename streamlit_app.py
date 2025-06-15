import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, fitur, dan label encoders
model = joblib.load('obesity_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Mapping label hasil prediksi
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

# Input user
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (m)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=65.0)
family_history = st.selectbox("Riwayat keluarga obesitas", ["yes", "no"])
favc = st.selectbox("Sering makan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Konsumsi sayur (0–3)", 0, 3, 2)
ncp = st.slider("Jumlah makan besar per hari", 1, 5, 3)
caec = st.selectbox("Camilan antar makan", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.slider("Air minum per hari (L)", 0.0, 5.0, 2.0)
scc = st.selectbox("Pantau kalori?", ["yes", "no"])
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 5.0, 1.5)
tue = st.slider("Waktu layar (jam/hari)", 0.0, 10.0, 2.0)
calc = st.selectbox("Minum alkohol", ["no", "Sometimes", "Frequently"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Proses input
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

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    # Cek apakah label valid
    # Hanya encode kolom kategorikal
categorical_cols = list(label_encoders.keys())

for col in categorical_cols:
    val = input_df[col].iloc[0]
    if val not in label_encoders[col].classes_:
        st.error(f"Nilai '{val}' di kolom '{col}' tidak dikenali saat training.")
        st.stop()
    input_df[col] = label_encoders[col].transform(input_df[col])


    # Susun dan transform input
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    predicted_class = label_mapping.get(prediction, "Unknown")

    st.success(f"Tingkat obesitas Anda diprediksi sebagai: **{predicted_class}**")
