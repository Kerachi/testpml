import streamlit as st
import pandas as pd
import joblib

st.title("🌱 Voorspelling Elektriciteit, Productie en CO₂")

# Laad model
model = joblib.load("random_forest_model_py39.joblib")

# Invoer
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=4.0)
rain = st.number_input("Neerslag (mm)", value=3.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

# Maak DataFrame
df = pd.DataFrame([{
    "Gemiddelde temp (°C)": temp,
    "Windsnelheid (m/s)": wind,
    "Neerslag (mm)": rain,
    "Zonneschijn (uur)": sun
}])

# Voorspellen
if st.button("Voorspel"):
    prediction = model.predict(df)[0]
    st.write(f"🔌 Elektriciteit (kWh): {prediction[0]:.2f}")
    st.write(f"🥬 Productie (kg): {prediction[1]:.2f}")
    st.write(f"🌫️ CO₂-uitstoot (kg): {prediction[2]:.2f}")
