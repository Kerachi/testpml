import streamlit as st
import pandas as pd
import joblib

# Laad het model
model = joblib.load("model.pkl")

st.title("🔌 Elektriciteitsverbruik Voorspeller")

st.markdown("Voer hieronder de waarden in:")

# Invoervelden
production = st.number_input("Productie (kg)", min_value=0.0, value=8.0)
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=3.0)
rain = st.number_input("Neerslag (mm)", value=2.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

# Voorspelling
if st.button("Voorspel Elektriciteit"):
    input_df = pd.DataFrame([[production, temp, wind, rain, sun]],
        columns=["Production (kg)", "Gemiddelde temp (°C)",
                 "Windsnelheid (m/s)", "Neerslag (mm)", "Zonneschijn (uur)"])
    prediction = model.predict(input_df)[0]
    st.success(f"⚡ Voorspelde Elektriciteit: {prediction:.2f} kWh")
