import streamlit as st
import pandas as pd
from pypmml import Model

# Laad het PMML-model
model = Model.load('tesr3.pmml')

# UI
st.title("📈 Elektriciteitsvoorspeller via Linear Regression (PMML)")
st.markdown("Model geladen uit `tesr3.pmml`")

# Invoer (gebruik exact dezelfde kolomnamen als je model verwacht)
production = st.number_input("Productie (kg)", value=8.0)
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=3.0)
rain = st.number_input("Neerslag (mm)", value=2.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

if st.button("Voorspel Elektriciteit"):
    # DataFrame maken met invoerwaarden
    input_data = pd.DataFrame([{
        "Production (kg)": production,
        "Gemiddelde temp (°C)": temp,
        "Windsnelheid (m/s)": wind,
        "Neerslag (mm)": rain,
        "Zonneschijn (uur)": sun
    }])

    # Voorspellen met het PMML-model
    prediction = model.predict(input_data)
    resultaat = prediction["predicted_Electricity (kWh)"].iloc[0]  # Naam kan verschillen!

    st.success(f"⚡ Voorspelde Elektriciteit: {resultaat:.3f} kWh")
