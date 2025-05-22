
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

st.title("🌱 Elektriciteit, Productie & CO₂ Voorspeller")

# Laad dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Kerachi/testpml/main/Voorspellingen_per_Periode%20(1).xlsx"
    return pd.read_excel(url)

df = load_data()

# Selecteer features en targets
features = ['Gemiddelde temp (°C)', 'Windsnelheid (m/s)', 'Neerslag (mm)', 'Zonneschijn (uur)']
targets = ['Prediction Electricity (kWh)', 'Prediction Production  (kg)', 'Prediction Co2 (kg) ']
df = df[features + targets].dropna()

# Splits input/output
X = df[features]
y = df[targets]

# Train model in de app
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X, y)

# Gebruikersinvoer
st.subheader("📥 Voer weersgegevens in")
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=4.0)
rain = st.number_input("Neerslag (mm)", value=3.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

input_df = pd.DataFrame([{
    "Gemiddelde temp (°C)": temp,
    "Windsnelheid (m/s)": wind,
    "Neerslag (mm)": rain,
    "Zonneschijn (uur)": sun
}])

# Voorspelling
if st.button("Voorspel"):
    prediction = model.predict(input_df)[0]
    st.success(f"🔌 Elektriciteit (kWh): {prediction[0]:.2f}")
    st.success(f"🥬 Productie (kg): {prediction[1]:.2f}")
    st.success(f"🌫️ CO₂-uitstoot (kg): {prediction[2]:.2f}")
