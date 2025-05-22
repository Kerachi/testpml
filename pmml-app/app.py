import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Data inladen
@st.cache_data
def load_data():
    df = pd.read_csv("electricity_data.csv")
    return df

df = load_data()

# Splitsen in features en target
X = df.drop(columns=["Electricity (kWh)"])
y = df["Electricity (kWh)"]

# Model trainen
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# UI
st.title("🔌 Elektriciteitsverbruik Voorspeller")
st.markdown("Train het model automatisch op basis van de CSV – geen pkl nodig!")

# Invoer
production = st.number_input("Productie (kg)", min_value=0.0, value=8.0)
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=3.0)
rain = st.number_input("Neerslag (mm)", value=2.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

# Voorspelling
if st
