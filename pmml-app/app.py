import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🔋 Elektriciteitsverbruik voorspellen (op basis van weer + productie)")

@st.cache_data
def load_data():
    return pd.read_excel("streemlit_cleaned.xlsx")

# Data inladen en voorbereiden
df = load_data()
X = df[["Gemiddelde temp (°C)", "Windsnelheid (m/s)", "Neerslag (mm)", "Zonneschijn (uur)", "Production (kg)"]]
y = df["Prediction Electricity (kWh)"]

# Model trainen
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("⚙️ Stel weersomstandigheden + productie in")

# Gebruikersinvoer via sliders
input_data = {
    "Gemiddelde temp (°C)": st.slider("🌡️ Temperatuur", float(X.min()[0]), float(X.max()[0]), float(X.mean()[0])),
    "Windsnelheid (m/s)": st.slider("💨 Windsnelheid", float(X.min()[1]), float(X.max()[1]), float(X.mean()[1])),
    "Neerslag (mm)": st.slider("🌧️ Neerslag", float(X.min()[2]), float(X.max()[2]), float(X.mean()[2])),
    "Zonneschijn (uur)": st.slider("☀️ Zonneschijn", float(X.min()[3]), float(X.max()[3]), float(X.mean()[3])),
    "Production (kg)": st.slider("🥬 Productie", float(X.min()[4]), float(X.max()[4]), float(X.mean()[4])),
}

input_df = pd.DataFrame([input_data])

if st.button("⚡ Voorspel elektriciteit"):
    prediction = model.predict(input_df)[0]
    st.success(f"🔌 Verwachte elektriciteit: {prediction:.2f} kWh")
