import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🔌 Elektriciteit Voorspeller op Basis van Weer")

# 1. Laad dataset vanuit GitHub repo
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Kerachi/testpml/main/pmml-app/elektriciteit_trainingsdata.xlsx"
    return pd.read_excel(url)

df = load_data()

# 2. Splits data
X = df.drop(columns="Prediction (kWh)")
y = df["Prediction (kWh)"]

# 3. Train model in de app zelf (100% compatibel)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Gebruikersinvoer
st.subheader("📥 Voer weersgegevens in")
input_data = {
    "Gemiddelde temp (°C)": st.number_input("Gemiddelde temperatuur", value=10.0),
    "Windsnelheid (m/s)": st.number_input("Windsnelheid", value=4.0),
    "Neerslag (mm)": st.number_input("Neerslag", value=3.0),
    "Zonneschijn (uur)": st.number_input("Zonneschijn", value=5.0)
}

input_df = pd.DataFrame([input_data])

# 5. Voorspelling
if st.button("⚡ Voorspel Elektriciteit"):
    prediction = model.predict(input_df)[0]
    st.success(f"🔋 Voorspelde elektriciteit: {prediction:.2f} kWh")
