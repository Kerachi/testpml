import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 📥 Data inladen
@st.cache_data
def load_data():
    df = pd.read_csv("electricity_data.csv")
    df = df.dropna()
    return df

df = load_data()

# 📊 Features en target kiezen
X = df.drop(columns=["Prediction Electricity (kWh)", "Electricity (kWh)"])
y = df["Prediction Electricity (kWh)"]

# 🎯 Model trainen met dezelfde instellingen
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
model.fit(X, y)

# 🖼️ Streamlit UI
st.title("🔌 Elektriciteitsverbruik Voorspeller")
st.markdown("Model getraind op 'Prediction Electricity (kWh)' uit je dataset.")

# 📥 Invoer van gebruiker
production = st.number_input("Productie (kg)", min_value=0.0, value=8.0)
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=3.0)
rain = st.number_input("Neerslag (mm)", value=2.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

# 🔮 Voorspelling uitvoeren
if st.button("Voorspel Elektriciteit"):
    input_df = pd.DataFrame([[production, temp, wind, rain, sun]],
        columns=["Production (kg)", "Gemiddelde temp (°C)", 
                 "Windsnelheid (m/s)", "Neerslag (mm)", "Zonneschijn (uur)"])
    prediction = model.predict(input_df)[0]
    st.success(f"⚡ Voorspelde Elektriciteit: {prediction:.3f} kWh")

    # 🔍 Toon oorspronkelijke waarde uit dataset (optioneel)
    match = df[
        (df["Production (kg)"] == production) &
        (df["Gemiddelde temp (°C)"] == temp) &
        (df["Windsnelheid (m/s)"] == wind) &
        (df["Neerslag (mm)"] == rain) &
        (df["Zonneschijn (uur)"] == sun)
    ]
    if not match.empty:
        original = match.iloc[0]["Prediction Electricity (kWh)"]
        st.info(f"📊 Oorspronkelijke voorspelling in dataset: {original:.3f} kWh")
