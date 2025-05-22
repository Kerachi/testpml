import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 📥 Data inladen
@st.cache_data
def load_data():
    df = pd.read_csv("electricity_data.csv")
    df = df.dropna()
    return df

df = load_data()

# 📊 Features en target
X = df.drop(columns=["Electricity (kWh)"])
y = df["Electricity (kWh)"]

# 🔁 Lineair regressiemodel trainen
model = LinearRegression()
model.fit(X, y)

# 🌐 UI
st.title("📉 Elektriciteitsvoorspeller (Linear Regression)")
st.markdown("Model traint live op de CSV en gebruikt lineaire regressie.")

# 📥 Invoer
production = st.number_input("Productie (kg)", min_value=0.0, value=8.0)
temp = st.number_input("Gemiddelde temperatuur (°C)", value=15.0)
wind = st.number_input("Windsnelheid (m/s)", value=3.0)
rain = st.number_input("Neerslag (mm)", value=2.0)
sun = st.number_input("Zonneschijn (uur)", value=5.0)

# 🔮 Voorspelling
if st.button("Voorspel Elektriciteit"):
    input_df = pd.DataFrame([[production, temp, wind, rain, sun]],
        columns=["Production (kg)", "Gemiddelde temp (°C)", 
                 "Windsnelheid (m/s)", "Neerslag (mm)", "Zonneschijn (uur)"])
    prediction = model.predict(input_df)[0]
    st.success(f"⚡ Voorspelde Elektriciteit: {prediction:.3f} kWh")
