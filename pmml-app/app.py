import streamlit as st
from pypmml import Model
import pandas as pd

st.title("🔮 PMML Voorspeller (RapidMiner model)")

# Laad model
model = Model.load("test.pmml")

# Voorbeeld: verander invoervelden naar wat jouw model verwacht
temperature = st.number_input("Temperatuur (°C)", value=20.0)
sunlight = st.number_input("Zonuren", value=5.0)
humidity = st.number_input("Luchtvochtigheid (%)", value=60.0)

# Input als DataFrame
data = pd.DataFrame([{
    "Temperature": temperature,
    "Sunlight": sunlight,
    "Humidity": humidity
}])

# Voorspel knop
if st.button("Voorspel"):
    prediction = model.predict(data)
    st.success(f"Voorspelling: {prediction.iloc[0, -1]}")
