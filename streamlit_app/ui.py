import os

import streamlit as st  # frontend UI
import requests

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/predict"

# fast api runs

st.title("🌦 Weather Thunderstorm Prediction App")
st.write("Enter atmospheric parameters to predict TH (Thunderstorm Occurrence)")

# Input fields
SWEAT_index = st.number_input("SWEAT Index")
K_index = st.number_input("K Index")
Totals_totals_index = st.number_input("Totals Totals Index")
Environmental_Stability = st.number_input("Environmental Stability")
Moisture_Indices = st.number_input("Moisture Indices")
Convective_Potential = st.number_input("Convective Potential")
Temperature_Pressure = st.number_input("Temperature Pressure")
Moisture_Temperature_Profiles = st.number_input("Moisture Temperature Profiles")

if st.button("Predict"):
    payload = {
        "SWEAT_index": SWEAT_index,
        "K_index": K_index,
        "Totals_totals_index": Totals_totals_index,
        "Environmental_Stability": Environmental_Stability,
        "Moisture_Indices": Moisture_Indices,
        "Convective_Potential": Convective_Potential,
        "Temperature_Pressure": Temperature_Pressure,
        "Moisture_Temperature_Profiles": Moisture_Temperature_Profiles,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
        if result.get("probability") is not None:
            st.info(f"Probability: {result['probability']:.4f}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to the prediction API at {API_BASE_URL}. "
            "Start it with `uvicorn api.main:app --host 127.0.0.1 --port 8000`."
        )
    except requests.exceptions.Timeout:
        st.error("The prediction API took too long to respond. Please try again.")
    except requests.exceptions.RequestException as exc:
        st.error(f"The prediction API returned an error: {exc}")
    except (ValueError, KeyError) as exc:
        st.error(f"The prediction API returned an invalid response: {exc}")
