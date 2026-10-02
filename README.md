# 🌦 Thunderstorm (TH) Forecasting System

A machine learning–based application for **Thunderstorm (TH) occurrence prediction** using atmospheric indices.  
The project uses a **pre-trained & compressed Random Forest model** and provides an **interactive Streamlit web interface** for real-time predictions.

---

## 🚀 Features

- ✅ Pre-trained **Random Forest Classifier**
- ✅ Model size optimized using **Joblib compression**
- ✅ Interactive **Streamlit UI**
- ✅ No retraining required (inference-only)
- ✅ Ready for **Docker** and **Cloud deployment (Render / AWS)**
- ✅ Modular & production-ready project structure

---

## 📊 Input Features

The model predicts thunderstorm occurrence using the following atmospheric parameters:

- SWEAT Index  
- K Index  
- Totals Totals Index  
- Environmental Stability  
- Moisture Indices  
- Convective Potential  
- Temperature Pressure  
- Moisture Temperature Profiles  

---

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier  
- **Training**: Offline (not included in this repo)  
- **Class Imbalance Handling**: SMOTE  
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Probability of Detection (POD)
  - False Alarm Rate (FAR)
  - Heidke Skill Score (HSS)
  - Critical Success Index (CSI)

- **Model Format**: `joblib`  
- **Compressed Size**: ~5–10 MB  

---




## Run locally

Install the runtime dependencies into the project environment, then start the API and frontend in separate terminals from the project root:

```powershell
uv sync
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
uv run streamlit run streamlit_app/ui.py
```

The API serves `POST /predict`, `GET /`, and `GET /health`. The Streamlit app uses `http://127.0.0.1:8000` by default; set `API_URL` to a different API base URL when needed. The model is loaded from `models/KNN_best_model.pkl` relative to the project, regardless of the current working directory.
