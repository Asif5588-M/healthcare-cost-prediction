from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Healthcare Cost Prediction API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Healthcare Cost Prediction API is Live!"}

class PatientData(BaseModel):
    Gender: int
    Visit_Count: int
    Test_Freq: int
    Test_Diversity: int
    Repeat_Test_Count: int
    num_BLOOD_tests: int
    num_RADIOLOGY_tests: int
    num_URINE_tests: int
    num_STOOL_tests: int
    num_SPUTUM_tests: int
    Monthly_Test_Trend: str
    Moral_Hazard_Index: int

@app.post("/predict")
def predict(data: PatientData):
    trend_encoded = le.transform([data.Monthly_Test_Trend])[0]
    
    features = np.array([[
        data.Gender, data.Visit_Count, data.Test_Freq,
        data.Test_Diversity, data.Repeat_Test_Count,
        data.num_BLOOD_tests, data.num_RADIOLOGY_tests,
        data.num_URINE_tests, data.num_STOOL_tests,
        data.num_SPUTUM_tests, trend_encoded,
        data.Moral_Hazard_Index
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        "prediction": int(prediction),
        "result": "High Cost Utilizer" if prediction == 1 else "Normal Utilizer",
        "confidence": round(float(max(probability)) * 100, 2)
    }