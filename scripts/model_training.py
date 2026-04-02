import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load Data
data_path = "data/patient_data.csv"  # relative path
df = pd.read_csv(data_path)

# Preprocessing
df_encoded = pd.get_dummies(df, columns=['Gender','Diagnosis','Chronic_Disease'], drop_first=True)
X = df_encoded.drop(['Patient_ID','Treatment_Cost'], axis=1)
y = df_encoded['Treatment_Cost']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Random Forest RMSE: {rmse:.2f}")
print(f"Random Forest R2 Score: {r2:.2f}")

# Save trained model
os.makedirs("outputs", exist_ok=True)
with open("outputs/trained_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Trained model saved to outputs/trained_model.pkl")