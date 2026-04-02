import pandas as pd
import pickle

# Load trained model
with open("outputs/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load patient data (new or same data)
df_new = pd.read_csv("data/patient_data.csv")

# Preprocess same as training
df_encoded = pd.get_dummies(df_new, columns=['Gender','Diagnosis','Chronic_Disease'], drop_first=True)
X_new = df_encoded.drop(['Patient_ID','Treatment_Cost'], axis=1)

# Predict treatment cost
df_new['Predicted_Cost'] = model.predict(X_new)

# Save predictions
df_new.to_csv("outputs/predictions.csv", index=False)
print("Predictions saved to outputs/predictions.csv")