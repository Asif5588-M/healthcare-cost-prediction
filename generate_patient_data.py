# generate_patient_data.py
import pandas as pd
import numpy as np

# Number of rows
n_rows = 1000

# Seed for reproducibility
np.random.seed(42)

# Generate Patient_ID
patient_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_rows+1)]

# Age: 0-90
ages = np.random.randint(0, 91, n_rows)

# Gender: Male/Female
genders = np.random.choice(['Male', 'Female'], n_rows)

# Diagnosis: 5 types
diagnoses = np.random.choice(['Flu', 'Diabetes', 'Hypertension', 'Cardiac', 'Orthopedic'], n_rows)

# Chronic_Disease: Yes/No (Probability: 30% Yes)
chronic = np.random.choice(['Yes', 'No'], n_rows, p=[0.3, 0.7])

# Previous_Visits: 0-20
prev_visits = np.random.randint(0, 21, n_rows)

# Treatment_Cost: Base cost + risk factors
# Base cost 1000-5000, increase for chronic disease and more visits
base_cost = np.random.randint(1000, 5001, n_rows)
treatment_cost = base_cost + (prev_visits * 200) + np.where(chronic=='Yes', 2000, 0)

# Create DataFrame
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Age': ages,
    'Gender': genders,
    'Diagnosis': diagnoses,
    'Chronic_Disease': chronic,
    'Previous_Visits': prev_visits,
    'Treatment_Cost': treatment_cost
})

# Save to CSV
df.to_csv('data/patient_data.csv', index=False)

print("patient_data.csv with 1000 rows created successfully in 'data/' folder!")