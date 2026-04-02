import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load predictions
df = pd.read_csv("outputs/predictions.csv")

# Create figures folder
os.makedirs("outputs/figures", exist_ok=True)

# Distribution of predicted cost
plt.figure(figsize=(8,5))
sns.histplot(df['Predicted_Cost'], bins=30, kde=True, color='skyblue')
plt.title("Predicted Treatment Cost Distribution")
plt.xlabel("Treatment Cost")
plt.ylabel("Frequency")
plt.savefig("outputs/figures/treatment_cost_distribution.png")
plt.close()

# Chronic Disease vs Predicted Cost
plt.figure(figsize=(6,4))
sns.boxplot(x='Chronic_Disease', y='Predicted_Cost', data=df)
plt.title("Chronic Disease vs Predicted Treatment Cost")
plt.savefig("outputs/figures/chronic_vs_cost.png")
plt.close()

# Diagnosis vs Predicted Cost
plt.figure(figsize=(8,5))
sns.boxplot(x='Diagnosis', y='Predicted_Cost', data=df)
plt.title("Diagnosis vs Predicted Treatment Cost")
plt.savefig("outputs/figures/diagnosis_vs_cost.png")
plt.close()

print("Figures saved to outputs/figures/")