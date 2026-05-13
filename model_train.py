import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Data load karo
df = pd.read_csv("final_ml_ready_health_data.csv")

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nTarget distribution:")
print(df['High_Cost_Utilizer'].value_counts())

# Monthly_Test_Trend encode karo
le = LabelEncoder()
df['Monthly_Test_Trend'] = le.fit_transform(df['Monthly_Test_Trend'])

# Features aur target
X = df.drop('High_Cost_Utilizer', axis=1)
y = df['High_Cost_Utilizer']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, lr_pred) * 100, 2), "%")

print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")

print("\n--- SVM ---")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, svm_pred) * 100, 2), "%")

# Best model save karo (Random Forest)
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)

# LabelEncoder bhi save karo
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nBest model (Random Forest) saved!")