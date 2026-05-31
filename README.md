# 🏥 Healthcare Cost Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![ML](https://img.shields.io/badge/ML-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A machine learning system that predicts patient treatment costs based on demographics, diagnosis history, and chronic disease profiles — enabling hospitals and insurers to optimize resource allocation and financial planning.

---

## 🎯 Problem Statement

Healthcare cost management is one of the biggest challenges facing hospitals and insurance providers. Unpredictable treatment costs lead to budget overruns and poor patient care planning. This project builds a **predictive model** that estimates treatment costs **before** a patient undergoes treatment — helping administrators make data-driven decisions.

---

## 📊 Key Results

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Random Forest | ~0.87 | ~$1,200 | ~$2,100 |
| Baseline (Mean) | 0.00 | ~$4,800 | ~$6,300 |

> ✅ Random Forest outperforms the baseline by **~75%** in prediction accuracy.

---

## 🧠 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age |
| Gender | Categorical | Male / Female |
| Diagnosis | Categorical | Primary diagnosis category |
| Chronic Diseases | Numeric | Number of chronic conditions |
| Hospital Stay (Days) | Numeric | Duration of stay |
| Region | Categorical | Geographic region |

---

## 📂 Project Structure

```
healthcare-cost-prediction/
│
├── data/
│   └── patient_data.csv          # Simulated patient dataset
│
├── notebooks/
│   └── eda_and_model.ipynb       # Full EDA + Model Training notebook
│
├── scripts/
│   ├── data_preprocessing.py     # Cleaning, encoding, feature engineering
│   ├── model_training.py         # Train & evaluate Random Forest
│   ├── prediction.py             # Predict cost for new patients
│   └── visualization.py         # Generate EDA plots
│
├── outputs/
│   ├── trained_model.pkl         # Saved trained model
│   ├── predictions.csv           # Sample predictions
│   └── figures/                  # All visualizations
│
├── Dockerfile                    # Docker support for deployment
├── requirements.txt              # Python dependencies
├── generate_patient_data.py      # Synthetic data generator
├── main.py                       # Entry point
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Asif5588-M/healthcare-cost-prediction.git
cd healthcare-cost-prediction
```

### 2. Set Up Environment
```bash
conda create -n healthcare_env python=3.10
conda activate healthcare_env
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
# Generate data
python generate_patient_data.py

# Train model
python scripts/model_training.py

# Make predictions
python scripts/prediction.py
```

### 4. Or Use Docker
```bash
docker build -t healthcare-cost .
docker run healthcare-cost
```

---

## 📈 Visualizations

### Chronic Diseases vs Treatment Cost
Higher number of chronic conditions correlates strongly with increased treatment costs — the model captures this non-linear relationship effectively.

### Feature Importance (Random Forest)
Top predictors of treatment cost:
1. Duration of Hospital Stay
2. Number of Chronic Diseases
3. Diagnosis Type
4. Age

---

## 🛠 Tech Stack

- **Language:** Python 3.10
- **ML:** Scikit-learn (Random Forest Regressor)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Docker
- **Notebook:** Jupyter

---

## 💡 Business Use Cases

- 🏦 **Insurance Companies** — Estimate premiums based on predicted costs
- 🏥 **Hospitals** — Plan budget and staffing based on patient profiles
- 📋 **Healthcare Researchers** — Understand cost drivers in patient populations

---

## 👤 About the Author

**Asif Malik**
MPhil Economics | Machine Learning Practitioner
Passionate about applying data science to solve real-world healthcare and economic problems.

📧 Connect: [GitHub](https://github.com/Asif5588-M)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
