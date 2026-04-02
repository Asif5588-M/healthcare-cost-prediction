import pandas as pd

def preprocess_data(df):
    # Convert categorical variables to numeric
    df_encoded = pd.get_dummies(df, columns=['Gender','Diagnosis','Chronic_Disease'], drop_first=True)
    X = df_encoded.drop(['Patient_ID','Treatment_Cost'], axis=1)
    y = df_encoded['Treatment_Cost']
    return X, y