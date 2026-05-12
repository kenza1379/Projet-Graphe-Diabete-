import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/diabetes_data_cleaned.csv")

cols_projet = [
    'Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
    'SystolicBP', 'CholesterolTotal', 'SleepQuality'
]

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(df[cols_projet]),
    columns=cols_projet
)

X_scaled.to_csv("../data/diabetes_data_normalized.csv", index=False)