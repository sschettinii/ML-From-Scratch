import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load and clean housing dataset
df_housing = pd.read_csv("data/housing.csv")
df_housing.drop_duplicates(inplace=True)
df_housing.dropna(inplace=True)

target_col = 'price' 

numeric_cols = df_housing.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df_housing[col].quantile(0.25)
    Q3 = df_housing[col].quantile(0.75)
    IQR = Q3 - Q1
    
    df_housing = df_housing[(df_housing[col] >= (Q1 - 1.5 * IQR)) & (df_housing[col] <= (Q3 + 1.5 * IQR))]

df_housing = pd.get_dummies(df_housing, drop_first=True, dtype=float)

X = df_housing.drop(columns=[target_col]).values
y = df_housing[target_col].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

X_train_housing = X_scaled.T
y_train_housing = y_scaled.reshape(1, -1)

def get_mse(predictions, y):
    n = len(predictions)
    mse = (1 / n) * np.sum(predictions - y)
    return mse

def estimate_betas(X_train, y_train, method):
    match method:
        case "MLE":
            betas = 1
            return betas
        case "OLS":
            betas = ((X_train.T * X_train) ** -1) * X_train.T * y_train
            return betas
    
    return betas

def get_predictions(X_train, i, b):
    predictions = X_train * b + i
    return predictions

# OLS -> Ordinary Least Squares 
# MLE -> Maximum Likelihood Estimation
def train_linear_regression(X_train, y_train, method="MLE"):
    betas = estimate_betas(X_train, y_train, method)
    predictions = get_predictions(X_train, betas)
    return betas

train_linear_regression(X_train_housing, y_train_housing)