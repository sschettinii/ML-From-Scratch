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