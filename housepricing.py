import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor 
import joblib
import os

# Load the dataset
data = pd.read_csv('data/houseprices.csv')

# Explore the dataset
print(data.head())
print(data.info())

# Check for non-numeric columns explicitly
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Non-numeric column found: {col}")
        print(data[col].unique())  # Show unique values of the non-numeric column

# Separate columns into numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(exclude=[np.number]).columns

# Impute missing data using KNNImputer for numeric columns
imputer = KNNImputer(n_neighbors=5)
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Encode categorical variables using pd.get_dummies (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)

# Ensure all data is numeric
for col in data.columns:
    if not np.issubdtype(data[col].dtype, np.number):
        print(f"Converting column '{col}' to numeric.")
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Handle any NaN values after conversion (if any columns were coerced into NaN)
data.fillna(0, inplace=True)


X = data.drop(columns=['AveragePrice'])  # Drop the target column
y = data['AveragePrice']                # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training the data by combining predictions from multiple models for better performance
ensemble = VotingRegressor(estimators=[
    ('rf', RandomForestRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42))
])
ensemble.fit(X_train, y_train)


# Ensure the 'models' directory exists
# Define a new path since did'nt have permision
save_dir = 'C:/models'
os.makedirs(save_dir, exist_ok=True)

# Save the model
joblib.dump(ensemble, os.path.join(save_dir, 'house_price_model.pkl'))


# Load the saved model
ensemble = joblib.load('C:/models/house_price_model.pkl')

# Predict on test data
y_pred = ensemble.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"Mean Squared Error: {rmse}")