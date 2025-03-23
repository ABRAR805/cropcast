import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ✅ Dataset path
DATASET_PATH = "C:/project/cropcast lite/datasets/crop_data.csv"

# ✅ Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: Dataset not found at {DATASET_PATH}. Check the file path.")
    exit(1)

# ✅ Load dataset
try:
    data = pd.read_csv(DATASET_PATH)
    print("✅ Dataset loaded successfully!")
except pd.errors.EmptyDataError:
    print("❌ ERROR: Dataset is empty. Add data to 'crop_data.csv'.")
    exit(1)

# ✅ Check if dataset is actually empty
if data.empty:
    print("❌ ERROR: Dataset is empty. Add data to 'crop_data.csv'.")
    exit(1)

# ✅ Encode categorical features (including crops)
label_encoders = {}
categorical_columns = ["water", "soil", "state", "district", "country", "crop1", "crop2", "crop3"]

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# ✅ Define features (X) and target variable (y)
X = data[["land", "water", "soil", "state", "district", "country"]]
y = data[["crop1", "crop2", "crop3"]]  # Only training for crop predictions

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the crop prediction model
crop_model = RandomForestRegressor(n_estimators=100, random_state=42)
crop_model.fit(X_train, y_train)

# ✅ Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ✅ Save trained model
with open("models/crop_prediction_model.pkl", "wb") as file:
    pickle.dump(crop_model, file)

# ✅ Save label encoders
with open("models/label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

print("✅ Model training complete. Saved as 'crop_prediction_model.pkl' and 'label_encoders.pkl'.")
