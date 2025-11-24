import joblib
import os

# Path from deployment/ folder to model_building/
model_path = "../model_building/tourism_xgb_model.pkl"
model = joblib.load(model_path)
print("Model loaded successfully!")
