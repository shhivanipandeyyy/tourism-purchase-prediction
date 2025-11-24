import joblib

# Simple path since the model is in the same folder
model_path = "tourism_xgb_model.pkl"

# Load the model
model = joblib.load(model_path)
