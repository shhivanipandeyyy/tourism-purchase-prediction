import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import xgboost as xgb

# Load dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Define target and features
target_col = 'ProdTaken'
X = df.drop(columns=[target_col, 'CustomerID'])
y = df[target_col]

# Numeric and categorical features
numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                    'PreferredPropertyStar', 'NumberOfTrips', 'MonthlyIncome', 
                    'NumberOfChildrenVisiting', 'PitchSatisfactionScore']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Model pipeline
xgb_model = xgb.XGBClassifier(random_state=42)
pipeline = make_pipeline(preprocessor, xgb_model)

# Train the model
pipeline.fit(X, y)

# Save the trained model
joblib.dump(pipeline, "tourism_project/model_building/tourism_xgb_model.pkl")
print("Model saved successfully!")
