
  import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import joblib

# -------------------------
# Step 0: Download dataset from GitHub (Fix for FileNotFoundError)
# -------------------------
!wget -q https://raw.githubusercontent.com/shhivanipandeyyy/tourism-purchase-prediction/main/tourism_project/data/tourism.csv -O tourism.csv
print("Dataset downloaded from GitHub successfully.")

# -------------------------
# Step 1: Load Tourism Dataset
# -------------------------
df = pd.read_csv("tourism.csv")
print("Tourism dataset loaded successfully.")

# -------------------------
# Step 2: Encode categorical column(s)
# -------------------------
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])

# -------------------------
# Step 3: Define target column and drop unique ID
# -------------------------
target_col = 'ProdTaken'
X = df.drop(columns=[target_col, 'CustomerID'])
y = df[target_col]

# -------------------------
# Step 4: Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Step 5: Define numeric & categorical features
# -------------------------
numeric_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'MonthlyIncome',
    'NumberOfChildrenVisiting', 'PitchSatisfactionScore'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
    'MaritalStatus', 'Designation'
]

# -------------------------
# Step 6: Preprocessing & Pipeline
# -------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(random_state=42)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------
# Step 7: Train the model
# -------------------------
model_pipeline.fit(X_train, y_train)
print("Model trained successfully.")

# -------------------------
# Step 8: Save the trained model
# -------------------------
os.makedirs("tourism_project/model_building", exist_ok=True)

model_path = "tourism_project/model_building/tourism_xgb_model.pkl"
joblib.dump(model_pipeline, model_path)

print(f"Model saved at: {model_path}")   kahan paste kro TRAIN.PY


