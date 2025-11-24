import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load the trained model
# ---------------------------
model_path = "tourism_xgb_model.pkl"
model = joblib.load(model_path)

st.title("Tourism Purchase Prediction")
st.write("Predict whether a customer will take the tourism product.")

# ---------------------------
# User Input
# ---------------------------
def user_input_features():
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=20, value=1)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=30000)
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    Occupation = st.text_input("Occupation", "Salaried")
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ProductPitched = st.selectbox("Product Pitched", ["Product1", "Product2", "Product3"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Designation = st.text_input("Designation", "Manager")

    data = {
        'Age': Age,
        'DurationOfPitch': DurationOfPitch,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'PreferredPropertyStar': PreferredPropertyStar,
        'NumberOfTrips': NumberOfTrips,
        'MonthlyIncome': MonthlyIncome,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'TypeofContact': TypeofContact,
        'Occupation': Occupation,
        'Gender': Gender,
        'ProductPitched': ProductPitched,
        'MaritalStatus': MaritalStatus,
        'Designation': Designation
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# ---------------------------
# Prediction
# ---------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)[:, 1]

st.subheader("Prediction")
if prediction[0] == 1:
    st.write("✅ Customer is likely to purchase the product.")
else:
    st.write("❌ Customer is unlikely to purchase the product.")

st.subheader("Prediction Probability")
st.write(prediction_proba[0])
