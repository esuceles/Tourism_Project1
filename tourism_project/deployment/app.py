import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="celesqa/GL-Tourism", filename="best_tourism_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing the newly launched Wellness Tourism Package.
Please enter the data below to get a prediction.
""")

# User input
CustomerID=st.number_input("Customer ID", min_value=1, max_value=99999999)
Age=st.number_input("Age", min_value=18, max_value=100)
TypeofContact=st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier=st.selectbox("City Tier", ["1", "2", "3"])
DurationOfPitch=st.number_input("Duration of Pitch", min_value=1, max_value=1000)
Occupation=st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender=st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting=st.number_input("Number of Person Visiting", min_value=1, max_value=10)
NumberOfFollowups=st.number_input("Number of Followups", min_value=1, max_value=10)
ProductPitched=st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe","King","Super Deluxe"])
PreferredPropertyStar=st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus=st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmariied"])
NumberOfTrips=st.number_input("Number of Trips", min_value=1, max_value=100)
Passport=st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore=st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10)
OwnCar=st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting=st.number_input("Number of Children Visiting", min_value=0, max_value=10)
Designation=st.selectbox("Designation", ["Manager", "AVP", "VP","Senior Manager","Executive"])
MonthlyIncome=st.number_input("Monthly Income", min_value=1000, max_value=100000)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    CustomerID='CustomerID',
    Age='Age',
    TypeofContact='TypeofContact',
    CityTier='CityTier',
    DurationOfPitch='DurationOfPitch',
    Occupation='Occupation',
    Gender='Gender',
    NumberOfPersonVisiting='NumberOfPersonVisiting',
    NumberOfFollowups='NumberOfFollowups',
    ProductPitched='ProductPitched',
    PreferredPropertyStar='PreferredPropertyStar',
    MaritalStatus='MaritalStatus',
    NumberOfTrips='NumberOfTrips',
    Passport='Passport' 1 if Passport == "Yes" else 0,
    PitchSatisfactionScore='PitchSatisfactionScore',
    OwnCar='OwnCar'1 if OwnCar == "Yes" else 0 ,
    NumberOfChildrenVisiting='NumberOfChildrenVisiting',
    Designation='Designation',
    MonthlyIncome='MonthlyIncome'
}])


if st.button("Predict Customer"):
    prediction = model.predict(input_data)[0]
    result = "Wellness Tourism Package is purchased" if prediction == 1 else "Wellness Tourism Package is not purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
