# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model (Pipeline with preprocessing + classifier)
model = joblib.load("client_retention_model.pkl")

st.title("🔄 Client Retention Predictor with SHAP")

st.write("Predict whether a client is likely to return based on their profile and understand why.")

# Input form
with st.form("prediction_form"):
    contact_method = st.selectbox("Contact Method", ['phone', 'email', 'in-person'])
    household = st.selectbox("Household Type", ['single', 'family'])
    preferred_language = st.selectbox("Preferred Language", ['english', 'other'])
    sex = st.selectbox("Sex", ['male', 'female'])
    status = st.selectbox("Status", ['new', 'returning', 'inactive'])
    season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
    month = st.selectbox("Month", [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    latest_lang_english = st.selectbox("Latest Language is English", ['yes', 'no'])

    age = st.slider("Age", 18, 100, 35)
    dependents_qty = st.number_input("Number of Dependents", 0, 10, 1)
    distance_km = st.number_input("Distance to Location (km)", 0.0, 50.0, 5.0)
    num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)

    submitted = st.form_submit_button("Predict")

# Prediction & SHAP
if submitted:
    # Prepare input DataFrame
    input_df = pd.DataFrame([{
        'contact_method': contact_method,
        'household': household,
        'preferred_languages': preferred_language,
        'sex_new': sex,
        'status': status,
        'Season': season,
        'Month': month,
        'latest_language_is_english': latest_lang_english,
        'age': age,
        'dependents_qty': dependents_qty,
        'distance_km': distance_km,
        'num_of_contact_methods': num_of_contact_methods
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Client is likely to return (Probability: {round(probability, 2)})")
    else:
        st.warning(f"⚠️ Client may not return (Probability: {round(probability, 2)})")

    # SHAP Explanation
    st.markdown("---")
    st.subheader("🔍 SHAP Explanation (Why this prediction?)")

    # Extract model and preprocessing
    model_only = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessing']

    # Transform input
    transformed_input = preprocessor.transform(input_df)

    # Get feature names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(
        ['contact_method', 'household', 'preferred_languages', 'sex_new', 'status', 'Season', 'Month', 'latest_language_is_english']
    )
    feature_names = list(cat_features) + ['age', 'dependents_qty', 'distance_km', 'num_of_contact_methods']

    # Create SHAP explainer
    explainer = shap.Explainer(model_only, transformed_input, feature_names=feature_names)
    shap_values = explainer(transformed_input)

    # Plot SHAP values
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
