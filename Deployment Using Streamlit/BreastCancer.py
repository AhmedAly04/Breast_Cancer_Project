import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier

model_path = r'X:\NTI\ML\Projects\Breast Cancer\Deployment Using Streamlit\BreastCancer_model.pkl'
pre_path = r'X:\NTI\ML\Projects\Breast Cancer\Deployment Using Streamlit\model_preprocessing.pkl'

with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)
with open(pre_path, 'rb') as f:
    preprocess = pickle.load(f)

scaler = preprocess['scaler']
feature_order = preprocess['feature_order']

st.title("🔬 Breast Cancer Prediction")
st.markdown("Enter the clinical measurements below.")

with st.form("prediction_form"):
    user_inputs = {}
    col1, col2 = st.columns(2)
    for i, feature_name in enumerate(feature_order):
        display_name = feature_name.replace('_', ' ').title()

        with col1 if i % 2 == 0 else col2:
            user_inputs[feature_name] = st.number_input(
                f"{display_name}", 
                value=0.0,
                format="%.4f"
            )
            
    submit = st.form_submit_button("Predict Diagnosis")
if submit:
    input_df = pd.DataFrame([user_inputs])
    
    input_df = input_df[feature_order]
    
    try:
        input_scaled = scaler.transform(input_df)
        
        prediction = loaded_model.predict(input_scaled)
        probs = loaded_model.predict_proba(input_scaled)
        
        st.divider()
        
        if prediction[0] == 0:
            confidence = probs[0][0]  
            st.error(f"### Result: MALIGNANT")
            st.write(f"The model is **{confidence*100:.2f}%** confident.")
        else:
            confidence = probs[0][1]
            st.success(f"### Result: BENIGN")
            st.write(f"The model is **{confidence*100:.2f}%** confident.")
            
    except Exception as e:
        st.error(f"Error during calculation: {e}")