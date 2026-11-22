import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="SurgiPredict",
    page_icon="🏥",
    layout="wide"
)

model = joblib.load("surgipredict_model.pkl")
scaler = joblib.load("surgipredict_scaler.pkl")
config = joblib.load("surgipredict_config.pkl")

feature_columns = config['feature_columns']
categorical_cols = config['categorical_cols']
numerical_cols = config['numerical_cols']
input_config = config['input_config']
threshold = config['threshold']

st.title("🏥 SurgiPredict")
st.subheader("Surgical Complications Prediction")

st.write(
    "Enter the patient information below to estimate "
    "the probability of post-operative complications."
)

st.info(
    "This application is an educational machine learning demo, "
    "not a medical diagnosis or clinical decision tool."
)

with st.form("prediction_form"):

    patient_data = {}

    st.subheader("Patient Information")

    for col in config['raw_columns']:

        field = input_config[col]

        if field['type'] in ['categorical', 'discrete']:

            patient_data[col] = st.selectbox(
                col,
                options=field['options']
            )

        else:

            patient_data[col] = st.number_input(
                col,
                min_value=float(field['min']),
                max_value=float(field['max']),
                value=float(field['default'])
            )

    submitted = st.form_submit_button(
        "Predict Complications",
        use_container_width=True
    )

if submitted:

    input_df = pd.DataFrame([patient_data])

    input_encoded = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        dtype=int
    )

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_encoded[numerical_cols] = scaler.transform(
        input_encoded[numerical_cols]
    )

    probability = model.predict_proba(input_encoded)[0][1]

    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Complication Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Classification Threshold",
            f"{threshold * 100:.0f}%"
        )

    if prediction == 1:
        st.error("The model predicts a possible complication.")

    else:
        st.success("The model predicts no complication.")

    st.caption(
        "This prediction is for educational purposes only "
        "and must not be used for actual medical decisions."
    )