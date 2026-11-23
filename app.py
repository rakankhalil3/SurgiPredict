import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SurgiPredict - Educational surgical complication risk demo
# ---------------------------------------------------------

st.set_page_config(
    page_title="SurgiPredict | Surgical Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem; max-width: 1450px;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    div[data-testid="stMetric"] {
        background: rgba(49,  51,  63, 0.04);
        border: 1px solid rgba(120,120,120,0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1rem;
    }
    .hero p {color: #e7f4f7; margin-bottom: 0;}
    .small-note {font-size: 0.88rem; color: #687782;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load exported training artifacts ----------
@st.cache_resource
def load_model_files():
    model = joblib.load("surgipredict_model.pkl")
    scaler = joblib.load("surgipredict_scaler.pkl")
    saved_columns = joblib.load("surgipredict_columns.pkl")
    config = joblib.load("surgipredict_config.pkl")
    return model, scaler, saved_columns, config


try:
    with st.spinner("Loading SurgiPredict model..."):
        rf_model, scaler, train_columns, config = load_model_files()
except Exception as exc:
    st.error("The model files could not be loaded.")
    st.exception(exc)
    st.stop()

# ---------- Configuration ----------
categorical_cols = config["categorical_cols"]
numerical_cols = config["numerical_cols"]
category_options = config["category_options"]
input_config = config["input_config"]
threshold_default = float(config.get("threshold", 0.4))

# Moon phase stays in the trained model's feature set, but is not shown
# as a user input. Prefer saving its training-set mode in config under
# "moonphase_default". The fallback below keeps older config files usable.
moonphase_default = config.get("moonphase_default", 0)
moonphase_default = int(moonphase_default)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Dashboard Settings")
st.sidebar.caption("SurgiPredict model controls")
st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Prediction Threshold",
    min_value=0.10,
    max_value=0.90,
    value=threshold_default,
    step=0.05,
)
st.sidebar.info(
    "A lower threshold generally identifies more positive cases, "
    "but may also increase false alarms."
)
st.sidebar.markdown("---")
st.sidebar.caption("Educational prototype — not for clinical decision-making.")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>🏥 SurgiPredict</h1>
      <p>AI-assisted estimation of post-operative complication risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational project only. This application is not a medical device and "
    "must not be used to diagnose patients or make clinical decisions."
)

tab_assessment, tab_performance, tab_about = st.tabs(
    ["🩺 Patient Assessment", "📊 Model Performance", "ℹ️ About"]
)

# =========================================================
# TAB 1 — PATIENT ASSESSMENT
# =========================================================
with tab_assessment:
    st.subheader("Patient Clinical Information")
    st.caption("Enter the available patient and procedure information below.")

    with st.form("patient_assessment_form"):
        # Main patient information
        st.markdown("#### 1. Patient Information")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            age_default = int(round(float(input_config["Age"]["default"])))
            age = st.number_input(
                "Age (العمر)",
                min_value=1,
                max_value=120,
                value=max(1, min(120, age_default)),
                step=1,
            )
        with c2:
            weight = st.number_input(
                "Weight (الوزن) — kg",
                min_value=30.0,
                max_value=250.0,
                value=75.0,
                step=0.5,
            )
        with c3:
            height_cm = st.number_input(
                "Height (الطول) — cm",
                min_value=100.0,
                max_value=230.0,
                value=170.0,
                step=0.5,
            )
        with c4:
            gender_label = st.selectbox("Gender (النوع)", ["Female", "Male"])
            # Keep the same numeric encoding used by the trained model.
            # Verify this mapping against the dataset dictionary before deployment.
            gender = 0 if gender_label == "Female" else 1

        bmi = weight / ((height_cm / 100) ** 2)
        st.metric("Calculated BMI (مؤشر كتلة الجسم)", f"{bmi:.2f}")

        st.markdown("---")
        st.markdown("#### 2. Medical Background")
        c1, c2, c3 = st.columns(3)

        asa_options = {
            "ASA I — Healthy patient (مريض سليم)": 0,
            "ASA II — Mild systemic disease (مرض جهازي خفيف)": 1,
            "ASA III — Severe systemic disease (مرض جهازي شديد)": 2,
        }
        with c1:
            available_asa = input_config.get("asa_status", {}).get("options", list(asa_options.values()))
            asa_options = {k: v for k, v in asa_options.items() if v in available_asa}
            if not asa_options:
                st.error("No supported ASA categories were found in the model configuration.")
                st.stop()
            asa_label = st.selectbox("ASA Status (تصنيف الحالة الصحية)", list(asa_options))
            asa_status = asa_options[asa_label]

        with c2:
            baseline_charlson = st.number_input(
                "Charlson Index (مؤشر الأمراض المصاحبة)",
                min_value=0.0,
                max_value=20.0,
                value=float(input_config["baseline_charlson"]["default"]),
                step=1.0,
            )
        with c3:
            ahrq_values = category_options["ahrq_ccs"]
            ahrq_ccs = st.selectbox(
                "AHRQ CCS (مجموعة التصنيف الطبي)",
                options=ahrq_values,
            )

        st.markdown("##### Medical History (التاريخ المرضي)")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            baseline_cancer = st.checkbox("Cancer history (تاريخ السرطان)")
            baseline_cvd = st.checkbox("Cardiovascular disease (أمراض القلب والأوعية)")
        with h2:
            baseline_dementia = st.checkbox("Dementia (الخرف)")
            baseline_diabetes = st.checkbox("Diabetes (السكري)")
        with h3:
            baseline_digestive = st.checkbox("Digestive disease (أمراض الجهاز الهضمي)")
            baseline_osteoart = st.checkbox("Osteoarthritis (التهاب المفاصل)")
        with h4:
            baseline_psych = st.checkbox("Psychiatric disorder (اضطرابات نفسية)")
            baseline_pulmonary = st.checkbox("Pulmonary disease (أمراض الرئة)")

        st.markdown("---")
        st.markdown("#### 3. Procedure Information")
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            dow_labels = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"]
            dow_values = category_options["dow"]
            dow_pairs = list(zip(dow_labels, dow_values))
            dow_label = st.selectbox(
                "Operation Day (يوم إجراء العملية)",
                options=[x[0] for x in dow_pairs],
            )
            dow = dict(dow_pairs)[dow_label]

        with p2:
            month_labels = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ]
            month_values = category_options["month"]
            month_pairs = list(zip(month_labels, month_values))
            month_label = st.selectbox(
                "Operation Month (شهر إجراء العملية)",
                options=[x[0] for x in month_pairs],
            )
            month = dict(month_pairs)[month_label]

        with p3:
            hour = st.number_input(
                "Operation Start Hour (ساعة بدء العملية)",
                min_value=0.0,
                max_value=23.5,
                value=float(input_config["hour"]["default"]),
                step=0.5,
            )
        with p4:
            race = st.selectbox(
                "Race (الفئة العِرقية)",
                options=category_options["race"],
            )

        st.markdown("---")
        st.markdown("#### 4. Risk Indicators")
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            ccs_complication_rate = st.number_input(
                "CCS Complication Rate (معدل مضاعفات التصنيف)",
                value=float(input_config["ccsComplicationRate"]["default"]),
                format="%.4f",
            )
        with r2:
            ccs_mort30_rate = st.number_input(
                "CCS 30-Day Mortality Rate (معدل الوفاة خلال 30 يومًا)",
                value=float(input_config["ccsMort30Rate"]["default"]),
                format="%.4f",
            )
        with r3:
            complication_rsi = st.number_input(
                "Complication RSI (مؤشر خطورة المضاعفات)",
                value=float(input_config["complication_rsi"]["default"]),
                format="%.2f",
            )
        with r4:
            mortality_rsi = st.number_input(
                "Mortality RSI (مؤشر خطورة الوفاة)",
                value=float(input_config["mortality_rsi"]["default"]),
                format="%.2f",
            )

        submit = st.form_submit_button(
            "🔍 Predict Complication Risk",
            use_container_width=True,
            type="primary",
        )

    if submit:
        # Keep every raw feature expected by the original trained model.
        # moonphase is hidden from the interface and filled internally.
        input_data = {
            "bmi": [bmi],
            "Age": [float(age)],
            "asa_status": [asa_status],
            "baseline_cancer": [int(baseline_cancer)],
            "baseline_charlson": [baseline_charlson],
            "baseline_cvd": [int(baseline_cvd)],
            "baseline_dementia": [int(baseline_dementia)],
            "baseline_diabetes": [int(baseline_diabetes)],
            "baseline_digestive": [int(baseline_digestive)],
            "baseline_osteoart": [int(baseline_osteoart)],
            "baseline_psych": [int(baseline_psych)],
            "baseline_pulmonary": [int(baseline_pulmonary)],
            "ahrq_ccs": [ahrq_ccs],
            "ccsComplicationRate": [ccs_complication_rate],
            "ccsMort30Rate": [ccs_mort30_rate],
            "complication_rsi": [complication_rsi],
            "dow": [dow],
            "gender": [gender],
            "hour": [hour],
            "month": [month],
            "moonphase": [moonphase_default],
            "mort30": [0],
            "mortality_rsi": [mortality_rsi],
            "race": [race],
        }

        input_df = pd.DataFrame(input_data)
        X_input = input_df.drop(columns=["complication", "mort30"], errors="ignore")

        X_input = pd.get_dummies(
            X_input,
            columns=categorical_cols,
            dtype=int,
        )
        X_input = X_input.reindex(columns=list(train_columns), fill_value=0)

        # Apply the scaler fitted on the training data only.
        X_input[numerical_cols] = scaler.transform(X_input[numerical_cols])

        probability = float(rf_model.predict_proba(X_input)[:, 1][0])
        prediction = int(probability >= threshold)

        st.markdown("---")
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.metric("Estimated Complication Probability", f"{probability:.1%}")
        with result_col2:
            if prediction == 1:
                st.error(
                    "**High Risk of Complications**\n\n"
                    "The model classified this input as higher risk. "
                    "This output is an educational estimate, not a clinical diagnosis."
                )
            else:
                st.success(
                    "**Lower Predicted Risk of Complications**\n\n"
                    "The model classified this input as lower risk. "
                    "This output does not guarantee that complications will not occur."
                )

        st.caption(
            f"Classification threshold: {threshold:.2f}. "
            "The probability is generated by the trained machine-learning model."
        )

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab_performance:
    st.subheader("Model Performance")
    st.caption(
        "Reported held-out test results from the saved model at threshold 0.40. "
        "These figures do not change when the prediction threshold slider changes."
    )

    # Original full-feature model's reported test metrics at threshold 0.40.
    acc = 0.8613
    prec = 0.7406
    rec = 0.6924
    f1 = 0.7157

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall", f"{rec:.2%}")
    m4.metric("F1 Score", f"{f1:.2%}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Confusion Matrix")
        cm = np.array([[2010, 179], [227, 511]])
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["No Complication", "Complication"])
        ax.set_yticks([0, 1], labels=["No Complication", "Complication"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Test Set Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Top 10 Feature Importances")
        importance_df = pd.DataFrame(
            {
                "Feature": list(train_columns),
                "Importance": rf_model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False).head(10)

        fig2, ax2 = plt.subplots(figsize=(6.2, 4.5))
        ax2.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1],
        )
        ax2.set_xlabel("Importance")
        ax2.set_title("Top 10 Model Features")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.info(
        "Feature importance describes how the fitted model used features to make "
        "predictions. It does not prove that a feature causes complications."
    )

# =========================================================
# TAB 3 — ABOUT
# =========================================================
with tab_about:
    st.subheader("About SurgiPredict")
    st.write(
        "SurgiPredict is an educational machine-learning project that estimates "
        "the probability of post-operative complications from structured patient "
        "and procedure information."
    )
    st.markdown(
        """
        **Model:** Random Forest Classifier

        **Target:** Post-operative complication (`complication`)

        **Excluded outcome field:** `mort30` (30-day mortality outcome)

        **Preprocessing:** One-Hot Encoding for categorical variables and "
        "StandardScaler for selected numerical variables.

        **Important:** The application is a student project and is not validated "
        "for real-world clinical use.
        """
    )
