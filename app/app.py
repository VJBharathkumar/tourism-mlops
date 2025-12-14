import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# -------------------------
# CONFIG (EDIT IF NEEDED)
# -------------------------
HF_MODEL_REPO = "VJBharathkumar/tourism-prod-prediction"     # <-- your model repo on HF
HF_DATASET_REPO = "VJBharathkumar/tourism-wellness"          # <-- your dataset repo on HF
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
TRAIN_FILENAME_IN_DATASET = "train.csv"                      # uploaded in Step 5

TARGET = "ProdTaken"

# These are the expected feature columns (18) from your dataset
FEATURE_COLS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

@st.cache_resource
def load_model_and_metadata():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    model = joblib.load(model_path)

    metrics = None
    try:
        metrics_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=METRICS_FILENAME,
            repo_type="model",
        )
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = None

    return model, metrics

@st.cache_data
def load_train_for_ui_hints():
    """
    Pull train.csv from HF dataset repo to:
    - get dropdown options for categorical columns
    - get min/max for numeric sliders
    """
    train_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=TRAIN_FILENAME_IN_DATASET,
        repo_type="dataset",
    )
    df = pd.read_csv(train_path)

    # If ProdTaken exists, drop it for UI feature work
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    # Keep only expected features (protects against accidental extra columns)
    df = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
    return df

def build_input_form(train_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Enter customer details")

    # Determine categorical vs numeric from training df
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in train_df.columns if c not in cat_cols]

    left, right = st.columns(2)
    values = {}

    # Helper to draw widget
    def draw_widget(col_name, container):
        if col_name in cat_cols:
            options = sorted([x for x in train_df[col_name].dropna().unique().tolist()])
            default = options[0] if options else ""
            values[col_name] = container.selectbox(col_name, options=options, index=0)
        else:
            # numeric
            series = pd.to_numeric(train_df[col_name], errors="coerce")
            min_v = float(np.nanmin(series.values))
            max_v = float(np.nanmax(series.values))
            med_v = float(np.nanmedian(series.values))

            # If it's basically an integer field, use number_input with step 1
            if np.all(np.isclose(series.dropna() % 1, 0)):
                values[col_name] = container.number_input(
                    col_name,
                    min_value=int(min_v),
                    max_value=int(max_v),
                    value=int(round(med_v)),
                    step=1,
                )
            else:
                values[col_name] = container.number_input(
                    col_name,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(med_v),
                )

    # Alternate columns for nicer layout
    for i, col_name in enumerate(FEATURE_COLS):
        if col_name not in train_df.columns:
            continue
        container = left if i % 2 == 0 else right
        draw_widget(col_name, container)

    input_df = pd.DataFrame([values], columns=[c for c in FEATURE_COLS if c in values])
    return input_df

def main():
    st.set_page_config(page_title="Tourism Package Prediction", layout="wide")

    st.title("Tourism Package Prediction")
    st.write("Predict whether the customer will take the package (`ProdTaken = 1`).")

    model, metrics = load_model_and_metadata()
    train_df = load_train_for_ui_hints()

    # Sidebar: show metrics + model info
    with st.sidebar:
        st.header("Model Info")
        st.write(f"Model repo: `{HF_MODEL_REPO}`")
        if metrics:
            st.subheader("Test Metrics")
            st.write(f"Accuracy: **{metrics.get('accuracy', 'NA')}**")
            st.write(f"F1: **{metrics.get('f1', 'NA')}**")
            st.write(f"ROC-AUC: **{metrics.get('roc_auc', 'NA')}**")
        else:
            st.info("metrics.json not found in model repo (optional).")

    input_df = build_input_form(train_df)

    st.divider()

    predict_btn = st.button("Predict", type="primary")

    if predict_btn:
        # Ensure column order matches training expectation
        input_df = input_df[[c for c in FEATURE_COLS if c in input_df.columns]].copy()

        proba = None
        pred = None

        # Some sklearn models support predict_proba; our pipeline does
        pred = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0][1])

        st.subheader("Prediction")
        st.write(f"Predicted class: **{pred}**  (1 = will take package, 0 = will not)")
        st.write(f"Probability of ProdTaken=1: **{proba:.3f}**")

        if pred == 1:
            st.success("Likely to take the package ✅")
        else:
            st.warning("Unlikely to take the package ⚠️")

        with st.expander("Show input row"):
            st.dataframe(input_df)

if __name__ == "__main__":
    main()