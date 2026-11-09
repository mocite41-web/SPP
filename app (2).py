import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration. Predictions are not diagnoses.")

try:
    model = load_model()
    data = load_data()
except FileNotFoundError:
    st.error("Required file not found. Run `python train_model.py` first to create student_model.pkl and student_data.csv.")
    st.stop()

if "G3" not in data.columns:
    st.error("student_data.csv must contain the G3 final grade column.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded student records")
st.write("Upload a CSV with the same predictor columns used during training. Do not include G3.")
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required_columns = list(model.feature_names_in_)
        missing = [column for column in required_columns if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif new_data.empty:
            st.warning("The uploaded CSV has no records.")
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if int(value) == 1 else "NOT AT RISK"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "student_predictions.csv",
                "text/csv"
            )
    except Exception as error:
        st.error(f"Could not process this CSV: {error}")
