import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Load vectorizer and model
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectoriser.pkl")
    model = joblib.load("xgb_model.pkl")
    return vectorizer, model

vectorizer, model = load_model()

st.title("Sentiment Analysis App using XGBoost")

uploaded_file = st.file_uploader("Upload a CSV file with a 'sentiment' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Sentence" not in df.columns:
        st.error("The CSV must contain a column named 'Sentence'.")
    else:
        st.write("Input Data:")
        st.dataframe(df)

        # Preprocess the text column
        X_transformed = vectorizer.transform(df["Sentence"])

        # Predict
        preds = model.predict(X_transformed)

        # Convert numeric predictions to labels
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        preds_labels = [label_map.get(p, "unknown") for p in preds]

        # Output dataframe
        output_df = df.copy()
        output_df["prediction"] = preds_labels

        st.write("Prediction Results:")
        st.dataframe(output_df)

        # Option to download
        csv = output_df.to_csv(index=False)
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
