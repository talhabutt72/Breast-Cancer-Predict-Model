import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("ML Model Demo")


df = pd.read_csv("breast_cancer.csv")
# replace with your actual file
feature_names = df.drop(["diagnosis", "id", "Unnamed: 32"], axis=1).columns.tolist()

inputs = []
st.subheader("Enter feature values:")
for col in feature_names:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    features = np.array(inputs).reshape(1, -1)
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.write("Cantrious")
        # st.balloons()
    else:
        print("Not Cantrious")
        st.balloons()