# app.py

from micrograd.engine import Value
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from utils import vectorize, build_vocab, tokenize
import numpy as np
import pandas as pd 

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    .stApp {
        background: linear-gradient(to right, #78909C, #B0BEC5);
        font-family: 'Poppins', sans-serif;
        color: #212121;
    }

    h1, h2, h3 {
        color: #0d47a1;
    }

    .stTextArea label, .stFileUploader label {
        font-weight: 600;
        color: #0d47a1;
        font-size: 16px;
    }

    textarea {
        background-color: #ffffff !important;
        color: #212121 !important;
        border-radius: 10px;
        border: 1px solid #cfd8dc;
        padding: 12px;
        font-size: 15px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
    }

    .stFileUploader > div {
        background-color: #ffffff !important;
        color: #212121 !important;
        border-radius: 12px;
        border: 1px solid #cfd8dc;
        padding: 12px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
    }

    .stButton > button {
        background-color: #1976d2;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        margin-top: 15px;
        transition: background-color 0.3s ease;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #1565c0;
    }

    /* Table styling */
    .stTable {
        background-color: #ffffff;
        border-radius: 10px;
        border: 2px solid #000000;  
        padding: 10px;
        font-size: 15px;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.05);
        color: #000000 !important;
    }

    .stTable table, .stTable th, .stTable td {
        color: #000000 !important;
        border: 1px solid #cfd8dc;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# Load trained model
with open("trained_model.pkl", "rb") as f:
    model, vocab = pickle.load(f)

st.title("📄 Resume Ranker (Powered by Micrograd)")
st.write("Upload a Job Description and Resumes (PDFs). See how well each resume matches the JD.")

# JD input
jd_text = st.text_area("Paste Job Description here")

# Resume uploads
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# Rank button
if st.button("Rank Resumes") and jd_text and uploaded_files:
    jd_vec = vectorize(jd_text, vocab)

    results = []
    for file in uploaded_files:
        pdf = PdfReader(file)
        resume_text = ""
        for page in pdf.pages:
            resume_text += page.extract_text()

        res_vec = vectorize(resume_text, vocab)
        x_vec = np.concatenate([jd_vec, res_vec])
        x_input = [Value(v) for v in x_vec]

        score = model(x_input).data
        results.append((file.name, score))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    #st.subheader("🧠 Match Scores")
    #for filename, score in results:
    #   st.write(f"📄 {filename}: **{score:.2f}** match")

    df = pd.DataFrame(results, columns=["Resume File", "Match Score"])
    df["Match Score"] = df["Match Score"].apply(lambda x: f"{x:.2f}")
    
    st.subheader("📊 Ranked Results")
    st.table(df)
