import streamlit as st
from utils.parser import extract_text
from utils.scorer import match_skills

st.title("AI Resume Screener 🚀")

uploaded_file = st.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf","txt","docx"])
job_description = st.text_area("Paste Job Description here")

if uploaded_file and job_description:
    # Extract text from uploaded file
    resume_text = extract_text(uploaded_file)

    # Show the extracted resume text
    st.subheader("📄 Extracted Resume Text")
    st.text_area("Resume Content", value=resume_text, height=300)

    # Compute match score
    score = match_skills(resume_text, job_description)
    
    st.subheader("🎯 Resume Match Score")
    st.success(f"{score}%")
    
    if score > 70:
        st.balloons()
        st.write("Excellent Match! 🎉")
    elif score > 40:
        st.write("Decent Match 👍")
    else:
        st.write("Low Match ❌")
