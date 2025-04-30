# app.py

import streamlit as st
import tempfile
import PyPDF2

from vertexai import generative_models
import vertexai


# --- Google Vertex AI Init ---
PROJECT_ID = "projectrag-458404"
REGION = "us-central1" 

vertexai.init(project=PROJECT_ID, location=REGION)
# model = generative_models.GenerativeModel("gemini-1.0-pro")
model = generative_models.GenerativeModel("gemini-2.5-pro-preview-03-25")	

# --- Helper Functions ---

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def generate_answer(prompt_text):
    response = model.generate_content(prompt_text)
    return response.text


# --- Streamlit UI ---
st.title("📄 PDF Chatbot with Google Gemini")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
user_question = st.text_input("Ask a question about the document or request a summary:")

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract text from PDF
    doc_text = extract_text_from_pdf(tmp_path)

    if user_question:
        # Prepare prompt
        prompt = f"""You are an intelligent assistant. Based on the following document, answer the question or summarize if requested.

        Document:
        {doc_text[:4000]}  # Gemini accepts roughly ~4K tokens as input.

        Question:
        {user_question}
        """

        # Get answer
        with st.spinner("Generating response..."):
            answer = generate_answer(prompt)
        st.subheader("🧠 Response")
        st.write(answer)
