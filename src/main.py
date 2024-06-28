import streamlit as st
from utils.doc_loader import extract_text_from_pdf
import os

# Function to reset the state of the application
def reset_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# Streamlit app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "PDF Summary", "RAG"])

if st.sidebar.button("Reset Page"):
    reset_state()
    st.experimental_rerun()

if page == "Home":
    st.title("Welcome to the Multi-Page Streamlit App")
    st.write("Use the sidebar to navigate to different pages.")

    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if 'pages' not in st.session_state:
            with st.spinner("Extracting text from PDF..."):
                st.session_state.pages, st.session_state.texts_doc, st.session_state.retriever  = extract_text_from_pdf(uploaded_file)
                # Save extracted text to Faiss
                st.sidebar.success("Text extraction and saving to Faiss complete!")

elif page == "PDF Summary":
    from pages.summary_page import summary_page
    summary_page()
elif page == "RAG":
    from pages.summary_page import summary_page
    summary_page()