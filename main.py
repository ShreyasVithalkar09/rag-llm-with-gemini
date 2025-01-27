import streamlit as st
import os
from utils.file_upload import file_upload_form
from utils.chat import chat_interface

# App title
st.title("Custom RAG Application")

# Ensure directories exist
upload_dir = "./uploaded_files"
os.makedirs(upload_dir, exist_ok=True)

# File Upload Form
file_uploaded = file_upload_form(upload_dir)

# Chat Interface
if "ready" in st.session_state and st.session_state["ready"]:
    chat_interface()
