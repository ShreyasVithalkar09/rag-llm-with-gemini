import streamlit as st
import os

def file_upload_form(upload_dir):
    """
    Render the file upload form and capture user inputs.
    :param upload_dir: Directory to save uploaded files
    :return: True if the form was submitted successfully, False otherwise
    """
    with st.sidebar.form("RAG Configuration"):
        st.header("Upload & Configure")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.3)
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=400, value=300, step=50)
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and uploaded_file:
        # Save uploaded file locally
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded and saved at: {file_path}")

        # Store parameters in session state
        st.session_state["file_path"] = file_path
        st.session_state["temperature"] = temperature
        st.session_state["chunk_size"] = chunk_size
        st.session_state["ready"] = True

        # Return success
        return True
    return False
