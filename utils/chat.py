import streamlit as st
from ragapp import setup_rag

def chat_interface():
    """
    Render the chat interface and process user queries.
    """


    # Retrieve RAG settings from session state
    retriever, llm = setup_rag(
        st.session_state["file_path"],
        st.session_state["chunk_size"],
        st.session_state["temperature"],
    )

    # Chat input and response
    query = st.chat_input("Say something...")
    if query:
        from langchain.chains.retrieval import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        # Define prompt
        system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Set up RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Get response
        response = rag_chain.invoke({"input": query})

        # Display response
        st.write("**Answer:**", response["answer"])
