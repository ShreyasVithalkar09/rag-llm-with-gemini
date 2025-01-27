from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def setup_rag(file_path, chunk_size, temperature):
    """
    Set up the RAG pipeline.
    :param file_path: Path to the uploaded file
    :param chunk_size: Chunk size for text splitting
    :param temperature: LLM temperature
    :return: Retriever and LLM objects
    """
    # Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    docs = text_splitter.split_documents(data)

    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="./chroma_db")

    # Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Set up LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temperature)

    return retriever, llm
