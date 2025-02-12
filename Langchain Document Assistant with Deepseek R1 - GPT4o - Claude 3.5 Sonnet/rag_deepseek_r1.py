import streamlit as st                                              # For building the web app
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader   # For loading PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter # For chunking documents
from langchain_core.vectorstores import InMemoryVectorStore         # For storing document vectors
from langchain_ollama import OllamaEmbeddings                       # For generating document embeddings (deepseek-r1:1.5b)
from langchain_openai import OpenAIEmbeddings                       # For generating document embeddings (OpenAI)
from langchain_core.prompts import ChatPromptTemplate               # For generating conversation prompts
from langchain_ollama.llms import OllamaLLM                         # For generating AI responses

# Load environment variables
load_dotenv()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create a custom CSS theme for the Streamlit app
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Define prompt template and model configurations
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 15 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Initialize the path for storing uploaded PDFs
PDF_STORAGE_PATH = 'document_store/pdfs/'

# Initialize embedding model
# EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")    # Use the local DeepSeek model for embeddings

EMBEDDING_MODEL = OpenAIEmbeddings(                             # Use OpenAI's latest model for embeddings
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large"
)

# Initialize vector store for document chunks
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# Initialize language model for generating responses
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")            # Use the local DeepSeek model for language model

# Function to save the uploaded PDF file
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Fuction to load PDF documents from the uploaded file
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Function to chunk the documents into smaller parts
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Function to index the document chunks
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# Function to find related documents based on the user query
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Function to generate an answer to the user query
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    raw_response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    cleaned_response = raw_response.replace("Answer:", "").strip()
    cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n'))

    return cleaned_response

# Function to display the chat history
def display_chat_history():
    # Display all messages in the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.write(message["content"])

# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# Add application explanation
st.expander("ℹ️ About DocuMind AI").markdown(
"""
    - This app allows you to upload a research document (in PDF format) and ask questions about its content.
    - The AI assistant will help you find answers to your questions based on the document content.
    - Choose an AI model from the sidebar and upload a PDF document to get started!
"""
)

# Add a clear chat button in the sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

# Main App Logic
if uploaded_pdf: # If a PDF file is uploaded
    saved_path = save_uploaded_file(uploaded_pdf) # Save the uploaded file
    raw_docs = load_pdf_documents(saved_path) # Load the PDF document
    processed_chunks = chunk_documents(raw_docs) # Chunk the document into smaller parts
    index_documents(processed_chunks) # Index the document chunks
    
    # Display success message
    st.success("✅ Document processed successfully! Ask your questions below.")

    # Display existing chat history
    display_chat_history()
    
    # Chat Input Section
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

        # Rerun to display the updated chat history
        st.rerun()