import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

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
If unsure, state that you don't know. Be concise and factual (max 7 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'

# Initialize embedding model (using OpenAI's ada-002)
EMBEDDING_MODEL = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Initialize vector store
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# Model options
MODEL_OPTIONS = {
    "GPT-4o": "gpt-4o",
    "Claude 3.5 Sonnet": "claude-3-sonnet-20240229"
}

# Initialize the chosen language model
def initialize_language_model(model_choice):
    if model_choice == "GPT-4o":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0
        )
    else:  # Claude 3.5 Sonnet
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-sonnet-20240229",
            temperature=0
        )

# Helper Functions
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents, language_model):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | language_model
    
    # Get the response
    raw_response = response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_text
    })
    
    # Extract and clean the content
    if hasattr(raw_response, 'content'):
        # For Claude's response format
        cleaned_response = raw_response.content
    else:
        # For GPT-4's response format
        cleaned_response = str(raw_response)
    
    # Remove any leading/trailing whitespace and extra newlines
    cleaned_response = cleaned_response.strip()
    
    # Replace multiple newlines with a single newline for better readability
    cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
    
    return cleaned_response

def display_chat_history():
    # Display all messages in the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.write(message["content"])

# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# Add a clear chat button in the sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Model Selection
selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    list(MODEL_OPTIONS.keys()),
    index=0
)

# Initialize the chosen language model
LANGUAGE_MODEL = initialize_language_model(selected_model)

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

# Main App Logic
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success(f"✅ Document processed successfully! Ask your questions below (using {selected_model})")
    
    # Display existing chat history
    display_chat_history()
    
    # Handle new user input
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs, LANGUAGE_MODEL)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to display the updated chat history
        st.rerun()