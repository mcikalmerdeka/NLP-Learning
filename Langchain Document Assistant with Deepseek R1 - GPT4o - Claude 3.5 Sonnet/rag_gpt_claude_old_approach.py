import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

"""
This is the old approach to the RAG system. Where it uses the InMemoryVectorStore to store the document chunks instead of the ChromaDB vector database.
There are several reasons for this, for details please refer to this https://claude.ai/share/b7341580-5bad-4c7c-9215-cd00345e51fc.
"""

# Load environment variables
load_dotenv()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create a custom CSS theme for the Streamlit app
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #262730;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #ffffff !important;
        color: #262730 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e6f3ff !important;
        border: 1px solid #b3d9ff !important;
        color: #262730 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff !important;
        border: 1px solid #e6e6e6 !important;
        color: #262730 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #4b9eff !important;
        color: #ffffff !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #262730 !important;
    }
    
    .stFileUploader {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #0068c9 !important;
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

# Initialize embedding model (using OpenAI's latest model)
EMBEDDING_MODEL = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large"
)

# Initialize vector store for document chunks
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# Model options
MODEL_OPTIONS = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1": "gpt-4.1",
    "Claude Sonnet 4": "claude-sonnet-4-20250514"
}

# Initialize the chosen language model
def initialize_language_model(model_choice):
    if model_choice == "GPT-4o":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0,
            max_tokens=1024
        )
    elif model_choice == "GPT-4.1":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4.1",
            temperature=0,
            max_tokens=1024
        )
    else:  # Claude Sonnet 4
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-sonnet-4-20250514",
            temperature=0,
            max_tokens=1024
        )

# Function to save the uploaded PDF file
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Fuction to load PDF documents from the uploaded file
def load_pdf_documents(file_path):
    document_loader = PyMuPDFLoader(file_path)
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

# Function to format retrieved documents into context
def format_docs(docs):
    """Format retrieved documents into a single context string"""
    return "\n\n".join([doc.page_content for doc in docs])

# Function to create RAG chain using LCEL
def create_rag_chain(language_model, retriever):
    """Create a comprehensive RAG chain using LangChain Expression Language"""
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Create the RAG chain using LCEL syntax
    rag_chain = (
        {
            "document_context": retriever | RunnableLambda(format_docs),
            "user_query": RunnablePassthrough()
        }
        | prompt
        | language_model
        | StrOutputParser()
    )
    
    return rag_chain

# Function to create retriever from InMemoryVectorStore
def create_retriever(k=5):
    """Create a retriever from the vector store"""
    return DOCUMENT_VECTOR_DB.as_retriever(search_kwargs={"k": k})

# Updated function to generate an answer using LCEL chain
def generate_answer_with_chain(user_query, rag_chain):
    """Generate answer using the RAG chain with LCEL"""
    try:
        # Invoke the chain with the user query
        response = rag_chain.invoke(user_query)
        
        # Clean up the response
        cleaned_response = response.strip()
        cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
        
        return cleaned_response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Legacy function for backward compatibility
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Updated legacy function to use StrOutputParser
def generate_answer(user_query, context_documents, language_model):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | language_model | StrOutputParser()
    
    # Get the response
    response = response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_text
    })
    
    # Clean up the response
    cleaned_response = response.strip()
    cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
    
    return cleaned_response

# Function to display the chat history
def display_chat_history():
    # Display all messages in the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.write(message["content"])

# UI Configuration
st.title("📘 DocuChat AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# Add application explanation
st.expander("ℹ️ About DocuChat AI").markdown(
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
if uploaded_pdf: # If a PDF file is uploaded
    saved_path = save_uploaded_file(uploaded_pdf) # Save the uploaded file
    raw_docs = load_pdf_documents(saved_path) # Load the PDF document
    processed_chunks = chunk_documents(raw_docs) # Chunk the document into smaller parts
    index_documents(processed_chunks) # Index the document chunks
    
    # Create the RAG chain using LCEL
    retriever = create_retriever(k=5)
    rag_chain = create_rag_chain(LANGUAGE_MODEL, retriever)
    
    # Display success message
    st.success(f"✅ Document processed successfully! Ask your questions below (using {selected_model} with LCEL chain)")
    
    # Display existing chat history
    display_chat_history()
    
    # Handle new user input
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Analyzing document with LCEL chain..."):
            # Use the new LCEL chain approach
            ai_response = generate_answer_with_chain(user_input, rag_chain)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to display the updated chat history
        st.rerun()