import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from styles.streamlit_theme import apply_custom_theme

# Load environment variables
load_dotenv()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Apply custom CSS theme
apply_custom_theme()

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
DOCUMENT_VECTOR_DB = Chroma(
    embedding_function=EMBEDDING_MODEL,
    collection_name="document_chunks",
    persist_directory="./chroma_db"
)

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

# Function to check if document already exists in vector store
def document_already_exists(file_name):
    """Check if document already exists in vector store"""
    try:
        existing_docs = DOCUMENT_VECTOR_DB.get()
        if existing_docs and 'metadatas' in existing_docs:
            existing_files = [doc.get('source', '') for doc in existing_docs['metadatas'] if doc]
            return any(file_name in file_path for file_path in existing_files)
    except:
        pass
    return False

# Function to reset or initialize vector store
def initialize_vector_store():
    # Check if collection exists and delete it
    try:
        DOCUMENT_VECTOR_DB.delete_collection()
    except:
        pass
    # Create fresh collection
    return Chroma(
        embedding_function=EMBEDDING_MODEL,
        collection_name="document_chunks",
        persist_directory="./chroma_db"
    )

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

# Function to find related documents based on the user query (now returns retriever)
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

# Legacy function for backward compatibility (keeping the old approach as fallback)
def find_related_documents(query, k=5): # k is the number of documents to return
    return DOCUMENT_VECTOR_DB.similarity_search(query, k=k)

# Legacy function for backward compatibility
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
    
    # Check if document already exists before processing
    if not document_already_exists(uploaded_pdf.name):
        raw_docs = load_pdf_documents(saved_path) # Load the PDF document
        processed_chunks = chunk_documents(raw_docs) # Chunk the document into smaller parts
        index_documents(processed_chunks) # Index the document chunks
        
        # Create the RAG chain using LCEL
        retriever = create_retriever(k=5)
        rag_chain = create_rag_chain(LANGUAGE_MODEL, retriever)
        
        # Display success message
        st.success(f"✅ New document processed and added to vector store! Ask your questions below (using {selected_model} with LCEL chain)")
    else:
        # Document already exists, just create retriever and chain
        retriever = create_retriever(k=5)
        rag_chain = create_rag_chain(LANGUAGE_MODEL, retriever)
        
        # Display info message
        st.info(f"📄 Document '{uploaded_pdf.name}' already exists in vector store! You can ask questions about it (using {selected_model} with LCEL chain)")
    
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