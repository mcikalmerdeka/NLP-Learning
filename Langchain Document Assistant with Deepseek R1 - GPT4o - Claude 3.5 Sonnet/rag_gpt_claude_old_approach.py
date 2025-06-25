import streamlit as st
import os
import sys

# Add the current directory to Python path so we can import from agents
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Import Streamlit UI components
from styles.streamlit_theme import apply_custom_theme
from components.ui_components import (
    render_app_header,
    render_app_info_expander,
    render_old_approach_flow_expander,
    render_model_selector,
    render_external_search_toggle,
    render_clear_chat_button,
    render_file_uploader,
    display_chat_history,
    render_status_message
)

# Import the external lookup agent
try:
    from agents.external_sources_lookup_agent import lookup
    EXTERNAL_SEARCH_AVAILABLE = True
except ImportError as e:
    st.warning(f"External search not available: {e}")
    EXTERNAL_SEARCH_AVAILABLE = False

# Import prompt templates
from prompts import (
    PROMPT_TEMPLATE,
    DOCUMENT_ONLY_PROMPT_TEMPLATE,
    ENHANCED_PROMPT_TEMPLATE
)

"""
This is the old approach to the RAG system. Where it uses the InMemoryVectorStore to store the document chunks instead of the ChromaDB vector database.
There are several reasons for this, for details please refer to this https://claude.ai/share/b7341580-5bad-4c7c-9215-cd00345e51fc.
"""

# Load environment variables
load_dotenv()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Apply custom CSS theme
apply_custom_theme()

# Model configurations
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

# Function to check if document already exists in vector store
def document_already_exists(file_name):
    """Check if document already exists in vector store"""
    try:
        # For InMemoryVectorStore, we need to check stored documents differently
        stored_docs = DOCUMENT_VECTOR_DB.similarity_search("", k=1000)  # Get many docs to check sources
        if stored_docs:
            existing_files = [doc.metadata.get('source', '') for doc in stored_docs if hasattr(doc, 'metadata')]
            return any(file_name in file_path for file_path in existing_files)
    except:
        pass
    return False

# Function to format retrieved documents into context
def format_docs(docs):
    """Format retrieved documents into a single context string"""
    return "\n\n".join([doc.page_content for doc in docs])

# Function to create RAG chain using LCEL with mode selection
def create_rag_chain(language_model, retriever, external_search_enabled=True):
    """Create a comprehensive RAG chain using LangChain Expression Language"""
    
    # Choose prompt template based on external search setting
    if external_search_enabled:
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    else:
        prompt = ChatPromptTemplate.from_template(DOCUMENT_ONLY_PROMPT_TEMPLATE)
    
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

# Enhanced function to generate answer with external search capability
def generate_enhanced_answer(user_query, rag_chain, language_model, external_search_enabled=True):
    """Generate answer using RAG chain with optional external search fallback"""
    try:
        # Step 1: Try to answer using document context first
        initial_response = rag_chain.invoke(user_query)
        
        # Step 2: Check if external search is needed and enabled
        if (external_search_enabled and 
            "[EXTERNAL_SEARCH_NEEDED]" in initial_response and 
            EXTERNAL_SEARCH_AVAILABLE):
            
            st.info("🔍 Document context insufficient. Searching external sources...")
            
            try:
                # Step 3: Perform external search
                external_results = lookup(user_query)
                
                # Step 4: Create enhanced prompt with both contexts
                enhanced_prompt = ChatPromptTemplate.from_template(ENHANCED_PROMPT_TEMPLATE)
                
                # Get retriever to get document context again
                retriever = create_retriever(k=5)
                docs = retriever.invoke(user_query)
                document_context = format_docs(docs)
                
                # Step 5: Generate enhanced response
                enhanced_chain = enhanced_prompt | language_model | StrOutputParser()
                
                final_response = enhanced_chain.invoke({
                    "user_query": user_query,
                    "document_context": document_context,
                    "external_context": external_results
                })
                
                # Clean up and return enhanced response
                cleaned_response = final_response.strip()
                cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
                
                return cleaned_response
                
            except Exception as e:
                st.error(f"External search failed: {str(e)}")
                # Fall back to original response without the search indicator
                fallback_response = initial_response.replace("[EXTERNAL_SEARCH_NEEDED]", "").strip()
                return fallback_response if fallback_response else "I don't have sufficient information to answer this query."
        
        else:
            # Step 6: Return normal response (either external search disabled or not needed)
            if not external_search_enabled:
                # Clean response for document-only mode
                cleaned_response = initial_response.strip()
            else:
                # Clean response and remove external search indicator if present
                cleaned_response = initial_response.replace("[EXTERNAL_SEARCH_NEEDED]", "").strip()
            
            cleaned_response = '\n'.join(line.strip() for line in cleaned_response.split('\n') if line.strip())
            return cleaned_response if cleaned_response else "I don't have sufficient information to answer this query."
            
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

# Function moved to components.ui_components

# Streamlit UI Configuration  
render_app_header("📘 DocuChat AI - Old Approach", "InMemoryVectorStore Implementation")

# Render information expanders
render_app_info_expander()
render_old_approach_flow_expander()

# Sidebar components
render_clear_chat_button()
selected_model = render_model_selector(MODEL_OPTIONS)
external_search_enabled = render_external_search_toggle(EXTERNAL_SEARCH_AVAILABLE)

# Initialize the chosen language model
LANGUAGE_MODEL = initialize_language_model(selected_model)

# File Upload Section
uploaded_pdf = render_file_uploader()

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
        rag_chain = create_rag_chain(LANGUAGE_MODEL, retriever, external_search_enabled)
        
        # Display success message
        mode_info = "with External Search" if external_search_enabled else "Document Only Mode"
        render_status_message("success", "New document processed and added to vector store! Ask your questions below", 
                            model_name=selected_model, mode_info=mode_info)
    else:
        # Document already exists, just create retriever and chain
        retriever = create_retriever(k=5)
        rag_chain = create_rag_chain(LANGUAGE_MODEL, retriever, external_search_enabled)
        
        # Display info message
        mode_info = "with External Search" if external_search_enabled else "Document Only Mode"
        render_status_message("info", f"Document '{uploaded_pdf.name}' already exists in vector store! You can ask questions about it", 
                            model_name=selected_model, mode_info=mode_info)

    # Display existing chat history
    display_chat_history()
    
    # Handle new user input
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Dynamic spinner message based on mode
        spinner_message = "Analyzing document with AI (external search enabled)..." if external_search_enabled else "Analyzing document with AI (document only mode)..."
        
        with st.spinner(spinner_message):
            # Use the enhanced answer generation with mode setting
            ai_response = generate_enhanced_answer(user_input, rag_chain, LANGUAGE_MODEL, external_search_enabled)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to display the updated chat history
        st.rerun()