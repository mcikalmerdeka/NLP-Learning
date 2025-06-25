import streamlit as st

def render_app_info_expander():
    """Render the application information expander"""
    return st.expander("ℹ️ About DocuChat AI").markdown(
        """
        - This app allows you to upload a research document (in PDF format) and ask questions about its content.
        - The AI assistant will help you find answers to your questions based on the document content.
        - Choose an AI model from the sidebar and upload a PDF document to get started!
        - **🔍 External Search Toggle**: Enable/disable external web search when document context is insufficient
            - **ON**: AI will search external sources for additional information when needed
            - **OFF**: AI will only use information from your uploaded documents
        """
    )

def render_developer_flow_expander():
    """Render the developer execution flow expander"""
    return st.expander("🔧 Application Execution Flow (For Developers)").markdown(
        """
        **1. Document Processing Pipeline:**
        - `save_uploaded_file()` → Save PDF to `document_store/pdfs/`
        - `document_already_exists()` → Check if file exists in vector store
        - `load_pdf_documents()` → Load PDF using PyMuPDFLoader
        - `chunk_documents()` → Split into 1000-char chunks with 200 overlap
        - `index_documents()` → Store embeddings in ChromaDB

        **2. RAG Chain Creation:**
        - `create_retriever()` → Initialize similarity search retriever (k=5)
        - `create_rag_chain()` → Build LCEL chain with chosen prompt template
        - Chain: `{context, query} → prompt → LLM → parser`

        **3. Answer Generation Flow:**
        - `generate_enhanced_answer()` → Main orchestration function
        - Step 1: Get initial response from RAG chain
        - Step 2: Check for `[EXTERNAL_SEARCH_NEEDED]` marker
        - Step 3: If needed & enabled: `lookup()` external search via Tavily
        - Step 4: Combine document + external context using enhanced prompt
        - Step 5: Generate final response and clean formatting

        **4. State Management:**
        - Session state stores chat history
        - ChromaDB persists document embeddings
        - Vector store reused across sessions for same documents
        """
    )

def render_app_header(title, subtitle=None):
    """Render standardized app header"""
    st.title(title)
    if subtitle:
        st.markdown(f"### {subtitle}")
    st.markdown("---")

def render_model_selector(model_options, default_index=0):
    """Render model selection sidebar"""
    return st.sidebar.selectbox(
        "Choose AI Model",
        list(model_options.keys()),
        index=default_index
    )

def render_external_search_toggle(external_search_available):
    """Render external search toggle with status"""
    external_search_enabled = st.sidebar.checkbox(
        "🔍 Enable External Search",
        value=external_search_available,
        disabled=not external_search_available,
        help="When enabled, the AI will search external sources if document context is insufficient. Requires TAVILY_API_KEY."
    )
    
    # Display status
    if external_search_available:
        if external_search_enabled:
            st.sidebar.success("✅ External Search: ON")
        else:
            st.sidebar.info("📚 Document Only Mode: ON")
    else:
        st.sidebar.warning("⚠️ External Search Unavailable (Missing API Key)")
    
    return external_search_enabled

def render_clear_chat_button():
    """Render clear chat button and handle logic"""
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def render_file_uploader(file_type="pdf", help_text="Select a PDF document for analysis"):
    """Render file uploader with standard configuration"""
    return st.file_uploader(
        f"Upload Research Document ({file_type.upper()})",
        type=file_type,
        help=help_text,
        accept_multiple_files=False
    )

def display_chat_history():
    """Display chat history from session state"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.write(message["content"])

def render_status_message(message_type, message, is_new_doc=False, model_name="", mode_info=""):
    """Render standardized status messages"""
    full_message = f"{message} (using {model_name} - {mode_info})" if model_name and mode_info else message
    
    if message_type == "success":
        st.success(f"✅ {full_message}")
    elif message_type == "info":
        st.info(f"📄 {full_message}")
    elif message_type == "warning":
        st.warning(f"⚠️ {full_message}")
    elif message_type == "error":
        st.error(f"❌ {full_message}") 