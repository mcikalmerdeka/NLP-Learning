import streamlit as st

def apply_custom_theme():
    """Apply custom CSS theme to the Streamlit app"""
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