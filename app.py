# app.py - Optimized Streamlit Chatbot with Streaming

import streamlit as st
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model

# Page configuration
st.set_page_config(
    page_title="🤖 AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Simplified CSS - removed heavy formatting
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🤖 AI Chatbot")
st.markdown("Powered by LangChain & Perplexity AI")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False

# Try to get API key from Streamlit secrets first
api_key = None
if "PPLX_API_KEY" in st.secrets:
    api_key = st.secrets["PPLX_API_KEY"]
    os.environ["PPLX_API_KEY"] = api_key
    st.session_state.api_key_configured = True

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Show API key status
    if st.session_state.api_key_configured:
        st.success("✅ API key loaded from secrets")
    else:
        st.warning("⚠️ API key not found in secrets")
        api_key = st.text_input(
            "Perplexity API Key",
            type="password",
            help="Get your API key from https://www.perplexity.ai"
        )
        if api_key:
            os.environ["PPLX_API_KEY"] = api_key
            st.session_state.api_key_configured = True
            st.rerun()
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant. Answer questions clearly and concisely.",
        height=100
    )
    
    # Model selection
    model_name = st.selectbox(
        "Model",
        ["sonar", "sonar-pro", "sonar-reasoning"]
    )
    
    # Streaming option
    enable_streaming = st.checkbox("Enable Streaming", value=True, 
                                   help="Show response as it's being generated")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Chat Info")
    st.markdown(f"**Messages in chat:** {len(st.session_state.messages)}")
    st.markdown(f"**Model:** {model_name}")
    
    # Refresh button
    if st.button("🔄 Refresh Model", use_container_width=True):
        st.session_state.model = None
        st.rerun()

# Initialize model when API key is available
if api_key and st.session_state.model is None:
    try:
        with st.sidebar:
            with st.spinner("Initializing model..."):
                st.session_state.model = init_chat_model(model_name, model_provider="perplexity")
                st.success("✅ Model initialized!")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize model: {str(e)}")
        st.session_state.model = None

# Display chat messages - using native Streamlit markdown (much faster)
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

# Check if model is ready
if not st.session_state.api_key_configured:
    st.warning("⚠️ Please configure your API key in the sidebar to start chatting.")
    st.stop()

if st.session_state.model is None:
    st.info("🔄 Initializing model... Please wait.")
    st.stop()

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Prepare messages for the model
    messages_for_model = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            messages_for_model.append(HumanMessage(content=msg["content"]))
        else:
            messages_for_model.append(AIMessage(content=msg["content"]))
    
    # Add current user message
    messages_for_model.append(HumanMessage(content=prompt))
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        try:
            if enable_streaming:
                # Streaming response - much faster perceived performance
                full_response = ""
                for chunk in st.session_state.model.stream(messages_for_model):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                # Non-streaming response
                message_placeholder.markdown("Thinking...")
                response = st.session_state.model.invoke(messages_for_model)
                message_placeholder.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    <small>Built with Streamlit, LangChain & Perplexity AI</small><br>
    <small>Streaming enabled for faster responses</small>
    </div>
    """,
    unsafe_allow_html=True
)
