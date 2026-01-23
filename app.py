# app.py - Streamlit Chatbot with LangChain & Perplexity

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

# Custom CSS for better formatting
st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-left: 5px solid #4f8bf9;
    }
    .chat-message.assistant {
        background-color: #2d3746;
        border-left: 5px solid #00d4aa;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1rem;
    }
    .message-content {
        line-height: 1.5;
        font-size: 16px;
    }
    .message-content h1, .message-content h2, .message-content h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .message-content ul, .message-content ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    .message-content li {
        margin-bottom: 0.25rem;
    }
    .message-content code {
        background-color: rgba(0,0,0,0.2);
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    }
    .message-content pre {
        background-color: rgba(0,0,0,0.3);
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        margin: 1rem 0;
    }
    .message-content blockquote {
        border-left: 3px solid #4f8bf9;
        padding-left: 1rem;
        margin-left: 0;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🤖 AI Chatbot")
st.markdown("Powered by LangChain & Perplexity AI")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Perplexity API Key",
        type="password",
        help="Get your API key from https://www.perplexity.ai"
    )
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant. Answer questions clearly and concisely with proper formatting.",
        height=100
    )
    
    # Model selection
    model_name = st.selectbox(
        "Model",
        ["sonar", "sonar-pro", "sonar-reasoning"]
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None

# Initialize model when API key is provided
if api_key and st.session_state.model is None:
    try:
        os.environ["PPLX_API_KEY"] = api_key
        st.session_state.model = init_chat_model(model_name, model_provider="perplexity")
        st.sidebar.success("✅ Model initialized!")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize model: {str(e)}")

# Function to format text with markdown
def format_message(text):
    """Format message with proper markdown handling"""
    # Convert markdown to HTML for better formatting
    import re
    
    # Handle headers
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    # Handle bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Handle italics
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Handle code blocks
    text = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    
    # Handle inline code
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # Handle lists (basic)
    lines = text.split('\n')
    in_list = False
    formatted_lines = []
    
    for line in lines:
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{line.strip()[2:]}</li>')
        elif line.strip().startswith('1. '):
            if not in_list:
                formatted_lines.append('<ol>')
                in_list = True
            formatted_lines.append(f'<li>{line.strip()[3:]}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>' if line.strip().startswith('- ') or line.strip().startswith('* ') else '</ol>')
                in_list = False
            formatted_lines.append(line)
    
    if in_list:
        formatted_lines.append('</ul>' if formatted_lines[-1].startswith('<li>') and '</ol>' not in formatted_lines[-1] else '</ol>')
    
    text = '\n'.join(formatted_lines)
    
    # Handle paragraphs
    text = text.replace('\n\n', '</p><p>')
    text = f'<p>{text}</p>'
    
    # Handle blockquotes
    text = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    
    return text

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"], unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                formatted_content = format_message(message["content"])
                st.markdown(formatted_content, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check if model is initialized
    if not api_key:
        st.error("Please enter your Perplexity API key in the sidebar first.")
        st.stop()
    
    if st.session_state.model is None:
        st.error("Model not initialized. Please check your API key.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Prepare messages for the model
    messages_for_model = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages[:-1]:  # Exclude current user message
        if msg["role"] == "user":
            messages_for_model.append(HumanMessage(content=msg["content"]))
        else:
            messages_for_model.append(AIMessage(content=msg["content"]))
    
    # Add current user message
    messages_for_model.append(HumanMessage(content=prompt))
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                # Get response from model
                response = st.session_state.model.invoke(messages_for_model)
                
                # Format and display the response
                formatted_response = format_message(response.content)
                st.markdown(formatted_response, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    <small>Built with Streamlit, LangChain & Perplexity AI</small>
    </div>
    """,
    unsafe_allow_html=True
)
