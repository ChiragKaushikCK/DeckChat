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

# Function to format text with markdown
def format_message(text):
    """Format message with proper markdown handling"""
    import re
    
    # Handle code blocks first (to prevent interference with other patterns)
    text = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', text, flags=re.DOTALL)
    
    # Handle headers
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    # Handle bold and italics
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(?!\*)(.*?)\*', r'<em>\1</em>', text)
    
    # Handle inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Handle lists
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    list_type = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for bullet list
        if line_stripped.startswith('- ') or line_stripped.startswith('* '):
            if not in_list or list_type != 'ul':
                if in_list:
                    formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                formatted_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            formatted_lines.append(f'<li>{line_stripped[2:]}</li>')
        
        # Check for numbered list
        elif re.match(r'^\d+\. ', line_stripped):
            if not in_list or list_type != 'ol':
                if in_list:
                    formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                formatted_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            formatted_lines.append(f'<li>{line_stripped[line_stripped.find(". ")+2:]}</li>')
        
        # Not a list item
        else:
            if in_list:
                formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                in_list = False
                list_type = None
            
            # Handle blockquotes
            if line_stripped.startswith('> '):
                formatted_lines.append(f'<blockquote>{line_stripped[2:]}</blockquote>')
            elif line_stripped:
                formatted_lines.append(f'<p>{line_stripped}</p>')
    
    # Close any open list
    if in_list:
        formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
    
    text = '\n'.join(formatted_lines)
    
    # Clean up empty paragraphs
    text = text.replace('<p></p>', '')
    
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
    for msg in st.session_state.messages[:-1]:  # Exclude current user message
        if msg["role"] == "user":
            messages_for_model.append(HumanMessage(content=msg["content"]))
        else:
            messages_for_model.append(AIMessage(content=msg["content"]))
    
    # Add current user message
    messages_for_model.append(HumanMessage(content=prompt))
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Get response from model
            response = st.session_state.model.invoke(messages_for_model)
            
            # Format and display the response
            formatted_response = format_message(response.content)
            message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
            
            # Add assistant response to chat history
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
    <small>The model will automatically load when API key is configured</small>
    </div>
    """,
    unsafe_allow_html=True
)
