# app.py - DeckChat Pro with Firebase, OpenRouter & Session Management

import streamlit as st
import os
import json
import hashlib
import base64
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="DeckChat | Advanced AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Premium UI/UX Styling (ChatGPT/Gemini Style)
# ----------------------
st.markdown("""
<style>
    /* Base Typography & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #343541; /* ChatGPT Dark Mode Background */
        color: #ECECF1;
    }

    /* Centered Chat Layout */
    .main .block-container {
        max-width: 850px; /* Center column like ChatGPT */
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1.5rem 1rem !important;
    }
    
    /* Assistant Message Background */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10a37f !important; /* OpenAI Green */
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #444654 !important; /* Slightly lighter dark for AI */
        border-radius: 8px;
    }

    /* Input Box Styling */
    .stChatInputContainer {
        padding-bottom: 30px !important;
        background-color: transparent !important;
    }
    div[data-testid="stForm"] {
        border: 1px solid #565869 !important;
        background: #40414f !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input {
        color: white !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #202123 !important;
        border-right: 1px solid #323232;
    }
    
    .new-chat-btn {
        background-color: transparent;
        border: 1px solid #565869;
        color: white;
        padding: 12px;
        border-radius: 6px;
        text-align: left;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s;
        margin-bottom: 15px;
    }
    .new-chat-btn:hover {
        background-color: #2A2B32;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Firebase Initialization
# ----------------------
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            if 'FIREBASE_CONFIG' in st.secrets:
                cred_dict = json.loads(st.secrets['FIREBASE_CONFIG'])
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            else:
                st.error("⚠️ FIREBASE_CONFIG missing in secrets.")
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase Error: {e}")
        return None

db = init_firebase()

# ----------------------
# Core Functions
# ----------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email, password):
    if not db: return "Database error"
    users_ref = db.collection('users')
    if list(users_ref.where('email', '==', email).stream()):
        return "User exists"
    users_ref.add({
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.utcnow().isoformat()
    })
    return "success"

def sign_in(email, password):
    if not db: return False
    docs = list(db.collection('users').where('email', '==', email)
                .where('password_hash', '==', hash_password(password)).stream())
    return True if docs else False

def save_message(user_email, session_id, role, content):
    """Save message tagged with a specific session ID"""
    if db:
        try:
            db.collection('messages').add({
                'user_email': user_email,
                'session_id': session_id,
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow()
            })
        except Exception as e:
            st.warning(f"Failed to sync to cloud: {e}")

def get_session_history(session_id):
    """Retrieve history for a specific chat session only"""
    if not db: return []
    try:
        docs = db.collection('messages').where('session_id', '==', session_id).stream()
        msgs = [{'role': d.to_dict()['role'], 'content': d.to_dict()['content'], 'timestamp': d.to_dict().get('timestamp')} for d in docs]
        msgs.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
        return msgs
    except: return []

# ----------------------
# OpenRouter / LangChain Init
# ----------------------
def get_llm(model_tier="Base"):
    """Initialize OpenRouter with dynamic model routing"""
    if "OPENROUTER_API_KEY" not in st.secrets:
        st.error("⚠️ OPENROUTER_API_KEY missing in secrets")
        return None
        
    # Map tier to OpenRouter model IDs
    model_name = "openai/gpt-3.5-turbo" if model_tier == "Base" else "groq/llama-3.3-70b-versatile"
    
    try:
        return ChatOpenAI(
            model=model_name,
            api_key=st.secrets["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            streaming=True,
            # Optional: OpenRouter headers for analytics
            default_headers={"HTTP-Referer": "https://deckchat.app", "X-Title": "DeckChat"}
        )
    except Exception as e:
        st.error(f"AI Initialization Error: {e}")
        return None

SYSTEM_PROMPT = """You are DeckChat, an advanced, highly capable AI assistant.
Respond thoughtfully using Markdown. Do not reveal your underlying model name (e.g., GPT, Llama) unless directly asked, simply state you are DeckChat, powered by advanced AI."""

# ----------------------
# Helper: Load GIF
# ----------------------
@st.cache_data
def load_gif_base64(gif_path="neon_star_animated.gif"):
    try:
        with open(gif_path, "rb") as f:
            return f"data:image/gif;base64,{base64.b64encode(f.read()).decode()}"
    except: return None

# ----------------------
# UI: Authentication
# ----------------------
def show_auth_screen():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        gif_url = load_gif_base64()
        logo_html = f'<img src="{gif_url}" width="60" style="vertical-align: middle;">' if gif_url else "✦"
        
        st.markdown(f"""
            <div style='text-align: center;'>
                <h1 style='font-size: 2.5rem; margin-bottom: 0;'>{logo_html} Welcome to DeckChat</h1>
                <p style='color: #8e8ea0; margin-bottom: 2rem; font-size: 1.1rem;'>Log in to continue your conversations</p>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            with st.form("login"):
                e = st.text_input("Email address")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Continue", use_container_width=True):
                    if sign_in(e, p):
                        st.session_state.authenticated = True
                        st.session_state.user_email = e
                        # Generate a unique ID for the first chat session
                        st.session_state.session_id = str(uuid.uuid4())
                        st.session_state.messages = []
                        st.rerun()
                    else: st.error("Wrong email or password.")
        with tab2:
            with st.form("signup"):
                ne = st.text_input("Email address")
                np = st.text_input("Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if ne and np:
                        res = sign_up(ne, np)
                        if res == "success": st.success("Account created! You can now log in.")
                        else: st.error(res)

# ----------------------
# UI: Main Application
# ----------------------
def show_chat_interface():
    # Session Management
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        
    if 'model_tier' not in st.session_state:
        st.session_state.model_tier = "Base"

    # Sidebar Navigation
    with st.sidebar:
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Selector
        st.caption("AI Engine")
        new_tier = st.selectbox(
            "Select Model",
            options=["Base", "Pro"],
            index=0 if st.session_state.model_tier == "Base" else 1,
            help="Base: Fast everyday tasks. Pro: Complex reasoning."
        )
        if new_tier != st.session_state.model_tier:
            st.session_state.model_tier = new_tier
            st.rerun()

        st.markdown("<br>"*15, unsafe_allow_html=True)
        st.divider()
        user_display = st.session_state.user_email.split('@')[0]
        st.markdown(f"**👤 {user_display}**")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    # Empty State (When starting a new session)
    if not st.session_state.messages:
        gif_url = load_gif_base64()
        logo_html = f'<img src="{gif_url}" width="45" style="vertical-align: middle; margin-right: 10px;">' if gif_url else "✦ "
        st.markdown(f"""
            <div style='text-align: center; margin-top: 10vh;'>
                <h1 style='color: white; font-size: 2.2rem;'>{logo_html}How can I help you today?</h1>
            </div>
        """, unsafe_allow_html=True)

    # Display Chat History for Current Session Only
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Chat Input Logic
    if prompt := st.chat_input("Message DeckChat..."):
        # Display User Input
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Async save to DB using current session_id
        save_message(st.session_state.user_email, st.session_state.session_id, "user", prompt)

        # Build Context (System Prompt + Last 10 Messages)
        chain_msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        for m in st.session_state.messages[-10:]:
            if m['role'] == 'user': chain_msgs.append(HumanMessage(content=m['content']))
            else: chain_msgs.append(AIMessage(content=m['content']))

        # Stream AI Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            # Initialize chosen model
            llm = get_llm(st.session_state.model_tier)
            
            if llm:
                try:
                    for chunk in llm.stream(chain_msgs):
                        full_response += chunk.content
                        # Add blinking cursor effect
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    save_message(st.session_state.user_email, st.session_state.session_id, "assistant", full_response)
                except Exception as e:
                    st.error(f"Generation Error: {e}")

def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_auth_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
