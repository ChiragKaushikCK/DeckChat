# app.py - DeckChat with Firebase & Groq AI

import streamlit as st
import os
import json
import hashlib
import base64
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="DeckChat | AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Enhanced UX/UI Styling
# ----------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e);
    }

    /* Chat Message Styling */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 15px !important;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    
    .stChatMessage:hover {
        transform: scale(1.005);
        border-color: #667eea;
    }

    /* User Sidebar Info Card */
    .user-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .stat-pill {
        background: rgba(255,255,255,0.2);
        padding: 5px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        margin-top: 8px;
        display: inline-block;
    }

    /* Input Styling */
    .stChatInputContainer {
        padding-bottom: 20px !important;
    }
    
    div[data-testid="stForm"] {
        border: 1px solid rgba(255,255,255,0.1) !important;
        background: rgba(255,255,255,0.03) !important;
        border-radius: 20px !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
    }

    /* Sidebar Divider */
    hr { margin: 1.5rem 0 !important; opacity: 0.1 !important; }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Firebase Setup
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
                st.error("⚠️ Firebase configuration missing in secrets.")
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase Error: {e}")
        return None

db = init_firebase()

# ----------------------
# Logic Functions (Auth & DB)
# ----------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email, password):
    if not db: return "Database connection error"
    users_ref = db.collection('users')
    if list(users_ref.where('email', '==', email).stream()):
        return "User already exists"
    users_ref.add({
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.utcnow().isoformat(),
        'total_messages': 0
    })
    return "success"

def sign_in(email, password):
    if not db: return False
    users_ref = db.collection('users')
    docs = list(users_ref.where('email', '==', email)
                .where('password_hash', '==', hash_password(password)).stream())
    return True if docs else False

def save_message(user_email, role, content):
    if db:
        try:
            db.collection('messages').add({
                'user_email': user_email,
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow()
            })
            users_ref = db.collection('users')
            user_docs = list(users_ref.where('email', '==', user_email).stream())
            if user_docs:
                user_ref = users_ref.document(user_docs[0].id)
                user_ref.update({'total_messages': firestore.Increment(1)})
        except: pass

def get_chat_history(user_email, limit=30):
    if not db: return []
    try:
        docs = db.collection('messages').where('user_email', '==', user_email).stream()
        msgs = [{'role': d.to_dict()['role'], 'content': d.to_dict()['content'], 'timestamp': d.to_dict().get('timestamp')} for d in docs]
        msgs.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
        return msgs[-limit:]
    except: return []

# ----------------------
# Model Initialization (Groq)
# ----------------------
@st.cache_resource
def init_model():
    """Initialize Groq Llama-3 model"""
    try:
        if "GROQ_API_KEY" in st.secrets:
            # Groq is significantly faster than Perplexity for chat
            return ChatGroq(
                groq_api_key=st.secrets["GROQ_API_KEY"],
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
        else:
            st.error("⚠️ GROQ_API_KEY missing in secrets")
            return None
    except Exception as e:
        st.error(f"Groq Init Error: {e}")
        return None

SYSTEM_PROMPT = """You are DeckChat, a sophisticated AI assistant.
1. Identify only as DeckChat.
2. NEVER mention Groq, Meta, Llama, or Perplexity.
3. Be helpful, concise, and use professional Markdown formatting.
4. If asked about your creator, you were built by the DeckChat team."""

# ----------------------
# UI Components
# ----------------------
@st.cache_data
def load_gif_base64(gif_path="neon_star_animated.gif"):
    try:
        with open(gif_path, "rb") as f:
            return f"data:image/gif;base64,{base64.b64encode(f.read()).decode()}"
    except: return None

def show_auth_screen():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        gif_url = load_gif_base64()
        logo_html = f'<img src="{gif_url}" width="80">' if gif_url else "✦"
        st.markdown(f"""
            <div style='text-align: center;'>
                <h1 style='font-size: 3rem; color: white; margin-bottom: 0;'>{logo_html} DeckChat</h1>
                <p style='color: #888; margin-bottom: 2rem;'>Intelligent. Swift. Private.</p>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        with tab1:
            with st.form("login"):
                e = st.text_input("Email")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Access DeckChat", use_container_width=True):
                    if sign_in(e, p):
                        st.session_state.authenticated, st.session_state.user_email = True, e
                        st.rerun()
                    else: st.error("Invalid credentials")
        with tab2:
            with st.form("signup"):
                ne = st.text_input("New Email")
                np = st.text_input("New Password", type="password")
                cp = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if np == cp and ne:
                        res = sign_up(ne, np)
                        if res == "success": st.success("Account ready! Please login.")
                        else: st.error(res)
                    else: st.warning("Check your details")

def show_chat_interface():
    # Load Model
    if 'model' not in st.session_state:
        st.session_state.model = init_model()
    
    # Load Messages
    if 'messages' not in st.session_state or not st.session_state.messages:
        st.session_state.messages = get_chat_history(st.session_state.user_email)

    # Sidebar UI
    with st.sidebar:
        user_name = st.session_state.user_email.split('@')[0].capitalize()
        st.markdown(f"""
            <div class='user-card'>
                <small>WELCOME BACK</small>
                <h2 style='margin:0;'>{user_name}</h2>
                <div class='stat-pill'>✦ DeckChat Pro User</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Control Center")
        if st.button("🗘 Refresh Chat", use_container_width=True):
            st.session_state.messages = get_chat_history(st.session_state.user_email)
            st.rerun()
            
        if st.button("🗑️ Clear Context", use_container_width=True):
            # Database clearing logic here if needed
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br>"*10, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()
        
        st.caption("Feedback: theconsciouschirag@gmail.com")

    # Main Chat View
    st.markdown("<h2 style='margin-bottom:0;'>DeckChat ✦</h2>", unsafe_allow_html=True)
    st.caption("Powered by Groq Intelligence")

    # Message Display
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Input Logic
    if prompt := st.chat_input("Message DeckChat..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.user_email, "user", prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            # Context window (Last 10 messages)
            chain = [SystemMessage(content=SYSTEM_PROMPT)]
            for m in st.session_state.messages[-10:]:
                if m['role'] == 'user': chain.append(HumanMessage(content=m['content']))
                else: chain.append(AIMessage(content=m['content']))
            
            try:
                for chunk in st.session_state.model.stream(chain):
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message(st.session_state.user_email, "assistant", full_response)
            except Exception as e:
                st.error(f"Engine Error: {e}")

def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_auth_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
