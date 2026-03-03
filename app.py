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
    page_title="DeckChat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Enhanced UI Styling
# ----------------------
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 18px;
        margin-bottom: 12px;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 18px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }

    .stat-box {
        background: rgba(255,255,255,0.15);
        padding: 10px;
        border-radius: 10px;
        margin: 6px 0;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
                st.error("⚠️ Firebase configuration not found in secrets")
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase Error: {e}")
        return None

db = init_firebase()

# ----------------------
# Auth Functions
# ----------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email, password):
    if not db:
        return "Database connection error"

    users_ref = db.collection('users')
    existing = list(users_ref.where('email', '==', email).stream())
    if existing:
        return "User already exists"

    users_ref.add({
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.utcnow().isoformat(),
        'total_messages': 0
    })
    return "success"

def sign_in(email, password):
    if not db:
        return False

    users_ref = db.collection('users')
    docs = list(users_ref.where('email', '==', email)
                .where('password_hash', '==', hash_password(password)).stream())
    return True if docs else False

# ----------------------
# Database Functions
# ----------------------
def save_message(user_email, role, content):
    if db:
        db.collection('messages').add({
            'user_email': user_email,
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow()
        })

def get_chat_history(user_email, limit=50):
    if not db:
        return []

    docs = db.collection('messages')\
             .where('user_email', '==', user_email)\
             .stream()

    messages = [{'role': d.to_dict()['role'],
                 'content': d.to_dict()['content'],
                 'timestamp': d.to_dict().get('timestamp')}
                for d in docs]

    messages.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
    return messages[-limit:] if len(messages) > limit else messages

# ----------------------
# Groq Model Initialization
# ----------------------
@st.cache_resource
def init_model():
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        else:
            st.error("⚠️ GROQ_API_KEY not found in secrets")
            return None

        model = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7,
            streaming=True
        )

        return model

    except Exception as e:
        st.error(f"Model Init Error: {e}")
        return None

# ----------------------
# System Prompt
# ----------------------
SYSTEM_PROMPT = """You are DeckChat, a helpful and intelligent AI assistant.

CRITICAL RULES:
1. If user asks your identity, respond: "I am DeckChat, your AI assistant."
2. Never mention Groq or any backend provider.
3. Be clear, helpful, friendly, and concise.
4. Use markdown formatting when helpful.
"""

# ----------------------
# Main Chat Interface
# ----------------------
def show_chat_interface():

    if 'model' not in st.session_state:
        with st.spinner("⚡ Connecting to Groq AI Engine..."):
            st.session_state.model = init_model()

    if 'messages' not in st.session_state:
        st.session_state.messages = get_chat_history(st.session_state.user_email)

    st.title("✦ DeckChat")
    st.caption("Fast • Intelligent • Powered by Groq")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input("Ask me anything..."):

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.user_email, "user", prompt)

        messages_for_model = [SystemMessage(content=SYSTEM_PROMPT)]

        for msg in st.session_state.messages[-10:]:
            if msg['role'] == 'user':
                messages_for_model.append(HumanMessage(content=msg['content']))
            else:
                messages_for_model.append(AIMessage(content=msg['content']))

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in st.session_state.model.stream(messages_for_model):
                    full_response += chunk.content
                    message_placeholder.markdown(
                        full_response + " <span style='opacity:0.5;'>▍</span>",
                        unsafe_allow_html=True
                    )

                message_placeholder.markdown(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                save_message(st.session_state.user_email, "assistant", full_response)

            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)}")

# ----------------------
# Authentication Screen
# ----------------------
def show_auth_screen():

    st.title("✦ DeckChat")
    st.subheader("Login to continue")

    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if sign_in(email, password):
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid credentials")

# ----------------------
# Main App
# ----------------------
def main():

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        show_auth_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
