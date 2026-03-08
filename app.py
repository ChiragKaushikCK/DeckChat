# app.py - DeckChat (Deepseek Clone)
# A clean, minimalist chatbot with proper session management

import streamlit as st
import os
import json
import hashlib
import base64
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="DeckChat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed like Deepseek
)

# ----------------------
# Custom CSS for Deepseek-like UI
# ----------------------
def load_css():
    css = """
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background-color: #ffffff;
        }
        
        /* Main container */
        .main > div {
            padding: 0rem 1rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Chat container */
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 1rem;
            height: calc(100vh - 120px);
            overflow-y: auto;
            scroll-behavior: smooth;
        }
        
        /* Message styling */
        .message {
            display: flex;
            padding: 1.5rem 1rem;
            gap: 1rem;
            border-bottom: 1px solid #f0f0f0;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background-color: #ffffff;
        }
        
        .assistant-message {
            background-color: #fafafa;
        }
        
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }
        
        .user-avatar {
            background-color: #e5e7eb;
            color: #4b5563;
        }
        
        .assistant-avatar {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
        }
        
        .content {
            flex: 1;
            line-height: 1.6;
            color: #1f2937;
            font-size: 16px;
            overflow-x: auto;
        }
        
        .content p {
            margin: 0 0 0.75rem 0;
        }
        
        .content p:last-child {
            margin-bottom: 0;
        }
        
        /* Code blocks */
        .content pre {
            background-color: #1e1e2f;
            color: #e5e7eb;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 0.75rem 0;
        }
        
        .content code {
            background-color: #f3f4f6;
            color: #ef4444;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .content pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
        }
        
        /* Input area */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, #ffffff, #ffffff, transparent);
            padding: 1.5rem 1rem 1rem;
            max-width: 900px;
            margin: 0 auto;
            z-index: 100;
        }
        
        .input-box {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 0.5rem 0.5rem 0.5rem 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: box-shadow 0.2s;
        }
        
        .input-box:focus-within {
            border-color: #2563eb;
            box-shadow: 0 4px 20px rgba(37,99,235,0.15);
        }
        
        .input-box input {
            flex: 1;
            border: none;
            outline: none;
            font-size: 16px;
            padding: 0.75rem 0;
            background: transparent;
        }
        
        .input-box button {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.25rem;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        
        .input-box button:hover {
            opacity: 0.9;
        }
        
        .input-box button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Sidebar - Deepseek style */
        .css-1d391kg {
            background-color: #ffffff;
            border-right: 1px solid #f0f0f0;
        }
        
        .sidebar-header {
            padding: 1.5rem;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .new-chat-btn {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem;
            font-weight: 600;
            width: 100%;
            cursor: pointer;
            margin-bottom: 1rem;
            transition: opacity 0.2s;
        }
        
        .new-chat-btn:hover {
            opacity: 0.9;
        }
        
        .session-item {
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: #4b5563;
        }
        
        .session-item:hover {
            background-color: #f3f4f6;
        }
        
        .session-item.active {
            background-color: #eff6ff;
            color: #2563eb;
        }
        
        .session-icon {
            font-size: 18px;
        }
        
        .session-title {
            flex: 1;
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .session-time {
            font-size: 12px;
            color: #9ca3af;
        }
        
        /* Model selector */
        .model-selector {
            background-color: #f9fafb;
            border-radius: 12px;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        
        .model-option {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
        }
        
        .model-option.selected {
            background-color: #2563eb;
            color: white;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            color: #6b7280;
        }
        
        .typing-dots {
            display: flex;
            gap: 0.25rem;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            background-color: #9ca3af;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }
        
        /* Welcome screen */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
            padding: 2rem;
        }
        
        .welcome-icon {
            font-size: 64px;
            margin-bottom: 1rem;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .welcome-title {
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .welcome-subtitle {
            color: #6b7280;
            font-size: 18px;
            max-width: 500px;
            margin-bottom: 2rem;
        }
        
        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            justify-content: center;
            max-width: 600px;
        }
        
        .chip {
            background-color: #f3f4f6;
            color: #4b5563;
            padding: 0.75rem 1.5rem;
            border-radius: 30px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        
        .chip:hover {
            background-color: #e5e7eb;
            border-color: #2563eb;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------------
# Firebase Setup
# ----------------------
@st.cache_resource
def init_firebase():
    """Initialize Firebase connection"""
    try:
        if not firebase_admin._apps:
            if 'FIREBASE_CONFIG' in st.secrets:
                try:
                    cred_dict = json.loads(st.secrets['FIREBASE_CONFIG'])
                except json.JSONDecodeError:
                    cred_dict = st.secrets['FIREBASE_CONFIG']
                
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                return firestore.client()
        else:
            return firestore.client()
    except Exception as e:
        st.error(f"Firebase Error: {e}")
        return None

# ----------------------
# Authentication Functions
# ----------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email: str, password: str) -> tuple:
    if not db:
        return False, "Database connection error"
    
    try:
        users_ref = db.collection('users')
        existing = list(users_ref.where('email', '==', email).stream())
        if existing:
            return False, "User already exists"
        
        user_data = {
            'email': email,
            'password_hash': hash_password(password),
            'created_at': datetime.utcnow(),
            'total_messages': 0,
            'total_sessions': 0
        }
        
        users_ref.add(user_data)
        return True, "Account created!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def sign_in(email: str, password: str) -> tuple:
    if not db:
        return False, "Database connection error"
    
    try:
        users_ref = db.collection('users')
        docs = list(users_ref.where('email', '==', email)
                   .where('password_hash', '==', hash_password(password)).stream())
        
        if docs:
            user_ref = users_ref.document(docs[0].id)
            user_ref.update({'last_active': datetime.utcnow()})
            return True, "Login successful!"
        
        return False, "Invalid credentials"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ----------------------
# Session Management
# ----------------------
def create_session(user_email: str, title: str = "New Chat") -> str:
    """Create a new chat session"""
    if not db:
        return str(uuid.uuid4())
    
    session_id = str(uuid.uuid4())
    try:
        db.collection('sessions').add({
            'user_email': user_email,
            'session_id': session_id,
            'title': title,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'message_count': 0,
            'model_used': st.session_state.get('current_model', 'base')
        })
    except:
        pass
    
    return session_id

def get_user_sessions(user_email: str) -> List[Dict]:
    """Get all sessions for user"""
    if not db:
        # Return mock sessions for demo
        return [
            {
                'session_id': 'current',
                'title': 'Current Chat',
                'created_at': datetime.utcnow(),
                'message_count': 0
            }
        ]
    
    try:
        docs = db.collection('sessions')\
                 .where('user_email', '==', user_email)\
                 .order_by('updated_at', direction=firestore.Query.DESCENDING)\
                 .stream()
        
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            sessions.append({
                'id': doc.id,
                'session_id': data['session_id'],
                'title': data['title'],
                'created_at': data['created_at'],
                'updated_at': data.get('updated_at', data['created_at']),
                'message_count': data.get('message_count', 0)
            })
        
        return sessions
    except:
        return []

def get_session_messages(user_email: str, session_id: str) -> List[Dict]:
    """Get messages for a specific session"""
    if not db:
        return []
    
    try:
        docs = db.collection('messages')\
                 .where('user_email', '==', user_email)\
                 .where('session_id', '==', session_id)\
                 .order_by('timestamp')\
                 .stream()
        
        return [{
            'role': d.to_dict()['role'],
            'content': d.to_dict()['content'],
            'timestamp': d.to_dict()['timestamp']
        } for d in docs]
    except:
        return []

def save_message(user_email: str, session_id: str, role: str, content: str, model_used: str = None):
    """Save message to a specific session"""
    if not db:
        return
    
    try:
        # Save message
        db.collection('messages').add({
            'user_email': user_email,
            'session_id': session_id,
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow(),
            'model_used': model_used
        })
        
        # Update session
        sessions = db.collection('sessions')\
                     .where('user_email', '==', user_email)\
                     .where('session_id', '==', session_id)\
                     .stream()
        
        for session in sessions:
            session_ref = db.collection('sessions').document(session.id)
            session_ref.update({
                'updated_at': datetime.utcnow(),
                'message_count': firestore.Increment(1)
            })
            
            # Update first message as title if needed
            if role == 'user' and session.to_dict().get('message_count', 0) == 0:
                # Use first few words as title
                title = content[:30] + "..." if len(content) > 30 else content
                session_ref.update({'title': title})
        
        # Update user stats
        users_ref = db.collection('users')
        user_docs = list(users_ref.where('email', '==', user_email).stream())
        if user_docs:
            user_ref = users_ref.document(user_docs[0].id)
            user_ref.update({'total_messages': firestore.Increment(1)})
            
    except Exception as e:
        st.warning(f"Save Error: {e}")

def delete_session(user_email: str, session_id: str):
    """Delete a session and its messages"""
    if not db:
        return
    
    try:
        # Delete messages
        messages = db.collection('messages')\
                     .where('user_email', '==', user_email)\
                     .where('session_id', '==', session_id)\
                     .stream()
        for msg in messages:
            msg.reference.delete()
        
        # Delete session
        sessions = db.collection('sessions')\
                     .where('user_email', '==', user_email)\
                     .where('session_id', '==', session_id)\
                     .stream()
        for session in sessions:
            session.reference.delete()
            
    except Exception as e:
        st.error(f"Delete Error: {e}")

# ----------------------
# Model Initialization
# ----------------------
def init_openrouter_model():
    """Initialize OpenRouter model (Base)"""
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
        if not api_key:
            return None
        
        return ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=2000,
            streaming=True
        )
    except:
        return None

def init_groq_model():
    """Initialize Groq model (Pro)"""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not api_key:
            return None
        
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.7,
            max_tokens=4000,
            streaming=True
        )
    except:
        return None

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data
def load_gif_base64(gif_path="neon_star_animated.gif"):
    try:
        with open(gif_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/gif;base64,{data}"
    except:
        return None

def format_time(dt):
    """Format time for display"""
    if not dt:
        return ""
    
    now = datetime.utcnow()
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except:
            return dt
    
    if dt.date() == now.date():
        return dt.strftime("%I:%M %p")
    elif dt.date() == (now - timedelta(days=1)).date():
        return "Yesterday"
    else:
        return dt.strftime("%d/%m/%Y")

def get_session_title(session):
    """Get display title for session"""
    if session.get('title') and session['title'] != "New Chat":
        return session['title']
    
    # If no title, use time
    created = session.get('created_at')
    if created:
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created)
            except:
                pass
        
        if isinstance(created, datetime):
            now = datetime.utcnow()
            if created.date() == now.date():
                return created.strftime("Chat - %I:%M %p")
            elif created.date() == (now - timedelta(days=1)).date():
                return "Yesterday's Chat"
            else:
                return created.strftime("Chat - %d %b")
    
    return "New Chat"

# ----------------------
# Authentication Screen
# ----------------------
def show_auth_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        gif_url = load_gif_base64()
        
        st.markdown(f"""
        <div style='text-align: center; margin: 40px 0;'>
            {'<img src="' + gif_url + '" style="width: 80px; margin-bottom: 20px;">' if gif_url else ''}
            <h1 style='font-size: 48px; font-weight: 700; 
                      background: linear-gradient(135deg, #2563eb, #7c3aed);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      margin: 0;'>DeckChat</h1>
            <p style='color: #6b7280; font-size: 18px; margin-top: 10px;'>
                Your intelligent conversation partner
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="Enter your email")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                if st.form_submit_button("Login", use_container_width=True, type="primary"):
                    if email and password:
                        success, msg = sign_in(email, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user_email = email
                            st.session_state.current_session = create_session(email, "Current Chat")
                            st.session_state.sessions = get_user_sessions(email)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill all fields")
        
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email", placeholder="Enter your email")
                new_password = st.text_input("Password", type="password", placeholder="Create password")
                confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                
                if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                    if new_email and new_password and confirm:
                        if new_password == confirm:
                            success, msg = sign_up(new_email, new_password)
                            if success:
                                st.success("Account created! Please login.")
                            else:
                                st.error(msg)
                        else:
                            st.error("Passwords don't match")
                    else:
                        st.warning("Please fill all fields")

# ----------------------
# Main Chat Interface
# ----------------------
def show_chat_interface():
    # Load CSS
    load_css()
    
    # Initialize models
    if 'base_model' not in st.session_state:
        st.session_state.base_model = init_openrouter_model()
    if 'pro_model' not in st.session_state:
        st.session_state.pro_model = init_groq_model()
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "base"
    
    # Initialize sessions
    if 'sessions' not in st.session_state:
        st.session_state.sessions = get_user_sessions(st.session_state.user_email)
    
    if 'current_session' not in st.session_state:
        # Create new session if none exists
        if not st.session_state.sessions:
            session_id = create_session(st.session_state.user_email, "New Chat")
            st.session_state.current_session = session_id
            st.session_state.sessions = get_user_sessions(st.session_state.user_email)
        else:
            st.session_state.current_session = st.session_state.sessions[0]['session_id']
    
    if 'messages' not in st.session_state:
        st.session_state.messages = get_session_messages(
            st.session_state.user_email, 
            st.session_state.current_session
        )
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-header'>
            <h2 style='margin:0; font-size:24px; font-weight:600;'>DeckChat</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True):
            session_id = create_session(st.session_state.user_email, "New Chat")
            st.session_state.current_session = session_id
            st.session_state.messages = []
            st.session_state.sessions = get_user_sessions(st.session_state.user_email)
            st.rerun()
        
        # Model selector
        st.markdown("### Model")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ Base", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_model == "base" else "secondary"):
                st.session_state.current_model = "base"
                st.rerun()
        with col2:
            if st.button("🚀 Pro", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_model == "pro" else "secondary"):
                st.session_state.current_model = "pro"
                st.rerun()
        
        # Chat sessions
        st.markdown("### Chats")
        
        for session in st.session_state.sessions:
            is_active = session['session_id'] == st.session_state.current_session
            title = get_session_title(session)
            
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                st.markdown("💬")
            with col2:
                if st.button(
                    f"{title[:25]}..." if len(title) > 25 else title,
                    key=f"session_{session['session_id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.current_session = session['session_id']
                    st.session_state.messages = get_session_messages(
                        st.session_state.user_email,
                        session['session_id']
                    )
                    st.rerun()
            with col3:
                if st.button("×", key=f"del_{session['session_id']}"):
                    delete_session(st.session_state.user_email, session['session_id'])
                    if session['session_id'] == st.session_state.current_session:
                        st.session_state.current_session = None
                        st.session_state.messages = []
                    st.session_state.sessions = get_user_sessions(st.session_state.user_email)
                    st.rerun()
        
        st.markdown("---")
        
        # User info
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #f9fafb; border-radius: 12px;'>
            <div style='font-weight:600;'>{st.session_state.user_email}</div>
            <div style='font-size:12px; color:#6b7280; margin-top:4px;'>
                {len(st.session_state.sessions)} chats
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main chat area
    if not st.session_state.messages:
        # Welcome screen
        gif_url = load_gif_base64()
        
        st.markdown(f"""
        <div class='welcome-container'>
            {'<img src="' + gif_url + '" class="welcome-icon" style="width:80px;">' if gif_url else '💬'}
            <h1 class='welcome-title'>Welcome to DeckChat</h1>
            <p class='welcome-subtitle'>
                Your intelligent conversation partner. Ask me anything!
            </p>
            <div class='suggestion-chips'>
                <span class='chip' onclick='navigator.clipboard.writeText("What is machine learning?")'>What is machine learning?</span>
                <span class='chip' onclick='navigator.clipboard.writeText("Write a Python function")'>Write a Python function</span>
                <span class='chip' onclick='navigator.clipboard.writeText("Explain quantum computing")'>Explain quantum computing</span>
                <span class='chip' onclick='navigator.clipboard.writeText("Tell me a joke")'>Tell me a joke</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle suggestion clicks via session state
        suggestion = st.selectbox(
            "Try asking:",
            ["", "What is machine learning?", "Write a Python function", 
             "Explain quantum computing", "Tell me a joke"],
            index=0,
            label_visibility="collapsed"
        )
        
        if suggestion:
            prompt = suggestion
        else:
            prompt = st.chat_input("Ask me anything...")
    else:
        # Display messages
        for message in st.session_state.messages:
            is_user = message['role'] == 'user'
            avatar = "🧑" if is_user else "✨"
            avatar_class = "user-avatar" if is_user else "assistant-avatar"
            message_class = "user-message" if is_user else "assistant-message"
            
            st.markdown(f"""
            <div class='message {message_class}'>
                <div class='avatar {avatar_class}'>
                    {avatar}
                </div>
                <div class='content'>
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        prompt = st.chat_input("Ask me anything...")
    
    # Handle prompt
    if prompt:
        # Get current model
        model = st.session_state.pro_model if st.session_state.current_model == "pro" else st.session_state.base_model
        
        if not model:
            st.error("Model not available. Please check your API keys.")
            return
        
        # Display user message
        st.markdown(f"""
        <div class='message user-message'>
            <div class='avatar user-avatar'>🧑</div>
            <div class='content'>{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Save user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        save_message(
            st.session_state.user_email,
            st.session_state.current_session,
            'user',
            prompt,
            'gpt-3.5' if st.session_state.current_model == 'base' else 'llama-3-70b'
        )
        
        # Prepare messages for model
        system_prompt = """You are DeckChat, a helpful AI assistant. Be concise, accurate, and friendly."""
        messages_for_model = [SystemMessage(content=system_prompt)]
        
        for msg in st.session_state.messages[-10:]:  # Last 10 for context
            if msg['role'] == 'user':
                messages_for_model.append(HumanMessage(content=msg['content']))
            else:
                messages_for_model.append(AIMessage(content=msg['content']))
        
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class='typing-indicator'>
            <div class='avatar assistant-avatar'>✨</div>
            <div class='typing-dots'>
                <span></span><span></span><span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        try:
            response = ""
            for chunk in model.stream(messages_for_model):
                if hasattr(chunk, 'content'):
                    response += chunk.content
            
            # Remove typing indicator
            typing_placeholder.empty()
            
            # Display response
            st.markdown(f"""
            <div class='message assistant-message'>
                <div class='avatar assistant-avatar'>✨</div>
                <div class='content'>{response}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Save response
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            save_message(
                st.session_state.user_email,
                st.session_state.current_session,
                'assistant',
                response,
                'gpt-3.5' if st.session_state.current_model == 'base' else 'llama-3-70b'
            )
            
            # Update sessions list
            st.session_state.sessions = get_user_sessions(st.session_state.user_email)
            
        except Exception as e:
            typing_placeholder.empty()
            st.error(f"Error: {str(e)}")
        
        st.rerun()

# ----------------------
# Main App
# ----------------------
def main():
    global db
    db = init_firebase()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_auth_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
