# app.py - DeckChat Pro with OpenRouter & Dual Models
# A premium AI chatbot experience with Firebase backend

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
    page_title="DeckChat Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/deckchat',
        'Report a bug': 'https://github.com/yourusername/deckchat/issues',
        'About': '# DeckChat Pro\nAdvanced AI Chatbot with Dual Model Support'
    }
)

# ----------------------
# Load Custom CSS
# ----------------------
def load_css():
    """Load custom CSS for Deepseek-like UI"""
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container */
        .stApp {
            background-color: #ffffff;
        }
        
        .main > div {
            padding: 0rem 1rem;
            max-width: 900px;
            margin: 0 auto;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Message styling */
        .chat-message {
            display: flex;
            padding: 24px 16px;
            gap: 16px;
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
        
        .message-content {
            flex: 1;
            line-height: 1.6;
            color: #1f2937;
            font-size: 16px;
            overflow-x: auto;
        }
        
        .message-content p {
            margin: 0 0 12px 0;
        }
        
        .message-content p:last-child {
            margin-bottom: 0;
        }
        
        /* Markdown styling */
        .message-content h1 {
            font-size: 24px;
            font-weight: 700;
            margin: 16px 0 8px 0;
            color: #111827;
        }
        
        .message-content h2 {
            font-size: 20px;
            font-weight: 600;
            margin: 14px 0 6px 0;
            color: #1f2937;
        }
        
        .message-content h3 {
            font-size: 18px;
            font-weight: 600;
            margin: 12px 0 4px 0;
            color: #2d3748;
        }
        
        .message-content h4 {
            font-size: 16px;
            font-weight: 600;
            margin: 10px 0 4px 0;
            color: #374151;
        }
        
        .message-content strong {
            font-weight: 600;
            color: #111827;
        }
        
        .message-content em {
            font-style: italic;
            color: #4b5563;
        }
        
        .message-content ul, .message-content ol {
            margin: 8px 0 12px 20px;
            padding-left: 0;
        }
        
        .message-content li {
            margin: 4px 0;
            line-height: 1.6;
        }
        
        .message-content code {
            background-color: #f3f4f6;
            color: #ef4444;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 14px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        
        .message-content pre {
            background-color: #1e1e2f;
            color: #e5e7eb;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 16px 0;
        }
        
        .message-content pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .message-content blockquote {
            border-left: 4px solid #2563eb;
            padding: 8px 16px;
            margin: 16px 0;
            background-color: #f8fafc;
            color: #334155;
            font-style: italic;
        }
        
        .message-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        
        .message-content th {
            background-color: #f3f4f6;
            font-weight: 600;
            padding: 8px 12px;
            border: 1px solid #e5e7eb;
        }
        
        .message-content td {
            padding: 8px 12px;
            border: 1px solid #e5e7eb;
        }
        
        .message-content hr {
            margin: 24px 0;
            border: none;
            border-top: 1px solid #e5e7eb;
        }
        
        /* Input area */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(to top, #ffffff, #ffffff, transparent);
            padding: 20px 16px 16px;
            max-width: 900px;
            margin: 0 auto;
            z-index: 100;
        }
        
        .input-box {
            display: flex;
            align-items: center;
            gap: 12px;
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 8px 8px 8px 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.2s;
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
            padding: 12px 0;
            background: transparent;
            color: #1f2937;
        }
        
        .input-box input::placeholder {
            color: #9ca3af;
        }
        
        .input-box button {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            min-width: 80px;
        }
        
        .input-box button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37,99,235,0.3);
        }
        
        .input-box button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
            border-right: 1px solid #f0f0f0;
        }
        
        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .new-chat-btn {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px;
            font-weight: 600;
            width: 100%;
            cursor: pointer;
            margin-bottom: 16px;
            transition: all 0.2s;
        }
        
        .new-chat-btn:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37,99,235,0.3);
        }
        
        .session-item {
            padding: 12px 16px;
            margin: 4px 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #4b5563;
        }
        
        .session-item:hover {
            background-color: #f3f4f6;
        }
        
        .session-item.active {
            background: linear-gradient(135deg, #2563eb10, #7c3aed10);
            color: #2563eb;
            border-left: 3px solid #2563eb;
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
        
        .delete-btn {
            opacity: 0;
            transition: opacity 0.2s;
            color: #ef4444;
            font-size: 18px;
            cursor: pointer;
            padding: 0 4px;
        }
        
        .session-item:hover .delete-btn {
            opacity: 1;
        }
        
        /* Model selector */
        .model-selector {
            background-color: #f9fafb;
            border-radius: 12px;
            padding: 8px;
            margin: 16px 0;
            display: flex;
            gap: 8px;
        }
        
        .model-btn {
            flex: 1;
            padding: 10px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            background-color: transparent;
            color: #6b7280;
        }
        
        .model-btn.active {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            box-shadow: 0 2px 8px rgba(37,99,235,0.3);
        }
        
        .model-btn:not(.active):hover {
            background-color: #e5e7eb;
            color: #1f2937;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px;
            color: #6b7280;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
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
            min-height: 70vh;
            text-align: center;
            padding: 2rem;
        }
        
        .welcome-icon {
            font-size: 64px;
            margin-bottom: 24px;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .welcome-title {
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 16px;
        }
        
        .welcome-subtitle {
            color: #6b7280;
            font-size: 18px;
            max-width: 500px;
            margin-bottom: 32px;
        }
        
        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            max-width: 600px;
        }
        
        .chip {
            background-color: #f3f4f6;
            color: #4b5563;
            padding: 12px 20px;
            border-radius: 30px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        
        .chip:hover {
            background-color: #e5e7eb;
            border-color: #2563eb;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
        
        /* Metadata */
        .message-meta {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 8px;
        }
        
        /* Divider */
        .divider {
            margin: 16px 0;
            border: none;
            border-top: 1px solid #f0f0f0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------------
# Firebase Setup
# ----------------------
@st.cache_resource
def init_firebase():
    """Initialize Firebase connection with error handling"""
    try:
        if not firebase_admin._apps:
            if 'FIREBASE_CONFIG' in st.secrets:
                try:
                    cred_dict = json.loads(st.secrets['FIREBASE_CONFIG'])
                except json.JSONDecodeError:
                    cred_dict = st.secrets['FIREBASE_CONFIG']
                
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            else:
                st.warning("⚠️ Firebase configuration not found. Running in local mode.")
                return None
        
        return firestore.client()
    except Exception as e:
        st.error(f"⚠️ Firebase connection error: {str(e)}")
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
            'total_sessions': 0,
            'last_active': datetime.utcnow(),
            'preferences': {
                'model': 'base',
                'theme': 'light'
            }
        }
        
        users_ref.add(user_data)
        return True, "Account created successfully!"
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
            user_ref.update({
                'last_active': datetime.utcnow(),
                'total_sessions': firestore.Increment(1)
            })
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
            'message_count': 0
        })
    except Exception as e:
        st.warning(f"Session creation error: {e}")
    
    return session_id

def get_user_sessions(user_email: str) -> List[Dict]:
    """Get all sessions for user"""
    if not db:
        return []
    
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
    except Exception as e:
        st.warning(f"Error loading sessions: {e}")
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
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                'role': data['role'],
                'content': data['content'],
                'timestamp': data['timestamp'].strftime("%I:%M %p") if data.get('timestamp') else None,
                'model_used': data.get('model_used', 'unknown')
            })
        
        return messages
    except Exception as e:
        st.warning(f"Error loading messages: {e}")
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
            
            # Set title from first user message
            if role == 'user' and session.to_dict().get('message_count', 0) == 0:
                title = content[:40] + "..." if len(content) > 40 else content
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

def get_user_stats(user_email: str) -> Dict:
    """Get user statistics"""
    if not db:
        return {'total_messages': 0, 'total_sessions': 0}
    
    try:
        users_ref = db.collection('users')
        user_docs = list(users_ref.where('email', '==', user_email).stream())
        
        if user_docs:
            data = user_docs[0].to_dict()
            return {
                'total_messages': data.get('total_messages', 0),
                'total_sessions': data.get('total_sessions', 0),
                'created_at': data.get('created_at', datetime.utcnow())
            }
    except:
        pass
    
    return {'total_messages': 0, 'total_sessions': 0}

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
            streaming=True,
            default_headers={
                "HTTP-Referer": "https://deckchat.streamlit.app",
                "X-Title": "DeckChat Pro"
            }
        )
    except Exception as e:
        st.error(f"OpenRouter init error: {e}")
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
    except Exception as e:
        st.error(f"Groq init error: {e}")
        return None

# ----------------------
# System Prompts
# ----------------------
SYSTEM_PROMPT = """You are DeckChat Pro, a helpful AI assistant. 
Be concise, accurate, and friendly. Use markdown formatting for better readability.
When asked about your identity, say you're DeckChat Pro."""

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data
def load_gif_base64(gif_path="neon_star_animated.gif"):
    """Load GIF and convert to base64"""
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
    
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except:
            return dt
    
    if isinstance(dt, datetime):
        now = datetime.utcnow()
        if dt.date() == now.date():
            return dt.strftime("%I:%M %p")
        elif dt.date() == (now - timedelta(days=1)).date():
            return "Yesterday"
        else:
            return dt.strftime("%d %b %Y")
    
    return str(dt)

# ----------------------
# Authentication Screen
# ----------------------
def show_auth_screen():
    """Display login/signup interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        gif_url = load_gif_base64()
        
        st.markdown(f"""
        <div style='text-align: center; margin: 60px 0 40px;'>
            {'<img src="' + gif_url + '" style="width: 100px; margin-bottom: 20px;">' if gif_url else '✨'}
            <h1 style='font-size: 48px; font-weight: 700; 
                      background: linear-gradient(135deg, #2563eb, #7c3aed);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      margin: 0;'>DeckChat Pro</h1>
            <p style='color: #6b7280; font-size: 18px; margin-top: 12px;'>
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
                            st.session_state.current_session = create_session(email, "New Chat")
                            st.session_state.sessions = get_user_sessions(email)
                            st.session_state.messages = []
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
                            if len(new_password) >= 6:
                                success, msg = sign_up(new_email, new_password)
                                if success:
                                    st.success("Account created! Please login.")
                                else:
                                    st.error(msg)
                            else:
                                st.error("Password must be at least 6 characters")
                        else:
                            st.error("Passwords don't match")
                    else:
                        st.warning("Please fill all fields")

# ----------------------
# Main Chat Interface
# ----------------------
def show_chat_interface():
    """Display main chat interface"""
    
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
            <h2 style='margin:0; font-size:24px; font-weight:600; 
                      background: linear-gradient(135deg, #2563eb, #7c3aed);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;'>DeckChat Pro</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # New Chat button
        if st.button("➕ New Chat", key="new_chat", use_container_width=True):
            session_id = create_session(st.session_state.user_email, "New Chat")
            st.session_state.current_session = session_id
            st.session_state.messages = []
            st.session_state.sessions = get_user_sessions(st.session_state.user_email)
            st.rerun()
        
        # Model selector
        st.markdown("### Model")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ Base", key="base_model_btn", use_container_width=True,
                        type="primary" if st.session_state.current_model == "base" else "secondary"):
                st.session_state.current_model = "base"
                st.rerun()
        with col2:
            if st.button("🚀 Pro", key="pro_model_btn", use_container_width=True,
                        type="primary" if st.session_state.current_model == "pro" else "secondary"):
                st.session_state.current_model = "pro"
                st.rerun()
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # Chat sessions
        st.markdown("### Chats")
        
        for session in st.session_state.sessions:
            is_active = session['session_id'] == st.session_state.current_session
            title = session.get('title', 'New Chat')
            created = format_time(session.get('created_at'))
            
            cols = st.columns([1, 8, 1])
            with cols[0]:
                st.markdown("💬")
            with cols[1]:
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
            with cols[2]:
                if st.button("×", key=f"del_{session['session_id']}"):
                    delete_session(st.session_state.user_email, session['session_id'])
                    if session['session_id'] == st.session_state.current_session:
                        # Find another session or create new
                        remaining = get_user_sessions(st.session_state.user_email)
                        if remaining:
                            st.session_state.current_session = remaining[0]['session_id']
                            st.session_state.messages = get_session_messages(
                                st.session_state.user_email, 
                                remaining[0]['session_id']
                            )
                        else:
                            new_id = create_session(st.session_state.user_email, "New Chat")
                            st.session_state.current_session = new_id
                            st.session_state.messages = []
                    st.session_state.sessions = get_user_sessions(st.session_state.user_email)
                    st.rerun()
            
            if created:
                st.caption(f"  {created}")
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # User info
        stats = get_user_stats(st.session_state.user_email)
        st.markdown(f"""
        <div style='padding: 16px; background-color: #f9fafb; border-radius: 12px;'>
            <div style='font-weight:600; color:#1f2937;'>{st.session_state.user_email}</div>
            <div style='display: flex; gap: 16px; margin-top: 8px; color:#6b7280; font-size:12px;'>
                <span>💬 {stats['total_messages']} msgs</span>
                <span>📁 {len(st.session_state.sessions)} chats</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main chat area
    gif_url = load_gif_base64()
    
    # Welcome screen if no messages
    if not st.session_state.messages:
        st.markdown(f"""
        <div class='welcome-container'>
            {'<img src="' + gif_url + '" class="welcome-icon" style="width:80px;">' if gif_url else '✨'}
            <h1 class='welcome-title'>Welcome to DeckChat Pro</h1>
            <p class='welcome-subtitle'>
                Ask me anything! I'm here to help with coding, questions, creative tasks, and more.
            </p>
            <div class='suggestion-chips'>
                <div class='chip' onclick="document.querySelector('input[type=text]').value='What is machine learning?'; document.querySelector('button[type=submit]').click();">🤖 What is machine learning?</div>
                <div class='chip' onclick="document.querySelector('input[type=text]').value='Write a Python function to reverse a string'; document.querySelector('button[type=submit]').click();">🐍 Write Python code</div>
                <div class='chip' onclick="document.querySelector('input[type=text]').value='Explain quantum computing simply'; document.querySelector('button[type=submit]').click();">⚛️ Explain quantum computing</div>
                <div class='chip' onclick="document.querySelector('input[type=text]').value='Tell me a joke'; document.querySelector('button[type=submit]').click();">😄 Tell me a joke</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hidden input handler for chips
        prompt = st.chat_input("Ask me anything...")
        
    else:
        # Display messages
        for message in st.session_state.messages:
            is_user = message['role'] == 'user'
            avatar = "🧑" if is_user else "✨"
            avatar_class = "user-avatar" if is_user else "assistant-avatar"
            message_class = "user-message" if is_user else "assistant-message"
            
            # Use st.chat_message for proper markdown rendering
            with st.chat_message("user" if is_user else "assistant", avatar=avatar):
                st.markdown(message['content'])
                if message.get('timestamp'):
                    st.caption(f"🕒 {message['timestamp']} | 🤖 {message.get('model_used', 'unknown')}")
        
        # Chat input
        prompt = st.chat_input("Ask me anything...")
    
    # Handle prompt
    if prompt:
        # Get current model
        current_model = st.session_state.pro_model if st.session_state.current_model == "pro" else st.session_state.base_model
        
        if not current_model:
            st.error("Model not available. Please check API keys.")
            return
        
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
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
        messages_for_model = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Add conversation context
        for msg in st.session_state.messages[-10:]:  # Last 10 for context
            if msg['role'] == 'user':
                messages_for_model.append(HumanMessage(content=msg['content']))
            else:
                messages_for_model.append(AIMessage(content=msg['content']))
        
        # Generate response
        with st.chat_message("assistant", avatar="✨"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response
                for chunk in current_model.stream(messages_for_model):
                    if hasattr(chunk, 'content'):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                
                # Final response
                message_placeholder.markdown(full_response)
                
                # Save assistant message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.utcnow().strftime("%I:%M %p"),
                    'model_used': 'gpt-3.5' if st.session_state.current_model == 'base' else 'llama-3-70b'
                })
                
                save_message(
                    st.session_state.user_email,
                    st.session_state.current_session,
                    'assistant',
                    full_response,
                    'gpt-3.5' if st.session_state.current_model == 'base' else 'llama-3-70b'
                )
                
                # Update sessions list
                st.session_state.sessions = get_user_sessions(st.session_state.user_email)
                
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")
        
        st.rerun()

# ----------------------
# Main App
# ----------------------
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Initialize Firebase
    global db
    db = init_firebase()
    
    # Route to appropriate screen
    if not st.session_state.authenticated:
        show_auth_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
