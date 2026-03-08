# app.py - DeckChat Pro with OpenRouter & Dual Models
# A premium AI chatbot experience with Firebase backend

import streamlit as st
import os
import json
import hashlib
import base64
import time
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# For better streaming
from streamlit.runtime.scriptrunner import get_script_run_ctx

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
# Load External CSS
# ----------------------
def load_css():
    """Load custom CSS for better UI"""
    css = """
    <style>
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .main > div {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 20px;
            margin: 10px;
            backdrop-filter: blur(10px);
        }
        
        /* Chat Messages */
        .stChatMessage {
            padding: 1.2rem;
            border-radius: 20px;
            margin-bottom: 15px;
            animation: slideIn 0.3s ease-out;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        @keyframes slideIn {
            from { 
                opacity: 0; 
                transform: translateX(-20px);
            }
            to { 
                opacity: 1; 
                transform: translateX(0);
            }
        }
        
        /* User Message */
        [data-testid="chat-message-user"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 20%;
        }
        
        /* Assistant Message */
        [data-testid="chat-message-assistant"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2d3748;
            margin-right: 20%;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        }
        
        .sidebar-content {
            color: white;
            padding: 20px;
        }
        
        /* User Profile Card */
        .user-profile {
            background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 20px;
            color: white;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); }
            to { box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6); }
        }
        
        /* Stats Cards */
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            backdrop-filter: blur(5px);
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            margin: 10px 0;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            margin: 0 3px;
            background: #667eea;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        /* Input Box */
        .stTextInput > div > div > input {
            border-radius: 25px !important;
            border: 2px solid #667eea !important;
            padding: 15px 20px !important;
            font-size: 16px !important;
            background: white !important;
        }
        
        .stTextInput > div > div > input:focus {
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 25px !important;
            background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 25px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 15px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 25px;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        /* Code Blocks */
        pre {
            border-radius: 10px !important;
            background: #1e1e2f !important;
            padding: 15px !important;
        }
        
        /* Animations */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(120deg, #764ba2 0%, #667eea 100%);
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
                    # Try to parse as JSON string
                    cred_dict = json.loads(st.secrets['FIREBASE_CONFIG'])
                except json.JSONDecodeError:
                    # If it's already a dict
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
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email: str, password: str) -> tuple:
    """Create new user account"""
    if not db:
        return False, "Database connection error"
    
    try:
        users_ref = db.collection('users')
        
        # Check if user exists
        existing = list(users_ref.where('email', '==', email).stream())
        if existing:
            return False, "User already exists"
        
        # Create new user
        user_data = {
            'email': email,
            'password_hash': hash_password(password),
            'created_at': datetime.utcnow(),
            'total_messages': 0,
            'total_sessions': 0,
            'last_active': datetime.utcnow(),
            'preferences': {
                'model': 'base',
                'theme': 'light',
                'notifications': True
            }
        }
        
        users_ref.add(user_data)
        return True, "Account created successfully!"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def sign_in(email: str, password: str) -> tuple:
    """Authenticate user"""
    if not db:
        return False, "Database connection error"
    
    try:
        users_ref = db.collection('users')
        docs = list(users_ref.where('email', '==', email)
                   .where('password_hash', '==', hash_password(password)).stream())
        
        if docs:
            # Update last active
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
# Database Functions
# ----------------------
def save_message(user_email: str, role: str, content: str, model_used: str = None):
    """Save message to Firebase with metadata"""
    if db:
        try:
            message_data = {
                'user_email': user_email,
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow(),
                'model_used': model_used,
                'tokens': len(content.split())  # Approximate token count
            }
            
            db.collection('messages').add(message_data)
            
            # Update user message count
            users_ref = db.collection('users')
            user_docs = list(users_ref.where('email', '==', user_email).stream())
            if user_docs:
                user_ref = users_ref.document(user_docs[0].id)
                user_ref.update({
                    'total_messages': firestore.Increment(1),
                    'last_active': datetime.utcnow()
                })
                    
        except Exception as e:
            st.warning(f"⚠️ Could not save message: {str(e)}")

def get_chat_history(user_email: str, limit: int = 50) -> List[Dict]:
    """Get user's chat history with pagination support"""
    if not db:
        return []
    
    try:
        # Get messages sorted by timestamp
        docs = db.collection('messages')\
                 .where('user_email', '==', user_email)\
                 .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                 .limit(limit)\
                 .stream()
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                'role': data['role'],
                'content': data['content'],
                'timestamp': data['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if data.get('timestamp') else None,
                'model_used': data.get('model_used', 'unknown')
            })
        
        # Return in chronological order
        return list(reversed(messages))
        
    except Exception as e:
        st.warning(f"⚠️ Could not load history: {str(e)}")
        return []

def get_user_stats(user_email: str) -> Dict:
    """Get comprehensive user statistics"""
    if not db:
        return {
            'total_messages': 0,
            'total_sessions': 0,
            'created_at': 'Unknown',
            'last_active': 'Unknown',
            'avg_messages_per_session': 0,
            'preferences': {}
        }
    
    try:
        users_ref = db.collection('users')
        user_docs = list(users_ref.where('email', '==', user_email).stream())
        
        if user_docs:
            data = user_docs[0].to_dict()
            
            # Get message count for today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_messages = list(db.collection('messages')
                                .where('user_email', '==', user_email)
                                .where('timestamp', '>=', today_start)
                                .stream())
            
            total_sessions = data.get('total_sessions', 1)
            total_messages = data.get('total_messages', 0)
            
            return {
                'total_messages': total_messages,
                'total_sessions': total_sessions,
                'created_at': data.get('created_at', datetime.utcnow()),
                'last_active': data.get('last_active', datetime.utcnow()),
                'today_messages': len(today_messages),
                'avg_messages_per_session': round(total_messages / max(total_sessions, 1), 1),
                'preferences': data.get('preferences', {})
            }
    
    except Exception as e:
        st.warning(f"⚠️ Could not load stats: {str(e)}")
    
    return {
        'total_messages': 0,
        'total_sessions': 0,
        'created_at': datetime.utcnow(),
        'last_active': datetime.utcnow(),
        'today_messages': 0,
        'avg_messages_per_session': 0,
        'preferences': {}
    }

def update_user_preferences(user_email: str, preferences: Dict):
    """Update user preferences"""
    if db:
        try:
            users_ref = db.collection('users')
            user_docs = list(users_ref.where('email', '==', user_email).stream())
            if user_docs:
                user_ref = users_ref.document(user_docs[0].id)
                user_ref.update({'preferences': preferences})
                return True
        except Exception as e:
            st.warning(f"⚠️ Could not update preferences: {str(e)}")
    return False

def clear_user_history(user_email: str):
    """Clear all chat history for user"""
    if db:
        try:
            batch = db.batch()
            docs = db.collection('messages').where('user_email', '==', user_email).stream()
            for doc in docs:
                batch.delete(doc.reference)
            batch.commit()
            return True
        except Exception as e:
            st.error(f"❌ Clear Error: {str(e)}")
    return False

def search_conversations(user_email: str, query: str) -> List[Dict]:
    """Search through user's chat history"""
    if not db:
        return []
    
    try:
        # Get all messages and search in memory (Firestore doesn't support text search natively)
        docs = db.collection('messages')\
                 .where('user_email', '==', user_email)\
                 .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                 .limit(500)\
                 .stream()
        
        results = []
        for doc in docs:
            data = doc.to_dict()
            content = data.get('content', '').lower()
            if query.lower() in content:
                results.append({
                    'role': data['role'],
                    'content': data['content'],
                    'timestamp': data['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if data.get('timestamp') else None,
                    'model_used': data.get('model_used', 'unknown')
                })
        
        return results[:20]  # Return top 20 matches
        
    except Exception as e:
        st.warning(f"⚠️ Search error: {str(e)}")
        return []

# ----------------------
# Model Initialization
# ----------------------
class StreamHandler(BaseCallbackHandler):
    """Custom stream handler for better streaming experience"""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

def init_openrouter_model(model_name: str = "openai/gpt-3.5-turbo"):
    """Initialize OpenRouter model"""
    try:
        # Get API key from secrets
        api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
        
        if not api_key:
            st.error("❌ OpenRouter API key not found in secrets")
            return None
        
        # Initialize model
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
            temperature=0.7,
            max_tokens=2000,
            callbacks=[StreamingStdOutCallbackHandler()],
            default_headers={
                "HTTP-Referer": "https://deckchat.streamlit.app",
                "X-Title": "DeckChat Pro"
            }
        )
        return model
        
    except Exception as e:
        st.error(f"❌ Model initialization error: {str(e)}")
        return None

def init_groq_model():
    """Initialize Groq model for pro version"""
    try:
        # Get API key from secrets
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        
        if not api_key:
            st.error("❌ Groq API key not found in secrets")
            return None
        
        # Initialize Groq model
        from langchain_groq import ChatGroq
        
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.7,
            max_tokens=4000,
            streaming=True
        )
        return model
        
    except Exception as e:
        st.error(f"❌ Groq initialization error: {str(e)}")
        return None

# ----------------------
# System Prompts
# ----------------------
SYSTEM_PROMPTS = {
    "default": """You are DeckChat Pro, an advanced AI assistant created to help users with any task.
    
Core Identity:
- Name: DeckChat Pro
- Creator: Built with cutting-edge AI technology
- Purpose: To provide helpful, accurate, and engaging responses

Guidelines:
1. Always identify yourself as "DeckChat Pro" when asked about your identity
2. Never claim to be another AI system (Claude, GPT, etc.)
3. Provide comprehensive, well-structured responses
4. Use markdown formatting for better readability
5. Be friendly, professional, and empathetic
6. Acknowledge limitations when appropriate
7. Encourage follow-up questions

Remember: You are DeckChat Pro - helpful, harmless, and honest.""",

    "code": """You are DeckChat Pro - Code Specialist.
Focus on providing clean, efficient code solutions with:
- Proper documentation
- Best practices
- Error handling
- Performance considerations
- Language-specific conventions""",

    "creative": """You are DeckChat Pro - Creative Partner.
Help users with:
- Story writing
- Poetry
- Creative ideas
- Brainstorming
- Artistic concepts
Be imaginative and inspiring!""",

    "academic": """You are DeckChat Pro - Academic Tutor.
Provide:
- Detailed explanations
- Step-by-step reasoning
- Citations when relevant
- Study tips
- Clear examples
Make complex topics accessible!"""
}

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
    except FileNotFoundError:
        # Return None if file not found (will use text fallback)
        return None

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, datetime):
        now = datetime.utcnow()
        diff = now - timestamp
        
        if diff.days == 0:
            if diff.seconds < 60:
                return "Just now"
            elif diff.seconds < 3600:
                return f"{diff.seconds // 60} minutes ago"
            else:
                return f"{diff.seconds // 3600} hours ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return timestamp.strftime("%Y-%m-%d")
    
    return "Unknown"

def export_chat_history(messages: List[Dict], format: str = "txt"):
    """Export chat history in various formats"""
    if format == "txt":
        content = ""
        for msg in messages:
            timestamp = msg.get('timestamp', '')
            role = msg['role'].upper()
            content_text = msg['content']
            content += f"[{timestamp}] {role}:\n{content_text}\n\n{'='*50}\n\n"
        return content
    
    elif format == "json":
        return json.dumps(messages, indent=2, default=str)
    
    elif format == "md":
        content = "# Chat Export\n\n"
        for msg in messages:
            timestamp = msg.get('timestamp', '')
            role = msg['role']
            content_text = msg['content']
            content += f"## {role.capitalize()} - {timestamp}\n\n{content_text}\n\n---\n\n"
        return content
    
    return ""

# ----------------------
# Authentication Screen
# ----------------------
def show_auth_screen():
    """Display enhanced login/signup interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Animated header with GIF
        gif_url = load_gif_base64()
        
        if gif_url:
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 40px; animation: fadeIn 1s;'>
                <img src="{gif_url}" style="width: 100px; height: 100px; margin-bottom: 20px;">
                <h1 style='background: linear-gradient(120deg, #667eea, #764ba2); 
                          -webkit-background-clip: text; 
                          -webkit-text-fill-color: transparent;
                          font-size: 48px; margin: 0;'>DeckChat Pro</h1>
                <p style='color: #666; font-size: 18px;'>Your Intelligent AI Companion</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align: center; margin-bottom: 40px;'>
                <h1 style='color: #667eea; font-size: 48px;'>✨ DeckChat Pro</h1>
                <p style='color: #666; font-size: 18px;'>Your Intelligent AI Companion</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create tabs for login/signup
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form", clear_on_submit=True):
                email = st.text_input(
                    "📧 Email Address",
                    placeholder="Enter your email",
                    help="We'll never share your email"
                )
                
                password = st.text_input(
                    "🔑 Password",
                    type="password",
                    placeholder="Enter your password"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    submit = st.form_submit_button(
                        "Login",
                        use_container_width=True,
                        type="primary"
                    )
                
                with col2:
                    if st.form_submit_button("Reset Password", use_container_width=True):
                        st.info("Password reset feature coming soon!")
                
                if submit:
                    if email and password:
                        with st.spinner("🔄 Authenticating..."):
                            success, message = sign_in(email, password)
                            
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.user_email = email
                                st.session_state.messages = []
                                st.session_state.current_model = "base"
                                st.success(f"✅ {message}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("⚠️ Please fill in all fields")
        
        with tab2:
            with st.form("signup_form", clear_on_submit=True):
                new_email = st.text_input(
                    "📧 Email Address",
                    placeholder="Enter your email",
                    key="signup_email"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    new_password = st.text_input(
                        "🔑 Password",
                        type="password",
                        placeholder="Create password",
                        help="Minimum 6 characters"
                    )
                
                with col2:
                    confirm_password = st.text_input(
                        "🔒 Confirm Password",
                        type="password",
                        placeholder="Confirm password"
                    )
                
                # Terms and conditions
                terms = st.checkbox(
                    "I agree to the Terms of Service and Privacy Policy",
                    help="Read our terms before signing up"
                )
                
                submit = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                    type="primary"
                )
                
                if submit:
                    if new_email and new_password and confirm_password:
                        if new_password == confirm_password:
                            if len(new_password) >= 6:
                                if terms:
                                    with st.spinner("🔄 Creating account..."):
                                        success, message = sign_up(new_email, new_password)
                                        
                                        if success:
                                            st.success(f"✅ {message}")
                                            st.balloons()
                                            st.info("Please login with your new account")
                                        else:
                                            st.error(f"❌ {message}")
                                else:
                                    st.warning("⚠️ Please accept the terms to continue")
                            else:
                                st.error("❌ Password must be at least 6 characters")
                        else:
                            st.error("❌ Passwords don't match")
                    else:
                        st.warning("⚠️ Please fill in all fields")

# ----------------------
# Settings Modal
# ----------------------
def show_settings_modal():
    """Display settings modal"""
    with st.expander("⚙️ Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎨 Appearance")
            theme = st.selectbox(
                "Theme",
                options=["Light", "Dark", "System"],
                index=0
            )
            
            font_size = st.slider(
                "Font Size",
                min_value=12,
                max_value=24,
                value=16
            )
        
        with col2:
            st.subheader("🤖 AI Settings")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=2000,
                step=100
            )
        
        # Save preferences
        if st.button("💾 Save Settings", use_container_width=True):
            preferences = {
                'theme': theme.lower(),
                'font_size': font_size,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            if update_user_preferences(st.session_state.user_email, preferences):
                st.success("✅ Settings saved!")
                st.session_state.preferences = preferences
            else:
                st.error("❌ Failed to save settings")

# ----------------------
# Main Chat Interface
# ----------------------
def show_chat_interface():
    """Display enhanced main chat interface"""
    
    # Load CSS
    load_css()
    
    # Initialize model based on selection
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "base"
    
    # Initialize models
    if 'base_model' not in st.session_state:
        with st.spinner("🚀 Initializing Base Model (GPT-3.5)..."):
            st.session_state.base_model = init_openrouter_model("openai/gpt-3.5-turbo")
    
    if 'pro_model' not in st.session_state:
        with st.spinner("🚀 Initializing Pro Model (Llama-3-70B)..."):
            st.session_state.pro_model = init_groq_model()
    
    # Load chat history
    if 'messages' not in st.session_state or not st.session_state.messages:
        with st.spinner("📚 Loading your conversations..."):
            st.session_state.messages = get_chat_history(st.session_state.user_email)
    
    # Load preferences
    if 'preferences' not in st.session_state:
        stats = get_user_stats(st.session_state.user_email)
        st.session_state.preferences = stats.get('preferences', {})
    
    # Sidebar
    with st.sidebar:
        # User Profile Section
        stats = get_user_stats(st.session_state.user_email)
        
        st.markdown(f"""
        <div class='user-profile fade-in'>
            <div style='font-size: 48px; margin-bottom: 10px;'>
                {st.session_state.user_email[0].upper()}
            </div>
            <h3 style='margin: 0;'>{st.session_state.user_email.split('@')[0]}</h3>
            <p style='opacity: 0.8; margin: 5px 0;'>{st.session_state.user_email}</p>
            <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
                <div class='stat-card'>
                    <div style='font-size: 24px; font-weight: bold;'>{stats['total_messages']}</div>
                    <div style='font-size: 12px;'>Total Msgs</div>
                </div>
                <div class='stat-card'>
                    <div style='font-size: 24px; font-weight: bold;'>{stats['today_messages']}</div>
                    <div style='font-size: 12px;'>Today</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        model_option = st.radio(
            "Choose your AI model:",
            options=[
                "Base (GPT-3.5) - Fast & Efficient",
                "Pro (Llama-3-70B) - Most Capable"
            ],
            index=0 if st.session_state.current_model == "base" else 1,
            help="Base: OpenRouter GPT-3.5 | Pro: Groq Llama-3-70B"
        )
        
        # Update current model
        new_model = "base" if "Base" in model_option else "pro"
        if new_model != st.session_state.current_model:
            st.session_state.current_model = new_model
            st.rerun()
        
        st.divider()
        
        # System Prompt Selection
        st.subheader("🎯 Assistant Persona")
        
        persona = st.selectbox(
            "Choose persona:",
            options=list(SYSTEM_PROMPTS.keys()),
            format_func=lambda x: x.capitalize(),
            help="Different personas for different tasks"
        )
        
        st.session_state.current_persona = persona
        
        st.divider()
        
        # Chat Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Stats", use_container_width=True):
                st.session_state.show_stats = not st.session_state.get('show_stats', False)
        
        # Export options
        export_format = st.selectbox(
            "Export format:",
            options=["txt", "md", "json"],
            format_func=lambda x: x.upper()
        )
        
        if st.button("📥 Export Chat", use_container_width=True):
            if st.session_state.messages:
                content = export_chat_history(st.session_state.messages, export_format)
                st.download_button(
                    "⬇️ Download",
                    content,
                    f"chat_export.{export_format}",
                    use_container_width=True
                )
        
        st.divider()
        
        # Search
        st.subheader("🔍 Search Conversations")
        search_query = st.text_input("Search...", placeholder="Enter keywords...")
        
        if search_query:
            with st.spinner("Searching..."):
                results = search_conversations(st.session_state.user_email, search_query)
                if results:
                    st.info(f"Found {len(results)} results")
                    for i, result in enumerate(results[:5]):
                        with st.expander(f"{result['role']} - {result.get('timestamp', '')[:10]}"):
                            st.write(result['content'][:200] + "...")
                else:
                    st.info("No results found")
        
        st.divider()
        
        # Danger Zone
        with st.expander("⚠️ Danger Zone"):
            st.warning("These actions cannot be undone!")
            
            if st.button("🗑️ Clear All History", use_container_width=True, type="primary"):
                if clear_user_history(st.session_state.user_email):
                    st.session_state.messages = []
                    st.success("History cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear history")
            
            if st.button("🚪 Logout", use_container_width=True):
                for key in ['authenticated', 'user_email', 'messages', 'base_model', 'pro_model']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        st.divider()
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px; padding: 10px;'>
            <p>Made with ❤️ by DeckChat Team</p>
            <p>📧 theconsciouschirag@gmail.com</p>
            <p>v2.0.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Chat Area
    col1, col2 = st.columns([1, 20])
    
    with col1:
        gif_url = load_gif_base64()
        if gif_url:
            st.markdown(
                f"""
                <img src="{gif_url}" 
                style="width: 50px; height: 50px; object-fit: contain; 
                       animation: spin 10s linear infinite;">
                <style>
                @keyframes spin {{
                    from {{ transform: rotate(0deg); }}
                    to {{ transform: rotate(360deg); }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("""
        <h1 style='background: linear-gradient(120deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent;
                   margin: 0;'>DeckChat Pro</h1>
        """, unsafe_allow_html=True)
    
    # Status indicators
    model_status = "🟢 Base Model (GPT-3.5)" if st.session_state.current_model == "base" else "🟢 Pro Model (Llama-3-70B)"
    st.caption(f"{model_status} | Persona: {persona.capitalize()}")
    
    # Settings
    show_settings_modal()
    
    # Stats display
    if st.session_state.get('show_stats', False):
        with st.expander("📊 Chat Statistics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Messages", stats['total_messages'])
            with col2:
                st.metric("Today", stats['today_messages'])
            with col3:
                st.metric("Avg/Session", stats['avg_messages_per_session'])
            with col4:
                st.metric("Sessions", stats['total_sessions'])
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(
            msg['role'],
            avatar="🧑" if msg['role'] == 'user' else "✨"
        ):
            st.markdown(msg['content'])
            
            # Show metadata if available
            if msg.get('timestamp'):
                st.caption(f"🕒 {msg['timestamp']} | 🤖 {msg.get('model_used', 'unknown')}")
    
    # Chat input
    prompt = st.chat_input("Type your message here...")
    
    if prompt:
        # Get current model
        current_model = st.session_state.pro_model if st.session_state.current_model == "pro" else st.session_state.base_model
        
        if not current_model:
            st.error("❌ Selected model is not available. Please try refreshing.")
            return
        
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Add to session
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Save to database
        save_message(
            st.session_state.user_email,
            "user",
            prompt,
            "gpt-3.5" if st.session_state.current_model == "base" else "llama-3-70b"
        )
        
        # Prepare messages for model
        system_prompt = SYSTEM_PROMPTS.get(
            st.session_state.get('current_persona', 'default'),
            SYSTEM_PROMPTS['default']
        )
        
        messages_for_model = [SystemMessage(content=system_prompt)]
        
        # Add conversation context (last 15 messages for performance)
        for msg in st.session_state.messages[-15:-1]:
            if msg['role'] == 'user':
                messages_for_model.append(HumanMessage(content=msg['content']))
            else:
                messages_for_model.append(AIMessage(content=msg['content']))
        
        # Add current prompt
        messages_for_model.append(HumanMessage(content=prompt))
        
        # Generate response with streaming
        with st.chat_message("assistant", avatar="✨"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show typing indicator
                with st.spinner(""):
                    # Stream response
                    for chunk in current_model.stream(messages_for_model):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    # Final response
                    message_placeholder.markdown(full_response)
                    
                    # Add to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_used": "gpt-3.5" if st.session_state.current_model == "base" else "llama-3-70b"
                    })
                    
                    # Save to database
                    save_message(
                        st.session_state.user_email,
                        "assistant",
                        full_response,
                        "gpt-3.5" if st.session_state.current_model == "base" else "llama-3-70b"
                    )
                    
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                
                # Add error message to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_used": "error"
                })
        
        # Scroll to bottom (using JavaScript)
        st.markdown(
            """
            <script>
                var element = document.documentElement;
                element.scrollTop = element.scrollHeight;
            </script>
            """,
            unsafe_allow_html=True
        )
    
    # Footer with GIF
    gif_url = load_gif_base64()
    if gif_url:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; gap: 10px; 
                        justify-content: center; margin-top: 30px;'>
                <img src="{gif_url}" style="width: 30px; height: 30px;">
                <p style='color: #666;'>DeckChat Pro - Powered by Advanced AI</p>
                <img src="{gif_url}" style="width: 30px; height: 30px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.caption("✨ DeckChat Pro - Your Intelligent AI Companion")

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
