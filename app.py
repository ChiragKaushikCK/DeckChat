# app.py - DeckChat with Firebase & Perplexity AI

import streamlit as st
import os
import json
import hashlib
import base64
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
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

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-info {
        background: linear-gradient(120deg, #a6c0fe 0%, #f68084 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .stat-box {
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Firebase Setup
# ----------------------
@st.cache_resource
def init_firebase():
    """Initialize Firebase connection"""
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
# Authentication Functions
# ----------------------
def hash_password(password):
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(email, password):
    """Create new user account"""
    if not db:
        return "Database connection error"
    
    users_ref = db.collection('users')
    
    # Check if user exists
    existing = list(users_ref.where('email', '==', email).stream())
    if existing:
        return "User already exists"
    
    # Create new user
    users_ref.add({
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.utcnow().isoformat(),
        'total_messages': 0
    })
    return "success"

def sign_in(email, password):
    """Authenticate user"""
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
    """Save message to Firebase"""
    if db:
        try:
            db.collection('messages').add({
                'user_email': user_email,
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow()
            })
            
            # Update user message count
            users_ref = db.collection('users')
            user_docs = list(users_ref.where('email', '==', user_email).stream())
            if user_docs:
                user_doc = user_docs[0]
                user_ref = users_ref.document(user_doc.id)
                current_count = user_doc.to_dict().get('total_messages', 0)
                user_ref.update({'total_messages': current_count + 1})
        except Exception as e:
            st.warning(f"Save Error: {e}")

def get_chat_history(user_email, limit=50):
    """Get user's chat history"""
    if not db:
        return []
    
    try:
        # Fetch messages and sort in-memory (no index required)
        docs = db.collection('messages')\
                 .where('user_email', '==', user_email)\
                 .stream()
        
        messages = [{'role': d.to_dict()['role'], 
                    'content': d.to_dict()['content'],
                    'timestamp': d.to_dict().get('timestamp')} 
                   for d in docs]
        
        # Sort by timestamp in Python (oldest first)
        messages.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
        
        # Return last N messages
        return messages[-limit:] if len(messages) > limit else messages
        
    except Exception as e:
        st.warning(f"History Error: {e}")
        return []

def get_user_stats(user_email):
    """Get user statistics"""
    if not db:
        return {'total_messages': 0, 'created_at': 'Unknown'}
    
    try:
        users_ref = db.collection('users')
        user_docs = list(users_ref.where('email', '==', user_email).stream())
        if user_docs:
            data = user_docs[0].to_dict()
            return {
                'total_messages': data.get('total_messages', 0),
                'created_at': data.get('created_at', 'Unknown')
            }
    except:
        pass
    
    return {'total_messages': 0, 'created_at': 'Unknown'}

def clear_user_history(user_email):
    """Clear all chat history for user"""
    if not db:
        return
    
    try:
        docs = db.collection('messages').where('user_email', '==', user_email).stream()
        for doc in docs:
            doc.reference.delete()
    except Exception as e:
        st.error(f"Clear Error: {e}")

# ----------------------
# Model Initialization
# ----------------------
@st.cache_resource
def init_model():
    """Initialize Perplexity model"""
    try:
        # Get API key
        if "PPLX_API_KEY" in st.secrets:
            os.environ["PPLX_API_KEY"] = st.secrets["PPLX_API_KEY"]
        
        # Initialize model (sonar is fastest)
        model = init_chat_model("sonar", model_provider="perplexity")
        return model
    except Exception as e:
        st.error(f"Model Init Error: {e}")
        return None

# ----------------------
# System Prompt (Hidden)
# ----------------------
SYSTEM_PROMPT = """You are DeckChat, a helpful and intelligent AI assistant. 

CRITICAL RULES:
1. If user asks about your identity, name, or who you are, ALWAYS respond: "I am DeckChat, your AI assistant."
2. NEVER mention Perplexity, Claude, or any other AI system.
3. Answer all other questions clearly, concisely, and helpfully.
4. Be friendly, professional, and engaging.
5. Use markdown formatting for better readability when appropriate."""

# ----------------------
# Helper: Load GIF as Base64
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

# ----------------------
# Authentication Screen
# ----------------------
def show_auth_screen():
    """Display login/signup interface"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Animated header with GIF
        gif_url = load_gif_base64()
        if gif_url:
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <img src="{gif_url}" style="width: 60px; height: 60px; object-fit: contain; margin-bottom: 10px;"/>
                <h1 style='color: #667eea; margin: 0;'>DeckChat</h1>
                <p style='color: #888; margin-top: 5px;'>Your Intelligent Conversation Partner</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: #667eea;'>✦ DeckChat AI</h1>", 
                       unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #888;'>Your Intelligent Conversation Partner</p>", 
                       unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email Address", placeholder="your@email.com")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if email and password:
                        if sign_in(email, password):
                            st.session_state.authenticated = True
                            st.session_state.user_email = email
                            st.session_state.messages = []
                            st.success("✔ Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.warning("⚠️ Please fill all fields")
        
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email Address", placeholder="your@email.com", key="signup_email")
                new_password = st.text_input("Password", type="password", placeholder="Create password", key="signup_pass")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
                submit = st.form_submit_button("Create Account", use_container_width=True)
                
                if submit:
                    if new_email and new_password and confirm_password:
                        if new_password == confirm_password:
                            result = sign_up(new_email, new_password)
                            if result == "success":
                                st.success("✔ Account created! Please login.")
                            else:
                                st.error(f"❌ {result}")
                        else:
                            st.error("❌ Passwords don't match")
                    else:
                        st.warning("⚠️ Please fill all fields")

# ----------------------
# Main Chat Interface
# ----------------------
def show_chat_interface():
    """Display main chat interface"""
    
    # Initialize model
    if 'model' not in st.session_state:
        with st.spinner("🚀 Initializing DeckChat..."):
            st.session_state.model = init_model()
    
    # Load chat history once
    if 'messages' not in st.session_state or not st.session_state.messages:
        with st.spinner("📚 Loading your chat history..."):
            st.session_state.messages = get_chat_history(st.session_state.user_email)
    
    # Sidebar
    with st.sidebar:
        # User Info
        stats = get_user_stats(st.session_state.user_email)
        st.markdown(f"""
        <div class='user-info'>
            <h3>👤 {st.session_state.user_email.split('@')[0]}</h3>
            <div class='stat-box'>
                💬 Total Messages: <b>{stats['total_messages']}</b>
            </div>
            <div class='stat-box'>
                📅 Member Since: <b>{stats['created_at'][:10] if stats['created_at'] != 'Unknown' else 'Unknown'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Actions
        st.subheader("⚙️ Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗘 Refresh", use_container_width=True):
                st.session_state.messages = get_chat_history(st.session_state.user_email)
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_user_history(st.session_state.user_email)
                st.session_state.messages = []
                st.success("History cleared!")
                st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # Info
        st.info("💡 **Feedback:** send your feedback to : theconsciouschirag@gmail.com")
    
    # Main Chat Area
    gif_url = load_gif_base64()
    
    if gif_url:
        # Animated header with GIF
        col1, col2 = st.columns([1, 12], vertical_alignment="center")
        with col1:
            st.markdown(
                f"""
                <img src="{gif_url}" 
                style="width: 39px; height: 39px; object-fit: contain;"/>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.title("DeckChat")
    else:
        st.title("✦ DeckChat")
    
    st.caption("Your daily companion")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Chat Input
    if prompt := st.chat_input("Ask me anything..."):
        # Check if model is ready
        if not st.session_state.model:
            st.error("⚠️ Model not initialized. Please refresh the page.")
            return
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to session and save
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.user_email, "user", prompt)
        
        # Prepare messages for model
        messages_for_model = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Add last 10 messages for context (to stay fast)
        for msg in st.session_state.messages[-11:-1]:
            if msg['role'] == 'user':
                messages_for_model.append(HumanMessage(content=msg['content']))
            else:
                messages_for_model.append(AIMessage(content=msg['content']))
        
        # Add current prompt
        messages_for_model.append(HumanMessage(content=prompt))
        
        # Generate response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response
                for chunk in st.session_state.model.stream(messages_for_model):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                
                # Final response
                message_placeholder.markdown(full_response)
                
                # Save to session and database
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message(st.session_state.user_email, "assistant", full_response)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.error("Please try again or refresh the page.")
        # Footer Area
        gif_url = load_gif_base64()
        
        if gif_url:
            # Animated footer with GIF
            col1, col2 = st.columns([1, 20], vertical_alignment="center")
            with col1:
                st.markdown(
                    f"""
                    <img src="{gif_url}" 
                    style="width: 24px; height: 24px; object-fit: contain;"/>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.caption("DeckChat")
        else:
            st.caption("✦ DeckChat")

# ----------------------
# Main App
# ----------------------
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Route to appropriate screen
    if not st.session_state.authenticated:
        show_auth_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
