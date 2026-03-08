# app.py - DeckChat Advanced AI with OpenRouter (Base & Pro Tiers)

import streamlit as st
import os
import json
import hashlib
import base64
from datetime import datetime
from openai import OpenAI
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
# Load CSS
# ----------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

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
        'total_messages': 0,
        'plan': 'base'
    })
    return "success"

def sign_in(email, password):
    if not db:
        return False
    users_ref = db.collection('users')
    docs = list(users_ref.where('email', '==', email)
                .where('password_hash', '==', hash_password(password)).stream())
    return True if docs else False

def get_user_plan(user_email):
    if not db:
        return "base"
    try:
        users_ref = db.collection('users')
        docs = list(users_ref.where('email', '==', user_email).stream())
        if docs:
            return docs[0].to_dict().get('plan', 'base')
    except:
        pass
    return "base"

# ----------------------
# Database Functions
# ----------------------
def save_message(user_email, role, content, session_id=None):
    if db:
        try:
            db.collection('messages').add({
                'user_email': user_email,
                'role': role,
                'content': content,
                'session_id': session_id or st.session_state.get('session_id', 'default'),
                'timestamp': datetime.utcnow()
            })
            users_ref = db.collection('users')
            user_docs = list(users_ref.where('email', '==', user_email).stream())
            if user_docs:
                user_doc = user_docs[0]
                user_ref = users_ref.document(user_doc.id)
                current_count = user_doc.to_dict().get('total_messages', 0)
                user_ref.update({'total_messages': current_count + 1})
        except Exception as e:
            st.warning(f"Save Error: {e}")

def save_conversation_title(user_email, session_id, title):
    if db:
        try:
            db.collection('conversations').document(f"{user_email}_{session_id}").set({
                'user_email': user_email,
                'session_id': session_id,
                'title': title,
                'updated_at': datetime.utcnow()
            }, merge=True)
        except:
            pass

def get_conversations(user_email, limit=20):
    if not db:
        return []
    try:
        docs = db.collection('conversations') \
                 .where('user_email', '==', user_email) \
                 .stream()
        convs = [d.to_dict() for d in docs]
        convs.sort(key=lambda x: x.get('updated_at', datetime.min), reverse=True)
        return convs[:limit]
    except:
        return []

def get_chat_history(user_email, session_id=None, limit=60):
    if not db:
        return []
    try:
        query = db.collection('messages').where('user_email', '==', user_email)
        if session_id:
            query = query.where('session_id', '==', session_id)
        docs = query.stream()
        messages = [{'role': d.to_dict()['role'],
                     'content': d.to_dict()['content'],
                     'timestamp': d.to_dict().get('timestamp')}
                    for d in docs]
        messages.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
        return messages[-limit:] if len(messages) > limit else messages
    except Exception as e:
        st.warning(f"History Error: {e}")
        return []

def get_user_stats(user_email):
    if not db:
        return {'total_messages': 0, 'created_at': 'Unknown', 'plan': 'base'}
    try:
        users_ref = db.collection('users')
        user_docs = list(users_ref.where('email', '==', user_email).stream())
        if user_docs:
            data = user_docs[0].to_dict()
            return {
                'total_messages': data.get('total_messages', 0),
                'created_at': data.get('created_at', 'Unknown'),
                'plan': data.get('plan', 'base')
            }
    except:
        pass
    return {'total_messages': 0, 'created_at': 'Unknown', 'plan': 'base'}

def clear_user_history(user_email, session_id=None):
    if not db:
        return
    try:
        query = db.collection('messages').where('user_email', '==', user_email)
        if session_id:
            query = query.where('session_id', '==', session_id)
        for doc in query.stream():
            doc.reference.delete()
    except Exception as e:
        st.error(f"Clear Error: {e}")

# ----------------------
# OpenRouter Model Config
# ----------------------
MODEL_CONFIG = {
    "base": {
        "model": "openai/gpt-3.5-turbo",
        "label": "Base · GPT-3.5 Turbo",
        "badge": "BASE",
        "badge_color": "#4ade80",
        "description": "Fast & efficient for everyday tasks",
        "icon": "⚡"
    },
    "pro": {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "label": "Pro · Llama 3.3 70B",
        "badge": "PRO",
        "badge_color": "#a78bfa",
        "description": "Advanced reasoning & complex tasks",
        "icon": "🚀"
    }
}

# Available models for pro users to choose from
PRO_MODELS = {
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B (via Groq-fast)",
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "openai/gpt-4o": "GPT-4o",
    "deepseek/deepseek-r1": "DeepSeek R1",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B",
}

@st.cache_resource
def get_openrouter_client():
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
        if not api_key:
            st.error("⚠️ OPENROUTER_API_KEY not found in secrets")
            return None
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    except Exception as e:
        st.error(f"OpenRouter init error: {e}")
        return None

def stream_response(client, model, messages, system_prompt):
    """Stream response from OpenRouter"""
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=all_messages,
        stream=True,
        max_tokens=4096,
        temperature=0.7,
        extra_headers={
            "HTTP-Referer": "https://deckchat.app",
            "X-Title": "DeckChat"
        }
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ----------------------
# System Prompt
# ----------------------
SYSTEM_PROMPT = """You are DeckChat, a highly intelligent and helpful AI assistant.

IDENTITY RULES:
1. If asked about your identity/name/who you are → ALWAYS say: "I am DeckChat, your AI assistant."
2. NEVER mention Claude, GPT, Gemini, Llama, or any underlying model.
3. You are DeckChat — that is your only identity.

RESPONSE QUALITY:
- Use markdown formatting: headers, bold, italic, tables, code blocks, lists
- For code: always use proper syntax-highlighted code blocks with language tags
- For math: use clear notation
- For lists: use proper bullet/numbered lists
- Keep responses well-structured and scannable
- Be concise yet thorough — quality over quantity
- Match the user's tone: casual for casual, technical for technical

PERSONALITY:
- Warm, helpful, professional
- Direct and honest
- Proactively suggest follow-up angles
- Acknowledge uncertainty honestly"""

# ----------------------
# GIF Helper
# ----------------------
@st.cache_data
def load_gif_base64(gif_path="neon_star_animated.gif"):
    try:
        with open(gif_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/gif;base64,{data}"
    except:
        return None

def new_session_id():
    return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

# ----------------------
# Auth Screen
# ----------------------
def show_auth_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        gif_url = load_gif_base64()
        if gif_url:
            st.markdown(f"""
            <div class="auth-header">
                <img src="{gif_url}" class="auth-logo"/>
                <h1 class="auth-title">DeckChat</h1>
                <p class="auth-subtitle">Your Intelligent Conversation Partner</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="auth-header">
                <div class="auth-logo-fallback">✦</div>
                <h1 class="auth-title">DeckChat</h1>
                <p class="auth-subtitle">Your Intelligent Conversation Partner</p>
            </div>
            """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email Address", placeholder="you@email.com")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Sign In →", use_container_width=True)
                if submit:
                    if email and password:
                        with st.spinner("Authenticating..."):
                            if sign_in(email, password):
                                st.session_state.authenticated = True
                                st.session_state.user_email = email
                                st.session_state.messages = []
                                st.session_state.session_id = new_session_id()
                                st.session_state.plan = get_user_plan(email)
                                st.session_state.selected_model = MODEL_CONFIG[st.session_state.plan]["model"]
                                st.success("✔ Welcome back!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials. Please try again.")
                    else:
                        st.warning("⚠️ Please fill all fields")

        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email Address", placeholder="you@email.com", key="su_email")
                new_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="su_pass")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                submit = st.form_submit_button("Create Account →", use_container_width=True)
                if submit:
                    if new_email and new_password and confirm_password:
                        if len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters")
                        elif new_password == confirm_password:
                            result = sign_up(new_email, new_password)
                            if result == "success":
                                st.success("✔ Account created! Sign in to continue.")
                            else:
                                st.error(f"❌ {result}")
                        else:
                            st.error("❌ Passwords don't match")
                    else:
                        st.warning("⚠️ Please fill all fields")

# ----------------------
# Chat Interface
# ----------------------
def show_chat_interface():
    gif_url = load_gif_base64()
    client = get_openrouter_client()

    # Session defaults
    if 'session_id' not in st.session_state:
        st.session_state.session_id = new_session_id()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'plan' not in st.session_state:
        st.session_state.plan = get_user_plan(st.session_state.user_email)
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = MODEL_CONFIG[st.session_state.plan]["model"]
    if 'history_loaded' not in st.session_state:
        with st.spinner("📚 Loading history..."):
            st.session_state.messages = get_chat_history(
                st.session_state.user_email,
                st.session_state.session_id
            )
        st.session_state.history_loaded = True

    # ---- SIDEBAR ----
    with st.sidebar:
        # Logo in sidebar
        if gif_url:
            st.markdown(f"""
            <div class="sidebar-logo">
                <img src="{gif_url}" class="sidebar-gif"/>
                <span class="sidebar-title">DeckChat</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='sidebar-logo'><span class='sidebar-title'>✦ DeckChat</span></div>",
                        unsafe_allow_html=True)

        # New Chat button
        if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
            st.session_state.session_id = new_session_id()
            st.session_state.messages = []
            st.session_state.history_loaded = False
            st.rerun()

        st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

        # Conversation History
        st.markdown("<div class='sidebar-section-label'>Recent Conversations</div>", unsafe_allow_html=True)
        conversations = get_conversations(st.session_state.user_email)
        if conversations:
            for conv in conversations[:10]:
                title = conv.get('title', 'Untitled')[:32]
                sid = conv.get('session_id')
                is_active = sid == st.session_state.session_id
                btn_class = "conv-btn-active" if is_active else "conv-btn"
                if st.button(f"💬 {title}", key=f"conv_{sid}", use_container_width=True):
                    st.session_state.session_id = sid
                    st.session_state.messages = get_chat_history(
                        st.session_state.user_email, sid
                    )
                    st.session_state.history_loaded = True
                    st.rerun()
        else:
            st.markdown("<div class='no-convs'>No conversations yet.<br/>Start chatting!</div>",
                        unsafe_allow_html=True)

        st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

        # Model Selection
        plan = st.session_state.plan
        cfg = MODEL_CONFIG[plan]
        st.markdown(f"""
        <div class='model-badge-box'>
            <span class='plan-badge' style='background:{cfg["badge_color"]};'>{cfg["badge"]}</span>
            <span class='model-name'>{cfg["icon"]} {cfg["label"]}</span>
        </div>
        """, unsafe_allow_html=True)

        if plan == "pro":
            selected = st.selectbox(
                "Choose Model",
                options=list(PRO_MODELS.keys()),
                format_func=lambda x: PRO_MODELS[x],
                index=list(PRO_MODELS.keys()).index(
                    st.session_state.selected_model
                    if st.session_state.selected_model in PRO_MODELS else list(PRO_MODELS.keys())[0]
                ),
                key="model_selector"
            )
            st.session_state.selected_model = selected
        else:
            st.session_state.selected_model = MODEL_CONFIG["base"]["model"]
            st.markdown("""
            <div class='upgrade-hint'>
                🌟 <b>Upgrade to Pro</b> for access to GPT-4o, Claude, Gemini & more!
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

        # User Stats
        stats = get_user_stats(st.session_state.user_email)
        username = st.session_state.user_email.split('@')[0]
        member_since = stats['created_at'][:10] if stats['created_at'] != 'Unknown' else 'Unknown'

        st.markdown(f"""
        <div class='user-card'>
            <div class='user-avatar'>{username[0].upper()}</div>
            <div class='user-details'>
                <div class='user-name'>{username}</div>
                <div class='user-meta'>💬 {stats['total_messages']} messages</div>
                <div class='user-meta'>📅 Since {member_since}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='sidebar-divider'/>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_user_history(st.session_state.user_email, st.session_state.session_id)
                st.session_state.messages = []
                st.success("Cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.messages = get_chat_history(
                    st.session_state.user_email, st.session_state.session_id
                )
                st.rerun()

        if st.button("🚪 Sign Out", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()

        st.markdown("""
        <div class='sidebar-footer'>
            <a href='mailto:theconsciouschirag@gmail.com'>📧 Send Feedback</a>
        </div>
        """, unsafe_allow_html=True)

    # ---- MAIN CHAT AREA ----
    # Top header bar
    current_model_label = PRO_MODELS.get(st.session_state.selected_model, st.session_state.selected_model)
    if gif_url:
        st.markdown(f"""
        <div class='chat-header'>
            <div class='chat-header-left'>
                <img src="{gif_url}" class='header-gif'/>
                <span class='header-title'>DeckChat</span>
            </div>
            <div class='chat-header-right'>
                <span class='header-model-badge'>
                    {MODEL_CONFIG[plan]["icon"]} {current_model_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-header'>
            <div class='chat-header-left'>
                <span class='header-title'>✦ DeckChat</span>
            </div>
            <div class='chat-header-right'>
                <span class='header-model-badge'>{current_model_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Empty state
    if not st.session_state.messages:
        username = st.session_state.user_email.split('@')[0]
        st.markdown(f"""
        <div class='empty-state'>
            <div class='empty-state-icon'>✦</div>
            <h2 class='empty-state-title'>Hello, {username}! 👋</h2>
            <p class='empty-state-subtitle'>I'm DeckChat, your AI assistant. How can I help you today?</p>
            <div class='suggestion-chips'>
                <div class='chip'>✍️ Help me write something</div>
                <div class='chip'>💡 Explain a concept</div>
                <div class='chip'>🐍 Write some code</div>
                <div class='chip'>📊 Analyze data</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = msg['role']
            content = msg['content']
            if role == "user":
                st.markdown(f"""
                <div class='message-row user-row'>
                    <div class='message-bubble user-bubble'>{content}</div>
                    <div class='msg-avatar user-avatar-icon'>👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar="✦"):
                    st.markdown(content)

    # Chat Input
    prompt = st.chat_input("Message DeckChat...", key="chat_input")

    if prompt:
        if not client:
            st.error("⚠️ OpenRouter client not initialized. Check your API key in secrets.")
            return

        # Show user message
        st.markdown(f"""
        <div class='message-row user-row'>
            <div class='message-bubble user-bubble'>{prompt}</div>
            <div class='msg-avatar user-avatar-icon'>👤</div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.user_email, "user", prompt, st.session_state.session_id)

        # Auto-title conversation from first message
        if len(st.session_state.messages) == 1:
            title = prompt[:50] + ("..." if len(prompt) > 50 else "")
            save_conversation_title(st.session_state.user_email, st.session_state.session_id, title)

        # Build message history for model (last 15 turns)
        history_for_model = []
        for m in st.session_state.messages[-16:-1]:
            history_for_model.append({"role": m["role"], "content": m["content"]})
        history_for_model.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant", avatar="✦"):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in stream_response(
                    client,
                    st.session_state.selected_model,
                    history_for_model,
                    SYSTEM_PROMPT
                ):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message(st.session_state.user_email, "assistant", full_response, st.session_state.session_id)

            except Exception as e:
                err = f"❌ Error: {str(e)}\n\nPlease try again or check your API configuration."
                placeholder.error(err)

    # Footer
    if gif_url:
        st.markdown(f"""
        <div class='chat-footer'>
            <img src="{gif_url}" class='footer-gif'/>
            <span class='footer-text'>DeckChat · Powered by OpenRouter · Your conversations are private</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='chat-footer'>
            <span class='footer-text'>✦ DeckChat · Powered by OpenRouter · Your conversations are private</span>
        </div>
        """, unsafe_allow_html=True)


# ----------------------
# Main
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
