# app.py - DeckChat Pro (2025 clean style - session-based conversations)
import streamlit as st
import os
import json
import hashlib
from datetime import datetime
import time

# LangChain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="DeckChat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Simple modern CSS
# ────────────────────────────────────────────────
def load_css():
    css = """
    <style>
        .stApp {
            background: #f8f9fc;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
            width: 280px !important;
        }
        .main .block-container {
            padding: 2rem 1rem 10rem 1rem;
            max-width: 48rem;
        }
        .stChatMessage {
            padding: 14px 18px;
            border-radius: 18px;
            margin: 8px 0;
            max-width: 82%;
            line-height: 1.48;
            font-size: 1.03rem;
        }
        [data-testid="chat-message-user"] {
            background: #0066ff;
            color: white;
            margin-left: auto;
            margin-right: 8px;
        }
        [data-testid="chat-message-assistant"] {
            background: #f1f3f5;
            color: #111827;
            margin-right: auto;
            margin-left: 8px;
        }
        .stChatInput > div > div > input {
            border-radius: 24px !important;
            padding: 14px 20px !important;
            border: 1px solid #d1d5db !important;
        }
        .stChatInput > div > div > input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
        }
        .stButton > button {
            border-radius: 12px;
        }
        hr {
            margin: 1.8rem 0;
            border-color: #e5e7eb;
        }
        .chat-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.4rem;
        }
        .chat-preview {
            font-size: 0.9rem;
            color: #6b7280;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Firebase
# ────────────────────────────────────────────────
@st.cache_resource
def get_db():
    if not firebase_admin._apps:
        try:
            if 'FIREBASE_CONFIG' in st.secrets:
                cred = credentials.Certificate(dict(st.secrets['FIREBASE_CONFIG']))
            else:
                cred = credentials.Certificate("firebase-adminsdk.json")
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase init failed: {e}")
            return None
    return firestore.client()

db = get_db()

# ────────────────────────────────────────────────
# Auth helpers
# ────────────────────────────────────────────────
def hash_pw(pw): 
    return hashlib.sha256(pw.encode()).hexdigest()

def sign_in(email, password):
    if not db: return False, "No database"
    users = db.collection("users")
    docs = list(users.where("email", "==", email).where("password_hash", "==", hash_pw(password)).stream())
    if not docs:
        return False, "Wrong email or password"
    user_doc = docs[0]
    user_doc.reference.update({"last_active": firestore.SERVER_TIMESTAMP})
    return True, user_doc.id

def sign_up(email, password):
    if not db: return False, "No database"
    users = db.collection("users")
    if list(users.where("email", "==", email).limit(1).stream()):
        return False, "Email already registered"
    users.add({
        "email": email,
        "password_hash": hash_pw(password),
        "created_at": firestore.SERVER_TIMESTAMP,
        "last_active": firestore.SERVER_TIMESTAMP
    })
    return True, "Account created"

# ────────────────────────────────────────────────
# Conversation helpers
# ────────────────────────────────────────────────
def create_new_conversation(user_id):
    conv_ref = db.collection("conversations").add({
        "user_id": user_id,
        "title": "New Chat",
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "last_preview": ""
    })
    return conv_ref[1].id

def get_user_conversations(user_id, limit=40):
    convs = db.collection("conversations") \
              .where("user_id", "==", user_id) \
              .order_by("updated_at", direction=firestore.Query.DESCENDING) \
              .limit(limit) \
              .stream()
    return [{"id": c.id, **c.to_dict()} for c in convs]

def update_conversation_title_and_preview(conv_id, title, preview):
    db.collection("conversations").document(conv_id).update({
        "title": title,
        "last_preview": preview[:120],
        "updated_at": firestore.SERVER_TIMESTAMP
    })

def save_message(conv_id, user_id, role, content, model_name):
    db.collection("messages").add({
        "conv_id": conv_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "model": model_name,
        "created_at": firestore.SERVER_TIMESTAMP
    })

def load_messages(conv_id, limit=60):
    msgs = db.collection("messages") \
             .where("conv_id", "==", conv_id) \
             .order_by("created_at") \
             .limit(limit) \
             .stream()
    return [{"role": m.to_dict()["role"], "content": m.to_dict()["content"]} for m in msgs]

# ────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────
@st.cache_resource
def get_model(model_key):
    if model_key == "base":
        key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key: return None
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            streaming=True
        )
    else:  # pro
        key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not key: return None
        return ChatGroq(
            model="llama-3.1-70b-versatile",
            api_key=key,
            temperature=0.7,
            streaming=True
        )

# ────────────────────────────────────────────────
# Auth screen
# ────────────────────────────────────────────────
def auth_screen():
    st.title("DeckChat")
    tab1, tab2 = st.tabs(["Sign in", "Create account"])

    with tab1:
        with st.form("login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                ok, msg = sign_in(email, pw)
                if ok:
                    st.session_state.user_id = msg
                    st.session_state.email = email
                    st.success("Signed in")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(msg)

    with tab2:
        with st.form("signup"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Create account", type="primary", use_container_width=True):
                ok, msg = sign_up(email, pw)
                if ok:
                    st.success("Account created → please sign in")
                else:
                    st.error(msg)

# ────────────────────────────────────────────────
# Main chat UI
# ────────────────────────────────────────────────
def chat_ui():
    load_css()

    if "current_conv_id" not in st.session_state:
        # Auto-create first conversation if none exists
        convs = get_user_conversations(st.session_state.user_id, 1)
        if convs:
            st.session_state.current_conv_id = convs[0]["id"]
        else:
            st.session_state.current_conv_id = create_new_conversation(st.session_state.user_id)

    if "messages" not in st.session_state:
        st.session_state.messages = load_messages(st.session_state.current_conv_id)

    # ── Sidebar ───────────────────────────────────────
    with st.sidebar:
        st.title("DeckChat")

        if st.button("＋ New Chat", type="primary", use_container_width=True):
            new_id = create_new_conversation(st.session_state.user_id)
            st.session_state.current_conv_id = new_id
            st.session_state.messages = []
            st.rerun()

        st.divider()

        convs = get_user_conversations(st.session_state.user_id)

        for conv in convs:
            title = conv.get("title", "Chat")
            preview = conv.get("last_preview", "")
            label = f"{title}\n{preview}" if preview else title
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True,
                         type="secondary" if conv["id"] != st.session_state.current_conv_id else "primary"):
                st.session_state.current_conv_id = conv["id"]
                st.session_state.messages = load_messages(conv["id"])
                st.rerun()

        st.divider()

        model_choice = st.radio(
            "Model",
            ["Base (fast)", "Pro (stronger)"],
            horizontal=True
        )
        st.session_state.model_key = "base" if model_choice == "Base (fast)" else "pro"

        if st.button("Sign out"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Main area ─────────────────────────────────────
    st.markdown(f"**{st.session_state.email}**  ·  {model_choice}")

    # Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask anything..."):
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save user message
        save_message(
            st.session_state.current_conv_id,
            st.session_state.user_id,
            "user", prompt,
            st.session_state.model_key
        )

        # Auto title if first message
        if len(st.session_state.messages) == 1:
            title = prompt.strip()[:48].strip(" .,!?") + "..." if len(prompt) > 48 else prompt.strip()
            update_conversation_title_and_preview(st.session_state.current_conv_id, title, prompt)

        # Generate
        model = get_model(st.session_state.model_key)
        if not model:
            st.error("Model not available. Check API keys.")
            return

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""

            messages = [SystemMessage(content="You are a helpful assistant.")]
            for m in st.session_state.messages[-12:]:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))

            try:
                for chunk in model.stream(messages):
                    if chunk.content:
                        full += chunk.content
                        placeholder.markdown(full + "▌")
                placeholder.markdown(full)

                st.session_state.messages.append({"role": "assistant", "content": full})

                # Save & update preview
                save_message(
                    st.session_state.current_conv_id,
                    st.session_state.user_id,
                    "assistant", full,
                    st.session_state.model_key
                )
                update_conversation_title_and_preview(
                    st.session_state.current_conv_id,
                    st.session_state.messages[0]["content"][:48] + "...",
                    full[:120]
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    if "user_id" not in st.session_state:
        auth_screen()
    else:
        chat_ui()
