import streamlit as st
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="DeckChat",
    page_icon="💬",
    layout="wide"
)

# -------------------------
# CSS (ChatGPT Style UI)
# -------------------------

st.markdown("""
<style>

.stApp{
background:#0f172a;
color:white;
}

section[data-testid="stSidebar"]{
background:#020617;
}

[data-testid="chat-message-user"]{
background:#2563eb;
color:white;
border-radius:12px;
padding:12px;
margin-left:25%;
}

[data-testid="chat-message-assistant"]{
background:#1e293b;
color:#e2e8f0;
border-radius:12px;
padding:12px;
margin-right:25%;
}

textarea{
background:#1e293b !important;
color:white !important;
}

button{
border-radius:8px !important;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

# -------------------------
# FIREBASE
# -------------------------

@st.cache_resource
def init_firebase():

    if not firebase_admin._apps:

        cred = credentials.Certificate(dict(st.secrets["FIREBASE_CONFIG"]))
        firebase_admin.initialize_app(cred)

    return firestore.client()

db = init_firebase()

# -------------------------
# OPENROUTER MODEL
# -------------------------

@st.cache_resource
def load_model():

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
        temperature=0.7,
        streaming=True
    )

model = load_model()

# -------------------------
# SESSION STATE
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

# -------------------------
# DATABASE FUNCTIONS
# -------------------------

def create_chat():

    ref = db.collection("chats").add({
        "title":"New Chat",
        "created_at":datetime.utcnow()
    })

    return ref[1].id


def get_chats():

    chats = db.collection("chats").order_by(
        "created_at",
        direction=firestore.Query.DESCENDING
    ).stream()

    results=[]

    for c in chats:

        d=c.to_dict()

        results.append({
            "id":c.id,
            "title":d["title"]
        })

    return results


def save_message(chat_id,role,content):

    db.collection("chats")\
    .document(chat_id)\
    .collection("messages")\
    .add({
        "role":role,
        "content":content,
        "timestamp":datetime.utcnow()
    })


def load_messages(chat_id):

    docs = db.collection("chats")\
    .document(chat_id)\
    .collection("messages")\
    .order_by("timestamp")\
    .stream()

    msgs=[]

    for d in docs:
        msgs.append(d.to_dict())

    return msgs


def update_title(chat_id,prompt):

    db.collection("chats")\
    .document(chat_id)\
    .update({
        "title":prompt[:40]
    })

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.title("💬 DeckChat")

if st.sidebar.button("➕ New Chat"):

    st.session_state.chat_id=create_chat()
    st.session_state.messages=[]
    st.rerun()

st.sidebar.divider()

chats=get_chats()

for c in chats:

    if st.sidebar.button(c["title"],key=c["id"]):

        st.session_state.chat_id=c["id"]
        st.session_state.messages=load_messages(c["id"])
        st.rerun()

# -------------------------
# MAIN CHAT
# -------------------------

st.title("DeckChat")

if st.session_state.chat_id is None:

    st.session_state.chat_id=create_chat()

# -------------------------
# SHOW MESSAGES
# -------------------------

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# -------------------------
# INPUT
# -------------------------

prompt=st.chat_input("Ask anything...")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    save_message(st.session_state.chat_id,"user",prompt)

    if len(st.session_state.messages)==1:
        update_title(st.session_state.chat_id,prompt)

    messages=[SystemMessage(content="You are a helpful AI assistant.")]

    for m in st.session_state.messages:

        if m["role"]=="user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):

        placeholder=st.empty()

        full=""

        for chunk in model.stream(messages):

            if chunk.content:

                full+=chunk.content
                placeholder.markdown(full+"▌")

        placeholder.markdown(full)

    st.session_state.messages.append({
        "role":"assistant",
        "content":full
    })

    save_message(st.session_state.chat_id,"assistant",full)
