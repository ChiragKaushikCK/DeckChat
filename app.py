import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# -----------------------------
# Page Config & Styles
# -----------------------------
st.set_page_config(page_title="DocTalk Groq", page_icon="⚡", layout="centered")
load_dotenv()

# Get your free key from https://console.groq.com/
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

st.markdown("""
<style>
    .stChatMessage {border-radius: 12px; padding: 15px; border: 1px solid rgba(255,255,255,0.1);}
    .animated-text {animation: fadeIn 1.2s ease-in;}
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Local Search Engine (HuggingFace)
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_engine(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    loader = PyMuPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Local & Free Embeddings (No API calls used for this)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = FAISS.from_documents(chunks, embeddings)
    os.remove(temp_path)
    return vector_db, chunks

# -----------------------------
# Main Application UI
# -----------------------------
st.markdown("<h1 class='animated-text'>⚡ DocTalk: Groq Powered</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in .env or Secrets!")

if uploaded_file and GROQ_API_KEY:
    file_bytes = uploaded_file.read()
    
    with st.spinner("Analyzing document locally..."):
        vector_db, all_chunks = build_engine(file_bytes)

    # Initialize Groq (Blazing fast & Free Tier)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.0
    )

    # RAG Setup
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly using the context provided. If the answer is not in the context, say 'I cannot find this in the document.'\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    
    qa_chain = create_retrieval_chain(
        vector_db.as_retriever(search_kwargs={"k": 5}),
        create_stuff_documents_chain(llm, prompt)
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question or type 'summarize'"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Fetching answer from Groq LPU..."):
                if "summarize" in query.lower() or "summary" in query.lower():
                    # Fast Single-Shot Summary
                    text_to_sum = "\n".join([d.page_content for d in all_chunks[:40]]) 
                    answer = llm.invoke(f"Provide a structured summary of this text:\n\n{text_to_sum}").content
                else:
                    response = qa_chain.invoke({"input": query})
                    answer = response["answer"]
                
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF and add your Groq API Key to start.")
