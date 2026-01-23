# app.py - Streamlit Chatbot with Structured Output (Fastest)

import streamlit as st
import os
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.pydantic_v1 import BaseModel, Field

# Define structured output schema
class StructuredResponse(BaseModel):
    """Structured response format for fast parsing"""
    response: str = Field(description="The main response text")
    key_points: Optional[List[str]] = Field(description="List of key points or bullet items", default=None)
    code_examples: Optional[List[str]] = Field(description="Code snippets if applicable", default=None)
    sources: Optional[List[str]] = Field(description="Sources or citations if needed", default=None)

# Page setup
st.set_page_config(page_title="🤖 Fast AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Fast AI Chatbot")
st.caption("Using structured JSON output for maximum speed")

# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = None

# Get API key from secrets
api_key = st.secrets.get("PPLX_API_KEY") if "PPLX_API_KEY" in st.secrets else None

if api_key:
    os.environ["PPLX_API_KEY"] = api_key
    
    # Initialize model with structured output
    if st.session_state.model is None:
        try:
            model = init_chat_model("sonar", model_provider="perplexity")
            
            # Bind structured output
            st.session_state.model = model.with_structured_output(StructuredResponse)
            st.success("✅ Model ready with structured output!")
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)}")

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        if not st.session_state.model:
            st.error("Model not initialized. Check API key.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Get structured response
                    structured_response = st.session_state.model.invoke([
                        SystemMessage(content="You are a helpful assistant. Provide clear, structured responses."),
                        HumanMessage(content=prompt)
                    ])
                    
                    # Display response
                    st.markdown(f"**{structured_response.response}**")
                    
                    if structured_response.key_points:
                        st.markdown("**Key Points:**")
                        for point in structured_response.key_points:
                            st.markdown(f"- {point}")
                    
                    if structured_response.code_examples:
                        st.markdown("**Code Examples:**")
                        for code in structured_response.code_examples:
                            st.code(code)
                    
                    if structured_response.sources:
                        st.markdown("**Sources:**")
                        for source in structured_response.sources:
                            st.markdown(f"- {source}")
                    
                    # Store as text
                    response_text = structured_response.response
                    if structured_response.key_points:
                        response_text += "\n\nKey Points:\n" + "\n".join(f"- {p}" for p in structured_response.key_points)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
