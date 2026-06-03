import streamlit as st
from embedder import build_index
from retriever import answer_question
import os

import hmac

def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", key="password")
        if st.button("Login"):
            if hmac.compare_digest(st.secrets["APP_PASSWORD"], st.session_state["password"]):
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("Wrong password")
        st.stop()

check_password()

st.set_page_config(
    page_title=" AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.caption("Ask anything about Transformers, GPT-4, and RAG — based on original research papers.")

with st.sidebar:
    st.header("⚙️ Settings")
    
    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Building index..."):
            build_index()
        st.success("Index built!")
    
    st.divider()
    
    st.markdown("**📄 Loaded papers:**")
    st.markdown("- Attention Is All You Need")
    st.markdown("- GPT-4 Technical Report")
    st.markdown("- RAG Original Paper")
    
    st.divider()
    
    if os.path.exists("index.faiss"):
        st.success("✅ Index loaded")
    else:
        st.warning("⚠️ No index found. Click Build Index.")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not os.path.exists("index.faiss"):
    st.warning("👈 Click 'Build / Rebuild Index' in the sidebar to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.code(s)

if prompt := st.chat_input("Ask about transformers, GPT-4, or RAG..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = answer_question(prompt)
        
        st.markdown(result["answer"])
        
        with st.expander("📎 Sources retrieved"):
            for s in result["sources"]:
                st.code(s)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })