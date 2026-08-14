import json
import requests
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Agentic RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System (Glassmorphism, Dark Slate Theme, Custom Fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

    /* Global reset & typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Outfit', sans-serif;
        color: #f1f5f9;
    }

    /* Header Banner */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.75rem;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Badge & Cards */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 1rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Custom Chat Message Styling */
    div[data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(129, 140, 248, 0.15) !important;
    }

    /* Streamlit Status / Tool Execution Box */
    div[data-testid="stStatusWidget"] {
        border-radius: 12px !important;
        border: 1px solid rgba(129, 140, 248, 0.2) !important;
        background: rgba(30, 41, 59, 0.8) !important;
    }

    /* Inputs & Buttons */
    .stTextInput > div > div > input, .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Backend URL configuration
BACKEND_URL = "http://localhost:8000"

# ==========================================
# SIDEBAR: KNOWLEDGE & INGESTION
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ Studio Controls")
    st.caption("Manage document knowledge & search configuration")
    
    st.divider()

    st.markdown("#### 📄 Ingest PDF Document")
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"], key="pdf_uploader")
    pdf_desc = st.text_input("Document Context / Description", placeholder="e.g. Q3 Financial Report 2026", key="pdf_desc")
    
    if st.button("📥 Index Document", use_container_width=True, type="primary"):
        if pdf_file is not None:
            with st.spinner("Processing & indexing PDF..."):
                try:
                    files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                    headers = {"X-Description": pdf_desc} if pdf_desc else {}
                    res = requests.post(f"{BACKEND_URL}/api/ingest/pdf", files=files, headers=headers)
                    if res.status_code == 200:
                        st.success(f"Successfully indexed `{pdf_file.name}`!")
                    else:
                        st.error(f"Ingestion failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please select a PDF file first.")

    st.divider()

    st.markdown("#### 🌐 Ingest Webpage")
    web_url = st.text_input("Target Web URL", placeholder="https://example.com/article", key="web_url")
    if st.button("🌐 Crawl & Index Webpage", use_container_width=True):
        if web_url:
            with st.spinner("Crawling webpage via Crawl4AI..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/api/ingest/web", params={"webpage": web_url})
                    if res.status_code == 200:
                        st.success(f"Successfully crawled & indexed web page!")
                    else:
                        st.error(f"Indexing failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please enter a valid URL.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# HERO HEADER & STATUS
# ==========================================
st.markdown("""
<div class="status-badge">
    <div class="status-dot"></div> Vector Store & Hybrid Search Ready
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">Agentic RAG Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Intelligent hybrid retrieval with real-time tool reasoning over indexed vectors & live web search.</div>', unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Starter Prompt Pills (Only when chat is empty)
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 What are the latest developments in AI agents?", use_container_width=True):
            st.session_state.pending_prompt = "What are the latest developments in AI agents?"
            st.rerun()
    with col2:
        if st.button("📊 Summarize key points from uploaded docs", use_container_width=True):
            st.session_state.pending_prompt = "Summarize key points from the uploaded documents in the vector store."
            st.rerun()
    with col3:
        if st.button("🌐 Search live web for tech news today", use_container_width=True):
            st.session_state.pending_prompt = "Search live web for top technology news today."
            st.rerun()

# Handle quick starter trigger
initial_prompt = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    initial_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ==========================================
# CHAT INTERFACE
# ==========================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a research question or query vector store...")
prompt = user_input or initial_prompt

if prompt:
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant processing with dynamic tool reasoning timeline
    with st.chat_message("assistant", avatar="⚡"):
        status = st.status("🧠 Agent Reasoning...", expanded=True)
        message_placeholder = st.empty()
        final_answer = ""

        try:
            response = requests.post(
                f"{BACKEND_URL}/api/chat/agentic",
                params={"query": prompt},
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Handle Event Types
                        if data["type"] == "tool_calls":
                            tools = data.get("tools", [])
                            for tool in tools:
                                tool_name = tool.get("name", "Tool Execution")
                                tool_args = tool.get("args", {})
                                
                                icon = "🔍" if "retrieve" in tool_name or "vector" in tool_name else "🌐"
                                status.markdown(f"**{icon} Calling Tool:** `{tool_name}`")
                                with status.expander(f"Tool Arguments: {tool_name}", expanded=False):
                                    st.json(tool_args)

                        elif data["type"] == "tool_result":
                            tool_name = data.get("name", "Tool")
                            status.markdown(f"**✅ Tool Completed:** `{tool_name}`")

                        elif data["type"] == "answer":
                            final_answer = data.get("content", "")
                            message_placeholder.markdown(final_answer)

            if final_answer:
                status.update(label="✨ Reasoning & Synthesis Complete!", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.rerun()
            else:
                status.update(label="⚠️ Finished with no output.", state="complete", expanded=False)

        except requests.exceptions.RequestException as e:
            status.update(label="❌ Backend Connection Failed", state="error", expanded=False)
            st.error(f"Error communicating with backend: {str(e)}")
            st.info(f"Ensure the FastAPI server is running (`uvicorn main:app --reload`) on port 8000.")

