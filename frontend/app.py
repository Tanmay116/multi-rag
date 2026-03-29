import json

import requests
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Premium sleek styling */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        color: #2e3b4e;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1em;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    /* Subtle background for chat bubbles */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #f8f9fa;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("✨ Agentic RAG ")
st.markdown('<p class="subtitle">Ask your research questions and watch the AI tools stream in real-time.</p>', unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# CHAT INTERFACE
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about machine learning, AI, or latest news..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant processing
    with st.chat_message("assistant"):
        status = st.status("🧠 Reasoning...", expanded=True)
        message_placeholder = st.empty()
        
        final_answer = ""
        
        try:
            # Connect to FastAPI SSE stream
            response = requests.post(
                "http://localhost:8000/api/chat/agentic",
                params={"query": prompt},
                stream=True
            )
            response.raise_for_status()

            # Process the stream
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Update UI based on event type
                        if data["type"] == "tool_calls":
                            tools = data.get("tools", [])
                            for tool in tools:
                                name = tool.get("name", "Unknown")
                                args = tool.get("args", {})
                                status.markdown(f"**🛠️ Calling Tool:** `{name}`")
                                with status.expander(f"Arguments for {name}"):
                                    st.json(args)
                                    
                        elif data["type"] == "tool_result":
                            name = data.get("name", "Unknown")
                            status.markdown(f"**✅ Completed Tool:** `{name}`")
                            
                        elif data["type"] == "answer":
                            final_answer = data.get("content", "")
                            message_placeholder.markdown(final_answer)
            
            # Finalize status box state
            if final_answer:
                status.update(label="Task Complete!", state="complete", expanded=False)
            else:
                status.update(label="Finished (No content)", state="complete", expanded=False)
            
            # Save final response to history
            if final_answer:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
        except requests.exceptions.RequestException as e:
            status.update(label="Connection Error", state="error", expanded=False)
            st.error(f"Failed to connect to backend: {str(e)}")
            st.info("Make sure the backend is running `uvicorn main:app --reload` on port 8000")
