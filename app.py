import streamlit as st
from rag_engine import RAGEngine

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ RAG Assistant")

st.write(
    "Ask general questions, upload documents, or use both together."
)

# --------------------------------------------------
# API KEY
# --------------------------------------------------

groq_key = st.secrets["GROQ_API_KEY"]

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine(groq_key)

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Documents")

    mode = st.selectbox(
        "Select Mode",
        [
            "Auto",
            "Document Only",
            "General Chat"
        ]
    )

    uploaded_files = st.file_uploader(
        "Choose Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Process Documents"):

            with st.spinner("Processing documents..."):

                all_chunks = []

                for file in uploaded_files:

                    file_type = (
                        file.name
                        .split(".")[-1]
                        .lower()
                    )

                    text = (
                        st.session_state.engine.extract_text(
                            file,
                            file_type
                        )
                    )

                    chunks = (
                        st.session_state.engine.chunk_text(
                            text
                        )
                    )

                    all_chunks.extend(chunks)

                st.session_state.engine.build_index(
                    all_chunks
                )

                st.session_state.document_ready = True

            st.success(
                f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} document(s)"
            )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.markdown("### Current Mode")

    if mode == "Auto":
        st.info(
            "Uses uploaded documents if relevant, otherwise falls back to normal AI chat."
        )

    elif mode == "Document Only":
        st.info(
            "Answers only from uploaded documents."
        )

    else:
        st.info(
            "Ignores uploaded documents and behaves like ChatGPT."
        )

    st.markdown("---")

    st.markdown("### Example Questions")

    st.markdown("""
    - Explain layered agent architecture
    - What is RAG?
    - What is ransomware?
    - Summarize the uploaded document
    - What persistence mechanism is used?
    - Explain malware analysis
    """)

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(
            msg["content"]
        )

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

query = st.chat_input(
    "Ask anything..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = (
                st.session_state.engine.ask(
                    query,
                    mode
                )
            )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

