import streamlit as st
from main import ingest, chat

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬")
st.title("YouTube RAG Chatbot")

with st.sidebar:
    st.header("Add a YouTube Video")
    url = st.text_input("YouTube URL")
    if st.button("Ingest", use_container_width=True):
        if url:
            with st.spinner("Downloading & transcribing... this may take a minute."):
                try:
                    n = ingest(url)
                    st.success(f"Done! Stored {n} chunks.")
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Please enter a YouTube URL.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something about the video...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chat(query, st.session_state.history[:-1])
        st.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
