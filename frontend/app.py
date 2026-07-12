import os
import streamlit as st
import requests

st.set_page_config(page_title="AI RAG FINANCIAL ASSISTANT")
st.title("AI RAG FINANCIAL ASSISTANT")
st.info(
    "Analyze SEC filings of major U.S. companies using Retrieval-Augmented Generation (RAG). "
    "You can ask questions in **any language**."
)

with st.expander("Supported companies"):

    st.markdown("""
- AAPL – Apple
- MSFT – Microsoft
- NVDA – NVIDIA
- AMD – Advanced Micro Devices
- AMZN – Amazon
- META – Meta Platforms
- GOOGL – Alphabet
- TSLA – Tesla
- AVGO – Broadcom
- NFLX – Netflix
- PLTR – Palantir Technologies
- ORCL – Oracle
- CRM – Salesforce
- IBM – IBM
- JPM – JPMorgan Chase
- V – Visa
- MA – Mastercard
- KO – Coca-Cola
- WMT – Walmart
""")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input(
    "Ask a question about the financial reports in your preferred language..."
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    payload = {
        "query": query,
        "history": st.session_state.messages[:-1]
    }

    with st.spinner("Analyzing financial reports..."):
        try:
            response = requests.post(f"{BACKEND_URL}/ask", json=payload)
            if response.status_code == 200:
                answer = response.json().get("answer")
            else:
                answer = f"Error from backend: {response.text}"
        except Exception as e:
            answer = f"Could not connect to backend: {str(e)}"

    # Wyświetlenie odpowiedzi
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
