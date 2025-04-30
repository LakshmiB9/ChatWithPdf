import streamlit as st
from chatbot_core import build_qa_chain   ## Imports function that builds the RAG pipeline

st.set_page_config(page_title=" PDF Chatbot", layout="wide") ## Set up the streamlit page with a title
st.title(" Chat with your PDF")

qa_chain = build_qa_chain("C:\\Users\\15086\\Downloads\\the-storybook-princess-AF-FKB.pdf")  ## Builds QA chain using the specified pdf file

## Initializes the chat history in Streamlit's session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

## Creates a text input filed for the user to ask a question
question = st.text_input("What would you like to know ?", key="input")

#If a question is submitted, the question is sent to the QA chain & stores the result
if question:
    result = qa_chain({
        "question": question,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append((question, result["answer"])) #Saves the question & the answer to the session history

    # Displays the chat history in reverse order (newest on top)
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**❓ Question {len(st.session_state.chat_history) - i}:** {q}")
        st.markdown(f"**🤖 Answer:** {a}")