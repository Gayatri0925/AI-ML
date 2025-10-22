import streamlit as st
from utils.chatbot import get_response

st.set_page_config(page_title="AI ChatBot", page_icon="🤖", layout="centered")
st.title("🤖 AI ChatBot (ChatGPT-like)")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.session_state.messages.append({"user": user_input, "bot": response})

for chat in st.session_state.messages:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
