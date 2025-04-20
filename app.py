import os
from pathlib import Path
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from pydantic import BaseModel, Field
from typing import Optional, Type, List, Dict, Any, Union
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Husky Navigator - Northeastern University",
    page_icon="🐺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .northeastern-title {
        color: #CC0000;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.image("https://brand.northeastern.edu/wp-content/uploads/2021/11/Primary_Reverse-1000x1000.png", width=150)
    st.markdown("## Husky Navigator 🐺")
    st.markdown("Your AI assistant for Northeastern University")

    st.markdown("---")

    # User name input
    your_name = st.text_input("What's your name?")

    # Reset conversation button
    if st.button("Reset Conversation"):
        # Initialize a new agent if reset is clicked
        if "husky_agent" in st.session_state:
            st.session_state.husky_agent.reset_memory()
        for key in list(st.session_state.keys()):
            if key in ["messages", "chat_history"]:
                del st.session_state[key]
        st.rerun()

    st.markdown("---")

    # Sample questions
    st.markdown("### Sample Questions")
    sample_questions = [
        "What courses are available in Fall 2025?",
        "Who teaches database courses?",
        "When does registration start?",
        "What are the requirements for CS degree?",
        "What is the last day to drop a class?"
    ]

    for question in sample_questions:
        if st.button(question):
            # Set the question as input and process it
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

            st.session_state.messages.append({"role": "user", "content": question})
            # Trigger a rerun to process the new message
            st.rerun()

# Initialize session state
if "husky_agent" not in st.session_state:
    # Initialize your agent
    st.session_state.husky_agent = HuskyNavigatorLlama3Agent()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Husky Navigator, your AI assistant for Northeastern University. How can I help you today?"}]

# Main title
if your_name:
    st.markdown(f"<div class='northeastern-title'>Hello, {your_name}! 👋</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='northeastern-title'>Husky Navigator 🐺</div>", unsafe_allow_html=True)

st.caption("Your AI assistant for Northeastern University Silicon Valley Campus")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Display tool badge if it exists
        if "tool_used" in message and message["role"] == "assistant":
            st.markdown(f"<div class='tool-badge'>Tool used: {message['tool_used']}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about Northeastern University..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from Husky Navigator agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")

        try:
            # Process the query with your agent
            response = st.session_state.husky_agent.query(prompt)

            # Display the answer
            message_placeholder.markdown(response["answer"])

            # Show which tool was used
            tool_used = response.get("tool_used", "general_chat")
            st.markdown(f"<div class='tool-badge'>Tool used: {tool_used}</div>", unsafe_allow_html=True)

            # If fallback was used, show a note
            if response.get("fallback", False):
                st.info("Note: I had to use a fallback approach to answer your question.")

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "tool_used": tool_used,
                "fallback": response.get("fallback", False)
            })

        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}. Please try again with a different question."
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Information section at the bottom
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Courses")
    st.markdown("Ask about course information, prerequisites, and content.")

with col2:
    st.markdown("### Academic Calendar")
    st.markdown("Find important dates, deadlines, and schedules.")

with col3:
    st.markdown("### Degree Programs")
    st.markdown("Learn about degree requirements and program details.")

# Footer
st.markdown("---")
st.caption("© 2025 Northeastern University. Powered by Husky Navigator AI.")