import streamlit as st
from husky import husky_agent

# Set page configuration
st.set_page_config(
    page_title="Husky Navigator - Northeastern University Silicon Valley",
    page_icon="🐺",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #CC0000;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .tool-info {
        font-size: 0.8rem;
        color: #888;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for messages and input
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🐺 Husky Navigator initialized! Ask me anything about Northeastern University Silicon Valley."
    })

# Function to reset conversation
def reset_conversation():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "🐺 Husky Navigator initialized! Ask me anything about Northeastern University Silicon Valley."
    }]
    husky_agent.reset_memory()
    st.success("Memory has been cleared. I've forgotten our previous conversation.")

# Header
st.markdown('<h1 class="main-header">🐺 Husky Navigator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Your virtual assistant for Northeastern University Silicon Valley</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.siberianhuskyrescue.org/wp-content/uploads/husky.jpg", width=200)
    st.markdown("## Settings")
    
    # Add the summary mode toggle
    summary_mode = st.checkbox("Enable summary mode (shorter responses)")

    st.markdown("## Commands")
    st.markdown("""
    - Type 'reset', 'clear memory', or 'forget' to clear conversation history
    - Type 'exit', 'quit', or 'bye' to reset and clear the chat
    """)
    
    if st.button("Reset Conversation"):
        reset_conversation()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "tool_info" in message:
            st.markdown(f'<p class="tool-info">{message["tool_info"]}</p>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("You:"):
    # Check for exit command
    if prompt.lower() in ['exit', 'quit', 'bye']:
        reset_conversation()
        # Add exit message
        st.session_state.messages.append({"role": "assistant", "content": "Goodbye! Have a great day!"})
        st.rerun()
    
    # Check for reset command
    if prompt.lower() in ['reset', 'clear memory', 'forget']:
        reset_conversation()
        st.rerun()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display thinking message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Husky Navigator is thinking...")
        
        try:
            # Process the query
            response = husky_agent.query(prompt, summary_mode=summary_mode)
            
            # Prepare tool info text based on fallback status
            tool_info = ""
            if response.get('fallback', False):
                tool_info = "(Used fallback RAG approach)"
            else:
                tool_info = f"(Used tool: {response.get('tool_used', 'unknown')})"
            
            # Update the message with the response
            thinking_placeholder.markdown(response['answer'])
            st.markdown(f'<p class="tool-info">{tool_info}</p>', unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['answer'],
                "tool_info": tool_info
            })
            
        except Exception as e:
            error_message = "I'm sorry, I encountered an error processing your request. Could you try rephrasing your question?"
            thinking_placeholder.markdown(error_message)
            
            # Add error message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message,
                "tool_info": f"(Error: {str(e)})"
            })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>Type 'exit' to quit or 'reset' to clear memory</p>
    <p>© 2025 Northeastern University</p>
</div>
""", unsafe_allow_html=True)