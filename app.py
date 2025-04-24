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
    .tool-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 8px;
        color: white;
    }
    .course-search-badge {
        background-color: #4285F4;
    }
    .faculty-search-badge {
        background-color: #34A853;
    }
    .academic-calendar-badge {
        background-color: #FBBC05;
        color: #333;
    }
    .degree-requirements-badge {
        background-color: #EA4335;
    }
    .course-schedule-badge {
        background-color: #8755C9;
    }
    .northeastern-knowledge-base-badge {
        background-color: #CC0000;
    }
    .general-chat-badge {
        background-color: #6C757D;
    }
    .fallback-badge {
        background-color: #FF6B6B;
    }
    .tools-list {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .tools-list-header {
        font-weight: bold;
        margin-bottom: 5px;
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
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("husky_advisor.png", width=200)
    
    st.markdown("## Settings")
    # Add the summary mode toggle
    summary_mode = st.checkbox("Enable summary mode (shorter responses)")
    
    # Add memory toggle
    use_memory = st.checkbox("Enable conversation memory", value=False)
    if use_memory:
        st.info("Memory is enabled - Husky will remember previous conversation context.")
    else:
        st.info("Memory is disabled - Each question will be treated independently.")
    
    # Available Tools Section
    st.markdown("## Available Tools")
    st.markdown("""
    <div class="tools-list">
        <div class="tools-list-header">Husky Navigator can access these tools:</div>
        <div class="tool-badge course-search-badge">🔍 Course Search</div>
        <div class="tool-badge faculty-search-badge">👨‍🏫 Faculty Search</div>
        <div class="tool-badge academic-calendar-badge">📅 Academic Calendar</div>
        <div class="tool-badge degree-requirements-badge">🎓 Degree Requirements</div>
        <div class="tool-badge course-schedule-badge">⏰ Course Schedule</div>
        <div class="tool-badge northeastern-knowledge-base-badge">📚 Knowledge Base</div>
        <div class="tool-badge general-chat-badge">💬 General Chat</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        if message["role"] == "assistant":
            # Display tool badge if available
            if "tool_used" in message or "fallback" in message:
                # Determine the badge style based on the tool used
                tool_badge = ""
                
                if message.get("fallback", False):
                    tool_badge = '<div class="tool-badge fallback-badge">🔄 Fallback RAG</div>'
                elif "tool_used" in message:
                    tool_used = message["tool_used"]
                    if tool_used == "course_search":
                        tool_badge = '<div class="tool-badge course-search-badge">🔍 Course Search</div>'
                    elif tool_used == "faculty_search":
                        tool_badge = '<div class="tool-badge faculty-search-badge">👨‍🏫 Faculty Search</div>'
                    elif tool_used == "academic_calendar":
                        tool_badge = '<div class="tool-badge academic-calendar-badge">📅 Academic Calendar</div>'
                    elif tool_used == "degree_requirements":
                        tool_badge = '<div class="tool-badge degree-requirements-badge">🎓 Degree Requirements</div>'
                    elif tool_used == "course_schedule":
                        tool_badge = '<div class="tool-badge course-schedule-badge">⏰ Course Schedule</div>'
                    elif tool_used == "northeastern_knowledge_base":
                        tool_badge = '<div class="tool-badge northeastern-knowledge-base-badge">📚 Knowledge Base</div>'
                    elif tool_used == "general_chat":
                        tool_badge = '<div class="tool-badge general-chat-badge">💬 General Chat</div>'
                
                # Display the badge
                st.markdown(tool_badge, unsafe_allow_html=True)
                
            # Display tool info text if available
            if "tool_info" in message:
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
            # Process the query with summary mode and memory options
            response = husky_agent.query(prompt, summary_mode=summary_mode, use_memory=use_memory)
            
            # Prepare tool info text based on fallback status
            tool_info = ""
            if response.get('fallback', False):
                tool_info = "(Used fallback RAG approach)"
            else:
                tool_info = f"(Used tool: {response.get('tool_used', 'unknown')})"
            
            # Update the message with the response
            thinking_placeholder.markdown(response['answer'])
            
            # Add tool badge
            tool_badge = ""
            if response.get('fallback', False):
                tool_badge = '<div class="tool-badge fallback-badge">🔄 Fallback RAG</div>'
            elif "tool_used" in response:
                tool_used = response["tool_used"]
                if tool_used == "course_search":
                    tool_badge = '<div class="tool-badge course-search-badge">🔍 Course Search</div>'
                elif tool_used == "faculty_search":
                    tool_badge = '<div class="tool-badge faculty-search-badge">👨‍🏫 Faculty Search</div>'
                elif tool_used == "academic_calendar":
                    tool_badge = '<div class="tool-badge academic-calendar-badge">📅 Academic Calendar</div>'
                elif tool_used == "degree_requirements":
                    tool_badge = '<div class="tool-badge degree-requirements-badge">🎓 Degree Requirements</div>'
                elif tool_used == "course_schedule":
                    tool_badge = '<div class="tool-badge course-schedule-badge">⏰ Course Schedule</div>'
                elif tool_used == "northeastern_knowledge_base":
                    tool_badge = '<div class="tool-badge northeastern-knowledge-base-badge">📚 Knowledge Base</div>'
                elif tool_used == "general_chat":
                    tool_badge = '<div class="tool-badge general-chat-badge">💬 General Chat</div>'
            
            st.markdown(tool_badge, unsafe_allow_html=True)
            st.markdown(f'<p class="tool-info">{tool_info}</p>', unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['answer'],
                "tool_info": tool_info,
                "tool_used": response.get('tool_used', 'unknown'),
                "fallback": response.get('fallback', False)
            })
            
        except Exception as e:
            error_message = "I'm sorry, I encountered an error processing your request. Could you try rephrasing your question?"
            thinking_placeholder.markdown(error_message)
            
            # Add error message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message,
                "tool_info": f"(Error: {str(e)})",
                "fallback": True
            })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>Type 'exit' to quit or 'reset' to clear memory</p>
    <p>© 2025 Northeastern University</p>
</div>
""", unsafe_allow_html=True)