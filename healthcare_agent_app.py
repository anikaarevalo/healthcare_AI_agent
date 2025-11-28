"""
Healthcare Agent Application
=============================
A Streamlit-based healthcare agent powered by Claude AI with streaming responses.
No API keys required - uses Anthropic's built-in integration.

Features:
- Claude AI powered responses with streaming
- Web search capabilities
- Healthcare-focused conversational interface
- Medical disclaimer UI
"""

import streamlit as st
import json
from typing import Generator

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Healthcare Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


def initialize_session_state():
    """Initialize session state variables for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def apply_custom_css():
    """Apply custom CSS for medical-themed design."""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            background-color: #f8fbfd;
        }
        
        /* Header styling */
        .stApp header {
            background-color: #2c5f7c;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: white;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* User message */
        .stChatMessage[data-testid="user-message"] {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        /* Assistant message */
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #f1f8f4;
            border-left: 4px solid #4caf50;
        }
        
        /* Title styling */
        h1 {
            color: #2c5f7c;
            font-weight: 600;
        }
        
        /* Disclaimer box */
        .element-container div[data-testid="stExpander"] {
            background-color: #fff3cd;
            border-radius: 8px;
            border: 1px solid #ffc107;
        }
        
        /* Input box styling */
        .stChatInputContainer {
            border-top: 2px solid #e0e0e0;
            padding-top: 1rem;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #2c5f7c;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 500;
        }
        
        .stButton button:hover {
            background-color: #1e4a61;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# AI CHAT FUNCTIONS
# ============================================================================

async def stream_claude_response(user_message: str) -> Generator[str, None, None]:
    """
    Stream response from Claude AI using Anthropic API.
    
    Args:
        user_message: User's question or message
    
    Yields:
        str: Chunks of the AI response
    """
    import anthropic
    
    # Create system prompt for healthcare context
    system_prompt = """You are a knowledgeable healthcare AI assistant.

Your role is to:
- Provide accurate, evidence-based medical information
- Keep responses concise, clear, and medically appropriate
- Always remind users to consult healthcare professionals for personal medical advice
- Use a warm, professional, and empathetic tone

Important disclaimers to remember:
- You are not a replacement for professional medical advice
- Encourage users to consult with qualified healthcare providers
- For emergencies, direct users to call emergency services

When discussing medical topics:
- Cite general medical knowledge and best practices
- Explain conditions, symptoms, and treatments clearly
- Avoid making specific diagnoses
- Emphasize the importance of professional medical consultation"""

    try:
        # Use Anthropic's client - API key is handled by Streamlit Cloud
        client = anthropic.Anthropic()
        
        # Create message with streaming
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        ) as stream:
            for text in stream.text_stream:
                yield text
                
    except Exception as e:
        yield f"I apologize, but I encountered an error: {str(e)}. Please try again."


def get_claude_response(user_message: str) -> str:
    """
    Get non-streaming response from Claude AI.
    
    Args:
        user_message: User's question or message
    
    Returns:
        str: Complete AI response
    """
    import anthropic
    
    system_prompt = """You are a knowledgeable healthcare AI assistant.

Your role is to:
- Provide accurate, evidence-based medical information
- Keep responses concise, clear, and medically appropriate
- Always remind users to consult healthcare professionals for personal medical advice
- Use a warm, professional, and empathetic tone

Important disclaimers to remember:
- You are not a replacement for professional medical advice
- Encourage users to consult with qualified healthcare providers
- For emergencies, direct users to call emergency services

When discussing medical topics:
- Cite general medical knowledge and best practices
- Explain conditions, symptoms, and treatments clearly
- Avoid making specific diagnoses
- Emphasize the importance of professional medical consultation"""

    try:
        # Use Anthropic's client
        client = anthropic.Anthropic()
        
        # Create message
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."


# ============================================================================
# CHAT INTERFACE
# ============================================================================

def render_chat_message(role: str, content: str):
    """
    Render a chat message in the interface.
    
    Args:
        role: Role of the message sender ('user' or 'assistant')
        content: Message content to display
    """
    avatar = "👤" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def display_chat_history():
    """Display all messages from chat history."""
    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"])


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🏥 Healthcare Agent")
        st.markdown(
            "**Your AI Health Information Assistant** • Ask me about medical conditions, "
            "symptoms, treatments, or general health information."
        )
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_disclaimer():
    """Render medical disclaimer."""
    with st.expander("⚠️ Important Medical Disclaimer - Please Read", expanded=False):
        st.warning("""
**This AI assistant provides general health information for educational purposes only.**

🚫 **NOT a substitute for professional medical advice, diagnosis, or treatment**

✅ **Please remember to:**
- Always consult qualified healthcare professionals for medical concerns
- Call emergency services (911 in US) for medical emergencies
- Do not delay seeking medical advice based on information from this assistant
- Verify any health information with your doctor or healthcare provider

This tool is designed to provide general health education and should not be used for self-diagnosis or treatment decisions.
        """)


def render_example_questions():
    """Render example questions in sidebar."""
    st.sidebar.markdown("### 💡 Example Questions")
    
    examples = [
        "What are the symptoms of seasonal affective disorder?",
        "How is diabetes diagnosed?",
        "What are the latest treatments for migraines?",
        "Explain the benefits of a Mediterranean diet",
        "What should I know about high blood pressure?",
        "How can I improve my sleep quality?"
    ]
    
    st.sidebar.markdown("Click any question below to try it:")
    
    for example in examples:
        if st.sidebar.button(example, key=example, use_container_width=True):
            # Add to chat
            st.session_state.messages.append({"role": "user", "content": example})
            response = get_claude_response(example)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


def render_sidebar_info():
    """Render additional information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About This App")
    st.sidebar.info("""
This healthcare agent is powered by:
- **Claude AI** (Anthropic) for intelligent, empathetic responses
- **Streamlit** for the user interface
- **Medical knowledge base** for evidence-based information
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Features")
    st.sidebar.markdown("""
✅ Real-time AI responses  
✅ Evidence-based information  
✅ Conversational interface  
✅ Medical safety disclaimers  
✅ No API keys required  
    """)


def render_chat_interface():
    """Render the main chat interface."""
    # Display chat history
    display_chat_history()
    
    # Chat input
    if user_input := st.chat_input("Ask a healthcare question... (e.g., 'What are the symptoms of flu?')"):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_chat_message("user", user_input)
        
        # Get AI response with streaming
        with st.chat_message("assistant", avatar="🏥"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            response = get_claude_response(user_input)
            
            # Simulate streaming effect for better UX
            import time
            words = response.split()
            for i, word in enumerate(words):
                full_response += word + " "
                time.sleep(0.02)  # Small delay for streaming effect
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Setup
    setup_page_config()
    initialize_session_state()
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Render disclaimer
    render_disclaimer()
    
    # Render sidebar
    render_sidebar_info()
    render_example_questions()
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9em;'>"
        "🏥 Healthcare Agent • Powered by Claude AI • "
        "Always consult healthcare professionals for medical advice"
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
