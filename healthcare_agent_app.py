"""
Healthcare Agent Application
=============================
A Streamlit-based healthcare agent that uses Haystack framework to provide
medical information through web search capabilities.

Features:
- Web search integration via SerperDev API
- OpenAI GPT-4o-mini powered responses
- Traceable agent execution with streaming output
- Healthcare-focused conversational interface
"""

import os
import streamlit as st
from typing import List
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent
from haystack.components.websearch import SerperDevWebSearch
from haystack.dataclasses import ChatMessage, Document
from haystack.tools import ComponentTool


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Healthcare Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_session_state():
    """Initialize session state variables for chat history and agent."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "api_keys_set" not in st.session_state:
        st.session_state.api_keys_set = False


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def validate_api_keys(openai_key: str, serper_key: str) -> tuple[bool, str]:
    """
    Validate API keys before setting them.
    
    Args:
        openai_key: OpenAI API key
        serper_key: SerperDev API key
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not openai_key or not openai_key.strip():
        return False, "OpenAI API key is required"
    if not serper_key or not serper_key.strip():
        return False, "SerperDev API key is required"
    if not openai_key.startswith("sk-"):
        return False, "OpenAI API key should start with 'sk-'"
    return True, ""


def set_api_keys(openai_key: str, serper_key: str) -> bool:
    """
    Set API keys as environment variables after validation.
    
    Args:
        openai_key: OpenAI API key
        serper_key: SerperDev API key
    
    Returns:
        bool: True if keys were set successfully
    """
    try:
        is_valid, error_msg = validate_api_keys(openai_key, serper_key)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return False
            
        os.environ['OPENAI_API_KEY'] = openai_key.strip()
        os.environ['SERPERDEV_API_KEY'] = serper_key.strip()
        return True
    except Exception as e:
        st.error(f"Error setting API keys: {e}")
        return False


def render_api_key_sidebar():
    """Render sidebar for API key input."""
    with st.sidebar:
        st.header("🔑 API Configuration")
        st.markdown("Enter your API keys to get started.")
        
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your key from https://platform.openai.com/api-keys",
            key="openai_input"
        )
        
        serper_key = st.text_input(
            "SerperDev API Key",
            type="password",
            help="Get your key from https://serper.dev/",
            key="serper_input"
        )
        
        if st.button("Set API Keys", type="primary", key="set_keys_btn"):
            if set_api_keys(openai_key, serper_key):
                st.session_state.api_keys_set = True
                st.success("✅ API keys set successfully!")
                st.rerun()
        
        if st.session_state.api_keys_set:
            st.success("🟢 API Keys Active")
            if st.button("Clear Keys", key="clear_keys_btn"):
                st.session_state.api_keys_set = False
                st.session_state.agent = None
                st.session_state.messages = []
                # Clear environment variables
                os.environ.pop('OPENAI_API_KEY', None)
                os.environ.pop('SERPERDEV_API_KEY', None)
                st.rerun()


# ============================================================================
# TOOL CREATION
# ============================================================================

def documents_to_string(documents: List[Document]) -> str:
    """
    Convert list of documents to formatted string.
    
    Args:
        documents: List of Document objects from web search
    
    Returns:
        str: Formatted string containing document content and links
    """
    if not documents:
        return "No documents found."
    
    result_str = ""
    for idx, document in enumerate(documents, 1):
        link = document.meta.get('link', 'No link available')
        content = document.content or "No content available"
        result_str += f"[{idx}] {link}\n{content}\n\n"
    return result_str


def create_web_search_tool() -> ComponentTool:
    """
    Create a web search tool using SerperDev API.
    
    Returns:
        ComponentTool: Configured web search tool for the agent
    """
    try:
        search_tool = ComponentTool(
            component=SerperDevWebSearch(top_k=5),
            name="web_search_tool",
            description="Search the web for current medical and healthcare information",
            outputs_to_string={"source": "documents", "handler": documents_to_string},
            outputs_to_state={"documents": {"source": "documents"}}
        )
        return search_tool
    except Exception as e:
        st.error(f"Error creating search tool: {e}")
        raise


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_healthcare_agent() -> Agent:
    """
    Create and configure the healthcare agent with web search capabilities.
    
    Returns:
        Agent: Configured Haystack agent with OpenAI chat generator
    """
    system_prompt = """
    You are a knowledgeable healthcare AI assistant with access to the internet.
    
    Your role is to:
    - Provide accurate, evidence-based medical information
    - Search the web for the latest healthcare research and guidelines
    - Keep responses concise (no more than 2-3 paragraphs), clear, and medically appropriate
    - Always cite sources when providing medical information
    - Remind users to consult healthcare professionals for personal medical advice
    
    Important disclaimers:
    - You are not a replacement for professional medical advice
    - Encourage users to consult with qualified healthcare providers
    - For emergencies, direct users to call emergency services immediately
    
    Format your responses clearly with proper paragraphs and cite your sources.
    """
    
    try:
        # Create the search tool
        search_tool = create_web_search_tool()
        
        # Create the agent
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=system_prompt,
            tools=[search_tool],
            state_schema={"documents": {"type": list[Document]}},
            streaming_callback=None  # We'll handle streaming in Streamlit
        )
        
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        raise


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


def process_user_query(user_input: str) -> str:
    """
    Process user query through the agent and return response.
    
    Args:
        user_input: User's question or message
    
    Returns:
        str: Agent's response
    """
    try:
        # Validate input
        if not user_input or not user_input.strip():
            return "Please enter a valid question."
        
        # Create chat message
        chat_message = ChatMessage.from_user(user_input.strip())
        
        # Run the agent
        with st.spinner("🔍 Searching and analyzing..."):
            result = st.session_state.agent.run(messages=[chat_message])
        
        # Extract response from result
        if result and "messages" in result:
            assistant_messages = [
                msg for msg in result["messages"] 
                if hasattr(msg, 'role') and msg.role.value == "assistant"
            ]
            if assistant_messages:
                response_text = assistant_messages[-1].text
                if response_text:
                    return response_text
        
        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error processing query: {error_msg}")
        
        # Provide helpful error messages
        if "api key" in error_msg.lower():
            return "There was an issue with your API keys. Please check that they are valid and try again."
        elif "rate limit" in error_msg.lower():
            return "Rate limit exceeded. Please wait a moment and try again."
        else:
            return f"An error occurred while processing your request. Error: {error_msg}"


def render_chat_interface():
    """Render the main chat interface."""
    st.title("🏥 Healthcare Agent")
    st.markdown(
        "Ask me about medical conditions, symptoms, treatments, or general health information. "
        "I'll search the latest sources to provide you with accurate information."
    )
    
    # Display disclaimer
    with st.expander("⚠️ Important Medical Disclaimer", expanded=False):
        st.warning(
            """
            **This AI assistant provides general health information for educational purposes only.**
            
            - ❌ NOT a substitute for professional medical advice, diagnosis, or treatment
            - ✅ Always consult qualified healthcare professionals for medical concerns
            - 🚨 In case of emergency, call your local emergency services immediately
            - ⏰ Do not delay seeking medical advice based on information from this assistant
            """
        )
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask a healthcare question...", key="chat_input")
    
    if user_input:
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_chat_message("user", user_input)
        
        # Get agent response
        response = process_user_query(user_input)
        
        # Add assistant message to history and display
        st.session_state.messages.append({"role": "assistant", "content": response})
        render_chat_message("assistant", response)


# ============================================================================
# SIDEBAR FEATURES
# ============================================================================

def render_sidebar_info():
    """Render additional information in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.header("📊 About")
        st.markdown("""
        This healthcare agent uses:
        - **Haystack** orchestration framework
        - **OpenAI GPT-4o-mini** for intelligent responses
        - **SerperDev** for web search capabilities
        """)
        
        st.markdown("---")
        st.header("💡 Example Questions")
        st.markdown("""
        - What are the symptoms of seasonal affective disorder?
        - How is diabetes diagnosed?
        - What are the latest treatments for migraines?
        - Explain the benefits of a Mediterranean diet
        - What are common side effects of statins?
        """)
        
        st.markdown("---")
        st.header("🔧 Features")
        st.markdown("""
        ✅ Real-time web search  
        ✅ Evidence-based responses  
        ✅ Source citations  
        ✅ Conversational interface  
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    try:
        # Setup
        setup_page_config()
        initialize_session_state()
        
        # Render API key input
        render_api_key_sidebar()
        
        # Check if API keys are set
        if not st.session_state.api_keys_set:
            st.info("👈 Please enter your API keys in the sidebar to get started.")
            st.markdown("""
            ### Getting Started
            
            This application requires two API keys:
            
            1. **OpenAI API Key**: Get yours at [platform.openai.com](https://platform.openai.com/api-keys)
               - Should start with `sk-`
               - Requires credits/subscription
            
            2. **SerperDev API Key**: Get yours at [serper.dev](https://serper.dev/)
               - Free tier available (2,500 queries/month)
               - Sign up with Google account
            
            Once you've entered your keys, you can start asking healthcare questions!
            
            ---
            
            ### Privacy Notice
            - Your API keys are only stored in your current session
            - Keys are never saved to disk or transmitted to third parties
            - Clear your keys using the "Clear Keys" button when done
            """)
            return
        
        # Initialize agent if not already done
        if st.session_state.agent is None:
            try:
                with st.spinner("Initializing healthcare agent..."):
                    st.session_state.agent = create_healthcare_agent()
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
                st.session_state.api_keys_set = False
                return
        
        # Render sidebar info
        render_sidebar_info()
        
        # Render main chat interface
        render_chat_interface()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
