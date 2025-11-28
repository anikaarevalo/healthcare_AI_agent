"""
Healthcare Agent Application (Streamlit)
Modified to use Streamlit Secrets for API keys and to provide a streaming-like UI.

Notes for deployment on share.streamlit.io:
- Before deploying, add these secrets (in the Streamlit app settings):
    OPENAI_API_KEY: <your OpenAI key>
    SERPERDEV_API_KEY: <your SerperDev key>
- Do NOT collect API keys via UI. Keys are read from st.secrets.
"""

import os
import time
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

def apply_ui_theme():
    """Apply simple calming blue/teal styling to the app."""
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f4fbff 0%, #eef8fb 100%); }
        .stButton>button { background-color: #0ea5a4; border: none; color: white; }
        .disclaimer { background-color: #e6f7f7; padding: 12px; border-radius: 8px; }
        .chat-system { color: #075985; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def initialize_session_state():
    """Initialize session state variables for chat history and agent."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "api_keys_available" not in st.session_state:
        st.session_state.api_keys_available = False

# ============================================================================
# API KEY MANAGEMENT (Streamlit secrets)
# ============================================================================

def load_api_keys_from_secrets() -> bool:
    """
    Load API keys from Streamlit secrets and set environment variables.

    Expected secrets keys:
      - OPENAI_API_KEY
      - SERPERDEV_API_KEY

    Returns True if keys were loaded successfully.
    """
    try:
        # st.secrets will raise a KeyError if missing; handle gracefully
        openai_key = st.secrets.get("OPENAI_API_KEY")
        serper_key = st.secrets.get("SERPERDEV_API_KEY")

        if openai_key and serper_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["SERPERDEV_API_KEY"] = serper_key
            st.session_state.api_keys_available = True
            return True

        st.session_state.api_keys_available = False
        return False
    except Exception as e:
        st.session_state.api_keys_available = False
        st.error(f"Error accessing Streamlit secrets: {e}")
        return False

def render_missing_secrets_message():
    """Show instructions to set secrets in Streamlit Cloud."""
    st.info("API keys are required but not found in Streamlit Secrets.")
    st.markdown(
        """
        **How to provide API keys (recommended):**
        1. In your Streamlit app dashboard (share.streamlit.io) go to **Settings → Secrets**.
        2. Add two secrets:
           - `OPENAI_API_KEY`: *your OpenAI API key*
           - `SERPERDEV_API_KEY`: *your SerperDev API key*
        3. Save and re-deploy the app.
        
        After adding secrets, reload this page and the app will initialize automatically.
        """
    )
    st.warning("For safety, do **not** paste API keys into the app's UI. Use Streamlit Secrets.")

# ============================================================================
# TOOL CREATION
# ============================================================================

def documents_to_string(documents: List[Document]) -> str:
    """
    Convert list of documents to formatted string.
    """
    result_str = ""
    for document in documents:
        link = document.meta.get("link", "No link available")
        content = (document.content or "").strip()
        result_str += f"Content from {link}:\n{content}\n\n"
    return result_str

def create_web_search_tool() -> ComponentTool:
    """
    Create a web search tool using SerperDev API.
    """
    search_tool = ComponentTool(
        component=SerperDevWebSearch(top_k=5),
        name="web_search_tool",
        description="Search the web for current medical and healthcare information",
        outputs_to_string={"source": "documents", "handler": documents_to_string},
        outputs_to_state={"documents": {"source": "documents"}},
    )
    return search_tool

# ============================================================================
# AGENT CREATION
# ============================================================================

def create_healthcare_agent() -> Agent:
    """
    Create and configure the healthcare agent with web search capabilities.
    """
    system_prompt = """
    You are a knowledgeable healthcare AI assistant with access to the internet.

    Your role is to:
    - Provide accurate, evidence-based medical information
    - Search the web for the latest healthcare research and guidelines
    - Keep responses concise, clear, and medically appropriate
    - Always cite sources when providing medical information
    - Remind users to consult healthcare professionals for personal medical advice

    Important disclaimers:
    - You are not a replacement for professional medical advice
    - Encourage users to consult with qualified healthcare providers
    - For emergencies, direct users to call emergency services
    """

    search_tool = create_web_search_tool()

    # Create the agent; model uses OPENAI_API_KEY from environment or secrets
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        system_prompt=system_prompt,
        tools=[search_tool],
        state_schema={"documents": {"type": list[Document]}},
        streaming_callback=None,  # we will implement a simple UI streaming effect
    )
    return agent

# ============================================================================
# CHAT INTERFACE & STREAMING UTILS
# ============================================================================

def render_chat_message(role: str, content: str):
    avatar = "👤" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def display_chat_history():
    """Display all messages from chat history."""
    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"])

def simple_stream_display(text: str, placeholder, delay: float = 0.03):
    """
    Simulate streaming by gradually revealing the text in the placeholder.
    Splits by words for a natural effect. Blocking but small delays are used.
    """
    words = text.split()
    current = []
    for w in words:
        current.append(w)
        placeholder.markdown(" ".join(current) + "▌")
        time.sleep(delay)
    # final render without the caret
    placeholder.markdown(" ".join(current))

def process_user_query(user_input: str) -> str:
    """
    Process user query through the agent and return response.
    This function returns the full assistant text, but also the caller
    may stream the text with `simple_stream_display`.
    """
    try:
        chat_message = ChatMessage.from_user(user_input)

        # Ensure agent exists
        if st.session_state.agent is None:
            st.session_state.agent = create_healthcare_agent()

        # Run the agent (synchronous). Haystack returns a structured result.
        with st.spinner("🔍 Searching and analyzing..."):
            result = st.session_state.agent.run(messages=[chat_message])

        # Extract assistant message text from result, if present.
        assistant_text = None
        if result:
            # Haystack's output structure may vary depending on version.
            # Try common patterns:
            if isinstance(result, dict) and "messages" in result:
                # result["messages"] may be a list of ChatMessage-like objects
                assistant_messages = [
                    msg for msg in result["messages"] if getattr(msg, "role", None) and getattr(msg.role, "value", None) == "assistant"
                ]
                if assistant_messages:
                    assistant_text = getattr(assistant_messages[-1], "text", None) or str(assistant_messages[-1])
            elif isinstance(result, dict) and "output" in result:
                assistant_text = result.get("output")
            elif isinstance(result, str):
                assistant_text = result

        if not assistant_text:
            assistant_text = "I apologize — I couldn't generate a response. Please try again."

        return assistant_text

    except Exception as e:
        st.error(f"Error processing query: {e}")
        return "An error occurred while processing your request. Please check configuration."

def render_chat_interface():
    """Render the main chat interface."""
    st.markdown("<h1 class='chat-system'>🏥 Healthcare Agent</h1>", unsafe_allow_html=True)
    st.markdown(
        "Ask about medical conditions, symptoms, treatments, or general health information. "
        "The agent will search current sources and provide evidence-backed answers."
    )

    # Disclaimer
    with st.expander("⚠️ Important Medical Disclaimer"):
        st.markdown(
            """
            <div class="disclaimer">
            This AI assistant provides general health information for educational purposes only.
            It is NOT a substitute for professional medical advice, diagnosis, or treatment.<br><br>
            - Always consult qualified healthcare professionals for medical concerns.<br>
            - In case of emergency, call your local emergency services immediately.<br>
            - Do not delay seeking medical advice based on information from this assistant.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display history
    display_chat_history()

    # Chat input
    if user_input := st.chat_input("Ask a healthcare question..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_chat_message("user", user_input)

        # Create a placeholder for assistant streaming message
        with st.chat_message("assistant", avatar="🏥"):
            assistant_placeholder = st.empty()

        # Get full assistant response
        assistant_full_text = process_user_query(user_input)

        # Stream it word by word into the placeholder
        simple_stream_display(assistant_full_text, assistant_placeholder, delay=0.03)

        # Save assistant message to session history
        st.session_state.messages.append({"role": "assistant", "content": assistant_full_text})

# ============================================================================
# SIDEBAR INFO
# ============================================================================

def render_sidebar_info():
    """Render information in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.header("📊 About")
        st.markdown(
            """
            This healthcare agent uses:
            - **Haystack** orchestration framework
            - **OpenAI GPT-4o-mini** for intelligent responses
            - **SerperDev** for web search capabilities
            """
        )

        st.markdown("---")
        st.header("💡 Example Questions")
        st.markdown(
            """
            - What are the symptoms of seasonal affective disorder?
            - How is diabetes diagnosed?
            - What are the latest treatments for migraines?
            - Explain the benefits of a Mediterranean diet
            """
        )

        st.markdown("---")
        st.header("🔧 Features")
        st.markdown(
            """
            ✅ Real-time web search  \n
            ✅ Evidence-based responses  \n
            ✅ Source citations  \n
            ✅ Conversational interface
            """
        )

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    setup_page_config()
    apply_ui_theme()
    initialize_session_state()

    # Try to load API keys from Streamlit secrets immediately
    keys_loaded = load_api_keys_from_secrets()

    # If keys not loaded, instruct the user to add them in Streamlit Cloud secrets
    if not keys_loaded:
        st.title("🏥 Healthcare Agent (Setup Required)")
        render_missing_secrets_message()
        st.stop()  # halt; wait for secrets to be added and app reloaded

    # Keys available → create agent if needed
    if st.session_state.agent is None:
        with st.spinner("Initializing healthcare agent..."):
            st.session_state.agent = create_healthcare_agent()

    # Render sidebar and main chat
    render_sidebar_info()
    render_chat_interface()

if __name__ == "__main__":
    main()
