# -*- coding: utf-8 -*-
"""
Healthcare Agent Application - Refactored with User-Defined Functions
======================================================================
A modular implementation using Haystack framework for healthcare information retrieval.

"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from haystack.utils import Secret
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack.components.agents import Agent
from haystack.components.websearch import SerperDevWebSearch
from haystack.tools import ComponentTool
from haystack.components.generators.utils import print_streaming_chunk


# ============================================================================
# SECTION 1: API KEY MANAGEMENT
# ============================================================================

def set_openai_api_key() -> Secret:
    """
    Prompt user for OpenAI API key and set it as environment variable.
    
    Returns:
        Secret: Haystack Secret object containing the API key
        
    Example:
        >>> openai_key = set_openai_api_key()
        Enter your OpenAI API Key: sk-...
    """
    load_dotenv()  # Load .env file if present
    openai_api_key = input("Enter your OpenAI API Key: ")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai_api = os.environ["OPENAI_API_KEY"]
    return Secret.from_token(openai_api)


def set_serperdev_api_key() -> Secret:
    """
    Prompt user for SerperDev API key and set it as environment variable.
    
    Returns:
        Secret: Haystack Secret object containing the API key
        
    Example:
        >>> serper_key = set_serperdev_api_key()
        Enter your SerperDev API Key: abc123...
    """
    load_dotenv()  # Load .env file if present
    serperdev_api_key = input("Enter your SerperDev API Key: ")
    os.environ["SERPERDEV_API_KEY"] = serperdev_api_key
    serper_api = os.environ["SERPERDEV_API_KEY"]
    return Secret.from_token(serper_api)


def initialize_api_keys() -> tuple[Secret, Secret]:
    """
    Initialize both OpenAI and SerperDev API keys.
    
    Returns:
        tuple: (OPENAI_API_KEY, SERPERDEV_API_KEY) as Secret objects
        
    Example:
        >>> openai_key, serper_key = initialize_api_keys()
    """
    print("=== API Key Setup ===")
    openai_key = set_openai_api_key()
    serper_key = set_serperdev_api_key()
    print("✅ API keys successfully configured!\n")
    return openai_key, serper_key


# ============================================================================
# SECTION 2: TOOL CREATION
# ============================================================================

def documents_to_string(documents: List[Document]) -> str:
    """
    Convert list of Document objects to formatted string for LLM consumption.
    
    Args:
        documents: List of Document objects from web search results
        
    Returns:
        str: Formatted string containing document content and metadata
        
    Example:
        >>> docs = [Document(content="...", meta={"link": "https://..."})]
        >>> formatted = documents_to_string(docs)
    """
    if not documents:
        return "No documents found."
    
    result_str = ""
    for idx, document in enumerate(documents, 1):
        link = document.meta.get('link', 'No link available')
        content = document.content or "No content available"
        result_str += f"[Source {idx}] {link}\n{content}\n\n"
    return result_str


def create_web_search_tool() -> ComponentTool:
    """
    Create a web search tool using SerperDev API wrapped in ComponentTool.
    
    The ComponentTool wrapper converts the SerperDevWebSearch component into
    a tool that the agent can reason about and call using function calling.
    
    Returns:
        ComponentTool: Configured web search tool for the agent
        
    Features:
        - Retrieves top 5 search results
        - Returns titles, snippets, links, and metadata as Haystack Documents
        - Formats output for LLM consumption
        - Stores documents in agent's internal state
        
    Example:
        >>> search_tool = create_web_search_tool()
        >>> # Tool is now ready to be passed to Agent
    """
    search_tool = ComponentTool(
        component=SerperDevWebSearch(top_k=5),
        name="web_search_tool",
        description="Search the web for current medical and healthcare information",
        outputs_to_string={"source": "documents", "handler": documents_to_string},
        outputs_to_state={"documents": {"source": "documents"}}
    )
    return search_tool


# ============================================================================
# SECTION 3: AGENT CONFIGURATION
# ============================================================================

def configure_healthcare_agent(
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
    enable_streaming: bool = True
) -> Agent:
    """
    Configure and create a traceable and reasonable healthcare agent.
    
    The agent is configured with:
    - OpenAI chat generator for intelligent responses
    - Web search tool for retrieving current information
    - System prompt defining agent behavior
    - State schema for storing retrieved documents
    - Streaming callback for real-time execution tracing
    
    Args:
        model: OpenAI model to use (default: "gpt-4o-mini")
        system_prompt: Custom system prompt (optional)
        enable_streaming: Enable streaming callback for debugging (default: True)
        
    Returns:
        Agent: Configured Haystack agent ready to process queries
        
    Example:
        >>> agent = configure_healthcare_agent()
        >>> # Agent is now ready to process healthcare queries
    """
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """
        You are a knowledgeable healthcare AI assistant with access to the internet.
        
        Your responsibilities:
        - Provide accurate, evidence-based medical information
        - Search the web for the latest healthcare research and guidelines
        - Keep responses concise (2-3 paragraphs maximum)
        - Always cite sources when providing medical information
        - Remind users to consult healthcare professionals for personal medical advice
        
        Important guidelines:
        - You are NOT a replacement for professional medical advice
        - Encourage users to consult with qualified healthcare providers
        - For emergencies, direct users to call emergency services immediately
        - Use the tools provided to answer questions accurately
        """
    
    # Create the web search tool
    search_tool = create_web_search_tool()
    
    # Configure streaming callback
    streaming_callback = print_streaming_chunk if enable_streaming else None
    
    # Create and configure the agent
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model=model),
        system_prompt=system_prompt,
        tools=[search_tool],
        state_schema={"documents": {"type": list[Document]}},
        streaming_callback=streaming_callback
    )
    
    print(f"✅ Healthcare agent configured with model: {model}")
    print(f"   - Web search tool: Enabled")
    print(f"   - Streaming callback: {'Enabled' if enable_streaming else 'Disabled'}")
    print(f"   - Document storage: Enabled\n")
    
    return agent


# ============================================================================
# SECTION 4: AGENT EXECUTION
# ============================================================================

def run_agent_query(
    agent: Agent,
    user_query: str,
    display_trace: bool = True
) -> Dict[str, Any]:
    """
    Run the agent with a user query and display execution trace.
    
    This function executes the agent with full traceability, showing:
    - Tool selection decisions
    - Query parameters sent to tools
    - Document results retrieved
    - Final LLM response generation
    
    Args:
        agent: Configured Agent instance
        user_query: User's question or message
        display_trace: Whether to display execution trace (default: True)
        
    Returns:
        dict: Agent execution results containing messages and state
        
    Example:
        >>> agent = configure_healthcare_agent()
        >>> results = run_agent_query(
        ...     agent,
        ...     "What are symptoms of seasonal affective disorder?"
        ... )
        >>> print(results["messages"][-1].text)
    """
    if display_trace:
        print("=" * 70)
        print(f"USER QUERY: {user_query}")
        print("=" * 70)
        print("\n🔍 Agent Execution Trace:\n")
    
    # Create chat message from user input
    chat_message = ChatMessage.from_user(user_query)
    
    # Run the agent and capture results
    agent_results = agent.run(messages=[chat_message])
    
    if display_trace:
        print("\n" + "=" * 70)
        print("✅ Agent execution completed!")
        print("=" * 70 + "\n")
    
    return agent_results


def extract_agent_response(agent_results: Dict[str, Any]) -> str:
    """
    Extract the final text response from agent results.
    
    Args:
        agent_results: Results dictionary from agent.run()
        
    Returns:
        str: Final assistant response text
        
    Example:
        >>> results = run_agent_query(agent, "What is diabetes?")
        >>> response = extract_agent_response(results)
        >>> print(response)
    """
    if "messages" in agent_results:
        assistant_messages = [
            msg for msg in agent_results["messages"]
            if hasattr(msg, 'role') and msg.role.value == "assistant"
        ]
        if assistant_messages:
            return assistant_messages[-1].text
    return "No response generated."


# ============================================================================
# SECTION 5: DOCUMENT INSPECTION
# ============================================================================

def inspect_saved_documents(
    agent_results: Dict[str, Any],
    display_details: bool = True
) -> List[Document]:
    """
    Inspect documents retrieved and saved in agent's internal state.
    
    Documents retrieved by the web search tool are automatically saved
    in the agent's state schema. This function extracts and optionally
    displays those documents for inspection.
    
    Args:
        agent_results: Results dictionary from agent.run()
        display_details: Whether to print document details (default: True)
        
    Returns:
        list: List of Document objects retrieved during execution
        
    Example:
        >>> results = run_agent_query(agent, "Latest COVID treatments")
        >>> docs = inspect_saved_documents(results)
        >>> print(f"Retrieved {len(docs)} documents")
    """
    # Extract documents from agent state
    documents = agent_results.get("state", {}).get("documents", [])
    
    if display_details:
        print("=" * 70)
        print(f"📄 RETRIEVED DOCUMENTS: {len(documents)} documents")
        print("=" * 70 + "\n")
        
        if not documents:
            print("⚠️  No documents were retrieved during execution.\n")
            return documents
        
        for idx, doc in enumerate(documents, 1):
            print(f"Document {idx}:")
            print(f"  Title: {doc.meta.get('title', 'N/A')}")
            print(f"  Link: {doc.meta.get('link', 'N/A')}")
            print(f"  Content preview: {doc.content[:200]}..." if len(doc.content) > 200 else f"  Content: {doc.content}")
            print(f"  ID: {doc.id}")
            print()
    
    return documents


def verify_search_component(
    serper_api_key: Secret,
    test_query: str = "What are common symptoms of Seasonal Affective Disorder?"
) -> List[Document]:
    """
    Verify that the SerperDev search component is working correctly.
    
    This function directly tests the SerperDevWebSearch component
    without using the agent, useful for debugging and validation.
    
    Args:
        serper_api_key: SerperDev API key as Secret object
        test_query: Query to test the search component (default: SAD symptoms)
        
    Returns:
        list: List of Document objects from search results
        
    Example:
        >>> serper_key = set_serperdev_api_key()
        >>> docs = verify_search_component(serper_key)
        >>> print(f"Search returned {len(docs)} results")
    """
    print("=" * 70)
    print("🔧 TESTING SEARCH COMPONENT")
    print("=" * 70)
    print(f"Query: {test_query}\n")
    
    # Create search component with API key
    search_component = SerperDevWebSearch(api_key=serper_api_key)
    
    # Run search
    results = search_component.run(query=test_query)
    
    # Extract documents
    documents = results.get("documents", [])
    
    print(f"✅ Search completed: {len(documents)} documents retrieved\n")
    
    # Display document metadata
    for idx, doc in enumerate(documents, 1):
        print(f"Result {idx}:")
        print(f"  Title: {doc.meta.get('title', 'N/A')}")
        print(f"  Link: {doc.meta.get('link', 'N/A')}")
        print()
    
    return documents


# ============================================================================
# SECTION 6: MAIN WORKFLOW
# ============================================================================

def main():
    """
    Main workflow demonstrating complete healthcare agent usage.
    
    This function orchestrates the entire agent workflow:
    1. Initialize API keys
    2. Configure the healthcare agent
    3. Run queries with trace logging
    4. Inspect retrieved documents
    5. Verify search component
    """
    print("\n" + "=" * 70)
    print("🏥 HEALTHCARE AGENT - COMPLETE WORKFLOW")
    print("=" * 70 + "\n")
    
    # Step 1: Initialize API keys
    print("STEP 1: Initialize API Keys")
    print("-" * 70)
    openai_key, serper_key = initialize_api_keys()
    
    # Step 2: Configure the agent
    print("\nSTEP 2: Configure Healthcare Agent")
    print("-" * 70)
    agent = configure_healthcare_agent(
        model="gpt-4o-mini",
        enable_streaming=True
    )
    
    # Step 3: Run a sample query
    print("\nSTEP 3: Run Agent Query")
    print("-" * 70)
    test_query = "What are common symptoms of Seasonal Affective Disorder?"
    agent_results = run_agent_query(
        agent=agent,
        user_query=test_query,
        display_trace=True
    )
    
    # Step 4: Extract and display the response
    print("\nSTEP 4: Extract Agent Response")
    print("-" * 70)
    response = extract_agent_response(agent_results)
    print(f"FINAL RESPONSE:\n{response}\n")
    
    # Step 5: Inspect saved documents
    print("\nSTEP 5: Inspect Saved Documents")
    print("-" * 70)
    documents = inspect_saved_documents(agent_results, display_details=True)
    
    # Step 6: Verify search component independently
    print("\nSTEP 6: Verify Search Component")
    print("-" * 70)
    verify_docs = verify_search_component(
        serper_api_key=serper_key,
        test_query=test_query
    )
    
    print("=" * 70)
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


# ============================================================================
# SECTION 7: USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """
    Example: Basic agent usage with minimal configuration.
    """
    print("\n=== EXAMPLE: Basic Usage ===\n")
    
    # Initialize
    openai_key, serper_key = initialize_api_keys()
    
    # Create agent
    agent = configure_healthcare_agent()
    
    # Run query
    results = run_agent_query(agent, "What is type 2 diabetes?")
    
    # Get response
    response = extract_agent_response(results)
    print(f"\nResponse: {response}")


def example_custom_prompt():
    """
    Example: Agent with custom system prompt for specialized behavior.
    """
    print("\n=== EXAMPLE: Custom System Prompt ===\n")
    
    custom_prompt = """
    You are a pediatric healthcare AI assistant.
    Focus on child and adolescent health topics.
    Always use age-appropriate language and examples.
    Emphasize the importance of parental guidance.
    """
    
    openai_key, serper_key = initialize_api_keys()
    agent = configure_healthcare_agent(
        system_prompt=custom_prompt,
        enable_streaming=False  # Disable trace for cleaner output
    )
    
    results = run_agent_query(
        agent,
        "What vaccines do children need?",
        display_trace=False
    )
    
    print(extract_agent_response(results))


def example_document_analysis():
    """
    Example: Detailed analysis of retrieved documents.
    """
    print("\n=== EXAMPLE: Document Analysis ===\n")
    
    openai_key, serper_key = initialize_api_keys()
    agent = configure_healthcare_agent(enable_streaming=False)
    
    # Run query
    results = run_agent_query(
        agent,
        "Latest research on Mediterranean diet benefits",
        display_trace=False
    )
    
    # Analyze documents
    documents = inspect_saved_documents(results, display_details=True)
    
    # Additional analysis
    print("\n📊 Document Statistics:")
    print(f"   Total documents: {len(documents)}")
    print(f"   Average content length: {sum(len(d.content) for d in documents) // len(documents) if documents else 0} chars")
    print(f"   Unique sources: {len(set(d.meta.get('link', '') for d in documents))}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the main workflow
    main()
    
    # Uncomment to run specific examples:
    # example_basic_usage()
    # example_custom_prompt()
    # example_document_analysis()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
# Simple usage
#openai_key, serper_key = initialize_api_keys()
#agent = configure_healthcare_agent()
#results = run_agent_query(agent, "Your question here")
#response = extract_agent_response(results)
#docs = inspect_saved_documents(results)
