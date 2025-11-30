# -*- coding: utf-8 -*-
"""haystack_ai_agent.ipynb

### Installations and API keys

Install `Haystack` orchestration framework, access `SerperDev API key` and `Open AI API key`.
"""

!python3 -m pip install --upgrade pip
!pip install -q haystack-ai

import os
from haystack.utils import Secret

from haystack.components.generators.chat import OpenAIChatGenerator # First component to connect to LLM

''''Use ChatMessage class to set user or system input''''
from haystack.dataclasses import ChatMessage, Document

''''When the agent is created, its behaviour can be shaped with a system prompt''''
from haystack.components.agents import Agent # Second component where we agent is created

''''When SerperDevWebSearch component is used to retrieve information from the
web, it canreturns a list of results with titles, snippets, links,
and metadata as Haystack Document objects'''
from haystack.components.websearch import SerperDevWebSearch # Third component to create agent's first tool: querying the web

'''ComponentTool class is used to wrap third component, converting it into a tool the agent can reason
about and call using function calling. Metadata from component tool is what the agent will use when
deciding how and when to use this tool''''
from haystack.tools import ComponentTool

'''Print_streaming_chunk callback is used to display tool calls
and their results as the agent runs. This is very useful for debugging
and understanding the agent's behavior in real-time'''
from haystack.components.generators.utils import print_streaming_chunk # This is how you find out how an agent is thinking
                                                                       # and interacting with its tools to arrive
                                                                       # at a final answer.

from dotenv import load_dotenv
from haystack.utils import Secret

# --- Function for OpenAI API Key ---
def set_openai_api_key():
    load_dotenv()  # Load .env if present
    openai_api_key = input("Enter your OpenAI API Key: ")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai_api = os.environ["OPENAI_API_KEY"]
    return Secret.from_token(openai_api)


# --- Function for SerperDev API Key ---
def set_serperdev_api_key():
    load_dotenv()  # Load .env if present
    serperdev_api_key = input("Enter your SerperDev API Key: ")
    os.environ["SERPERDEV_API_KEY"] = serperdev_api_key
    serper_api = os.environ["SERPERDEV_API_KEY"]
    return Secret.from_token(serper_api)

OPENAI_API_KEY = set_openai_api_key()
SERPERDEV_API_KEY = set_serperdev_api_key()

OPENAI_API_KEY = set_openai_api_key()
SERPERDEV_API_KEY = set_serperdev_api_key()

# Example usage:
#generator = OpenAIChatGenerator(api_key=OPENAI_API_KEY)
#search = SerperDevWebSearch(api_key=SERPERDEV_API_KEY)

"""### Configuring a Traceable and Reasonable Agent
Passing the `SerperDevWebsearch` component wrapped in the `Component` tool to the `Agent` instance, add a system prompt that reflects its web-searching capability.

Define `state_schema` to store the retrieved documents.
Use the built-in `print_streaming_chunk` callback to display tool calls and their results as the agent runs.
"""

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    system_prompt="""
    You are a helpful AI assistant that has access to internet.
    Keep your answer concise and use the tools that you're provided with to answer the user's questions.
    """,
    tools=[search_tool],
    state_schema={"documents":{"type":list[Document]}},
    streaming_callback=print_streaming_chunk
)

"""### Running the Agent with Trace Logs
Run the agent with a query and watch the full execution trace:
tool selection, query parameters, document results, and the final LLM response.
"""

message_user = "What are common symptoms of Seasonal Affective Disorder?"

agent_results = agent.run(messages=[ChatMessage.from_user(message_user)])

"""### Inspecting Saved Documents
The documents retrieved by the web tool were saved in the agent's internal state. Inspect them using the `documents` field of the output.
"""

# Wrap the API key with Secret
search_component = SerperDevWebSearch(api_key=Secret.from_token(serperdev_api_key))
results = search_component.run(query=message_user)

results["documents"]
