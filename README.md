## About the Healthcare AI Agent

I constructed a healthcare AI agent capable of searching the web using **Haystack**, an open-source (modular, model-agnostic) orchestration framework for LLM-based applications. This project is inspired by DataCamp’s “Building AI Agents with Haystack” interactive course.

My goal in this project is to prototype my own **tool-using AI agent** that is capable of interpreting natural language queries, make autonomous (accurate and appropriate) decisions, and act on them using the right tools at the right time.

This app is a **minimum viable product (MVP)** reflecting key agentic design principles and integration of components (Agent, Chat Generator, and Tool class) into an intelligent system.

---

## App Deployment via Streamlit

OpenAI’s **ChatGPT-5** transformed my source code into a production-ready Streamlit application to be deployed on Streamlit cloud.

Here are the key transformations implemented:

### 1. Modular Structure
* Organized code into clear sections: Configuration, API Management, Tool Creation, Agent Creation, Chat Interface, Main Application, and also a Sidebar
* Each function has a single responsibility with proper docstrings

### 2. User-Friendly Features
* **Chat Interface**: Clean, conversational chat interface with scrolling message history, simulated streaming responses, and a medical-themed UI
* **Medical Disclaimer**: Important warning about professional medical advice
* **Example Questions**: Helpful suggestions in the sidebar
* **Clear Chat**: Button to reset conversation

### 3. Production-Ready Elements
* Session state management for persistence
* Error handling throughout
* User feedback (spinners, success/error messages)
* Proper configuration for web deployment

---

## Usage of the App

1.  Go to [https://healthcareaiagentbyanika.streamlit.app/](https://healthcareaiagentbyanika.streamlit.app/) to access the Healthcare AI Agent app.
2.  Start asking healthcare questions.
3.  The agent will search the web and provide evidence-based responses.

---

## Visuals

### web version
<img width="650" alt="pipeline 2022-04-21 at 15 49 28" src="">


### mobile version
<img width="650" alt="pipeline 2022-04-21 at 15 49 28" src="">


---

## Future Plans

To make my app production-grade, testing is required across multiple dimensions: validating user interactions, performing unit tests and qualitative evaluations, and ensuring guardrails and observability.

1.  Upgrading the main functionality of fetching and generating a response from the web with the following features:
    * Persistent chat history across sessions
    * Conversation export
    * Real OpenAI token with streaming callbacks
2.  Building reusable **Haystack pipelines** with a second, complementary core functionality: querying structured personal (patient) data from a private database.
3.  Serving the MCP tool for MCP servers.
