## About the Healthcare AI Agent

Capable of navigating the the web in real time, I constructed a healthcare AI agent using [Haystack](https://haystack.deepset.ai/)--an open-source (modular, model-agnostic) orchestration framework for LLM-based applications. This project is inspired by DataCamp’s ["Building AI Agents with Haystack"](https://app.datacamp.com/learn/courses/building-ai-agents-with-haystack) interactive course.

My goal in this project is to prototype my own **tool-using AI agent** that is capable of interpreting natural language queries, make autonomous (accurate and appropriate) decisions, and act on them using the right tools at the right time.

For this project, the use of OpenAI API and SerperDev API keys are necessary. This allows the AI agent access to gpt-4o-mini (a cost-effective OpenAI model) and source/retrieve data online. 

This app is a **minimum viable product (MVP)** reflecting key agentic design principles and integration of components (Agent, Chat Generator, and Tool class) into an intelligent system as reusable pipelines.

---

## App Deployment via Streamlit

OpenAI’s [ChatGPT-5](https://chatgpt.com/) transformed my source code ("haystack_ai_agent.py") into a production-ready Streamlit application. It is deployed on [Streamlit cloud](https://share.streamlit.io/).

Here are the key transformations I requested and were implemented:

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

1.  To access the Healthcare AI Agent app click [here](https://healthcareaiagentbyanika.streamlit.app).
2.  Start asking healthcare questions.
3.  The agent will search the web and provide evidence-based responses.

---

## Visuals

### web version
<img width="650" alt="pipeline 2022-04-21 at 15 49 28" src="https://github.com/anikaarevalo/healthcare_AI_agent/blob/40c856dc9f4d1db89c2bb94b58da5742228a8c59/assets/web_healthcare_agent.png">


### mobile version
<img width="250" alt="pipeline 2022-04-21 at 15 49 28" src="https://github.com/anikaarevalo/healthcare_AI_agent/blob/40c856dc9f4d1db89c2bb94b58da5742228a8c59/assets/mobile_healthcare_agent.jpeg">


---

## Future Plans

To make the app production-grade, testing is required across multiple dimensions: validating user interactions, performing unit tests and qualitative evaluations, and ensuring guardrails and observability.

1.  Upgrading the agent's main functionality of fetching data (source to date is web based) and generating a response with the following features:
    * Persistent chat history across sessions
    * Conversation export
    * Real streaming callbacks
2.  Building more reusable **Haystack pipelines** by implementing a second, complementary core functionality: querying structured personal (patient) data from a private database.
3.  Enabling access to project functionality via MCP tooling (i.e., serving as MCP tool to MCP clients).
4.  Developing own machine learning model (instead of relying on commercially-available gen AI models) for the AI healthcare agent to interface with. 
