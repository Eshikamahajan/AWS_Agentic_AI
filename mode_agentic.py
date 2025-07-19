import os
from typing import Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.agents import tool, initialize_agent, AgentType
from langchain.agents.agent_toolkits import Tool
from langchain_groq import ChatGroq

from image_recognize import get_detected_text, get_detected_label

load_dotenv()

# 1️⃣ Define the shared state
class State(TypedDict):
    image_bytes: Optional[bytes]
    user_input: Optional[str]
    extracted_text: Optional[str]
    labels: Optional[list]
    content: Optional[str]
    final_output: Optional[str]


# 2️⃣ Define tool-wrapped functions
@tool
def extract_text_from_image(image_bytes: bytes) -> str:
    return get_detected_text(image_bytes)

@tool
def extract_labels_from_image(image_bytes: bytes) -> list:
    return get_detected_label(image_bytes)

@tool
def combine_with_user_input(text: str, user_input: str) -> str:
    return f"{text} | Additional context: {user_input}"

@tool
def generate_linkedin_post(content: str) -> str:
    return f"Here’s a polished LinkedIn post:\n\n{content}"


# 3️⃣ LLM via Groq
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# 4️⃣ Initialize agent with tools
agent_executor = initialize_agent(
    tools=[
        extract_text_from_image,
        extract_labels_from_image,
        combine_with_user_input,
        generate_linkedin_post
    ],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # still compatible with Groq
    verbose=True
)


# 5️⃣ Node to run the agent dynamically
def agent_node(state: State) -> State:
    print("\n--- Agent Node Invoked ---")
    print("Current state:", state)

    # Prompt context for agent
    context = "You're an intelligent assistant that builds LinkedIn posts from images and optional user text. Use the tools available. "
    if state.get("user_input"):
        context += "User has provided additional text to combine."

    input_payload = {
        "image_bytes": state.get("image_bytes"),
        "user_input": state.get("user_input"),
        "text": state.get("extracted_text"),
        "content": state.get("content")
    }

    # Let the agent reason & call tools
    result = agent_executor.run(context)
    return {**state, "final_output": result}


# 6️⃣ Define LangGraph with agent node
builder = StateGraph(State)
builder.set_entry_point("agent_node")
builder.add_node("agent_node", agent_node)
builder.set_finish_point("agent_node")
graph = builder.compile()
# Optionally visualize graph
with open("aws-event/langraph_graphs/agentic_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

# 7️⃣ Streamlit interface
st.title("Agentic LinkedIn Post Generator (Groq + LangGraph)")

uploaded_file = st.file_uploader("Upload an image 📸")
user_input = st.text_area("Optional: Add a message or context 📝")

if st.button("🚀 Generate Post"):
    if uploaded_file:
        image_bytes = uploaded_file.read()

        initial_state = {
            "image_bytes": image_bytes,
            "user_input": user_input
        }

        final_result = graph.invoke(initial_state)

        st.success("✅ Generated LinkedIn Post")
        st.write(final_result["final_output"])
    else:
        st.error("❗ Please upload an image to begin.")
