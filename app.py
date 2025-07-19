import streamlit as st
from dotenv import load_dotenv
import os
from image_recognize import load_image, get_detected_text, get_detected_label
from langchain_groq import ChatGroq

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

st.set_page_config(page_title="Image + Text Uploader", layout="centered")
st.title("🖼️ Upload an Image and Add Text")

# Load variables from .env file into environment
load_dotenv()

# Access them using os.getenv()
groq_api_key = os.getenv("groq_api_key")
langsmith = os.getenv("langsmith_api_key")

os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="CourseLanggraph"

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")


class State(TypedDict):
  # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
  messages:Annotated[list,add_messages]
  name:str
  image_bytes : bytes
  user_input:str
  detected_text:list
  detected_labels:list
  combined_content:str
  result:str

# File uploader
uploaded_file = st.file_uploader("Upload a PNG or JPG image", type=["png", "jpg", "jpeg"])
# Optional user text input
user_text = st.text_area("Optional: Add some text related to the image", height=150)


image_path="aws-event/assets/2.jpg"

def user_input_node(state:State) -> State:
    if user_text is not None:
        state['user_input'] = user_text
    return state

def get_texts_and_labels_node(state:State) -> State:
    if uploaded_file is not None:
        
        detected_text = get_detected_text(image_bytes)
        detected_label = get_detected_label(image_bytes)

        state['detected_text'] = detected_text
        state['detected_labels'] = detected_label
    else:
        st.write("No image uploaded")

    return state


def combine_content_node(state:State) -> State:
    state['combined_content'] = (
        f"Detected Texts: {state['detected_text']}\n"
        f"Detected Labels: {state['detected_labels']}\n"
        f"Event Description: {state['user_input']}\n"
            )    
    return state

def content_generation_node(state:State) -> State:
    llm_prompt = f""" You are a professional content writer. 
    Based on the following extracted information from an event image and user description, 
    write a concise LinkedIn post (within 75 words) summarizing the event.

    Details:
    {state['combined_content']}
    """

    response = llm.invoke(llm_prompt)
    state['result'] = response.content
    return state

builder = StateGraph(State)

builder.add_node("user_input_node", user_input_node)
builder.add_node("get_texts_and_labels_node", get_texts_and_labels_node)
builder.add_node("combine_content_node", combine_content_node)
builder.add_node("content_generation_node", content_generation_node)

builder.set_entry_point("user_input_node")

builder.add_edge("user_input_node", "get_texts_and_labels_node")
builder.add_edge("get_texts_and_labels_node", "combine_content_node")
builder.add_edge("combine_content_node", "content_generation_node")
builder.add_edge("content_generation_node", END)

graph = builder.compile()


with open("aws-event/langraph_graphs/get_content_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

image_bytes = uploaded_file.read() if uploaded_file else b""

initial_state = {
    "messages": [],
    "name": "",
    "image_bytes": image_bytes,
    "user_input": user_text or "",
    "detected_text": [],
    "detected_labels": [],
    "combined_content": "",
    "result": ""
}

if uploaded_file:
    result = graph.invoke(initial_state)
    st.title("🔵 Final Content:")
    st.write(result["result"])
else:
    st.warning("Please upload an image to generate content.")

print(State)

