from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv()

# Access them using os.getenv()
groq_api_key = os.getenv("groq_api_key")
langsmith = os.getenv("langsmith_api_key")

os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="CourseLanggraph"

from langchain_groq import ChatGroq

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
# print(llm)


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    user_input: str
    llm_response: str
    result: str

# 2. Define logic node
def check_length(state: State) -> State:
    user_text = state["user_input"]
    if len(user_text) > 5:
        return {"result": "Hurray!"}
    else:
        return {"result": "Too short!"}


def call_llm(state: State) -> State:
    prompt = f"Classify this as a question or statement: {state['user_input']}"
    response = llm.invoke(prompt)
    return {"llm_response": response.content}

def classify(state: State) -> State:
    if "question" in state["llm_response"].lower():
        return {"result": "I'll try to answer that."}
    else:
        return {"result": "That's a statement."}
    
# # 3. Build the graph
# graph_builder = StateGraph(State)
# graph_builder.add_node("check_length", check_length)
# graph_builder.set_entry_point("check_length")
# graph_builder.set_finish_point("check_length")  # this node ends the flow
# graph = graph_builder.compile()

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_node("classify", classify)
graph.set_entry_point("llm")
graph.add_edge("llm", "classify")
graph.set_finish_point("classify")

compiled=graph.compile()

# Run
user_input = input("You: ")
result = compiled.invoke({"user_input": user_input})
print("Assistant:", result["result"])

with open("Lan1.png", "wb") as f:
    f.write(compiled.get_graph().draw_mermaid_png())

# # 4. Run the graph
# while True:
#     user_input = input("Enter something (or 'q' to quit): ")
#     if user_input.lower() in ["q", "quit"]:
#         break

#     final_state = graph.invoke({"user_input": user_input})
#     print("Assistant:", final_state["result"])

