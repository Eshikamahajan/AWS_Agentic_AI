from dotenv import load_dotenv
import os
from image_recognize import load_image, detect_text, detect_label
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

detected_text = [('Generative A', '88.83927917480469'), ('APPLICATIONS TO BOOST PRODUCTIVITY', '89.04224395751953'), ('Amazon Q Developer', '97.84193420410156'), ('Amazon Q Business', '96.90839385986328'), ('SOFTWARE DEVELOPMENT LIFE CYCLE', '96.96534729003906'), ('INSIGHTS AND AUTOMATION', '96.9379653930664'), ('MODELS AND TOOLS TO BUILD GENERATIVE AI APPS', '96.78956604003906'), ('Generative Al stack', '98.73432159423828'), ('Amazon Bedrock', '98.2164306640625'), ('APPLICATIONS TO BOOST PRODUCTIVITY', '94.23150634765625'), ('Amazon Q Developer', '98.22669982910156'), ('AMAZON MODELS PARTNER MODELS', '98.40999603271484'), ('Amazon Q Business', '97.71945190429688'), ('SOFTWARE DEVELOPMENT LIFE CYCLE', '97.55355072021484'), ('INSIGHTS AND AUTOMATION', '97.43328857421875'), ('INFRASTRUCTURE TO BUILD AND TRAIN AI MODELS', '91.96792602539062'), ('AWS Trainium', '98.17542266845703'), ('Amazon SageMaker', '98.01686096191406'), ('GPUs', '96.07637786865234'), ('MODELS AND TOOLS TO BUILD GENERATIVE AI APPS', '95.73411560058594'), ('AWS Inferentia', '94.72759246826172'), ('MANAGED INFRASTRUCTURE', '97.95304107666016'), ('HIGH PERFORMANCE', '98.10088348388672'), ('Amazon Bedrock', '97.9629898071289'), ('AMAZON MODELS PARTNER MODELS', '98.47936248779297'), ('AWS Community All rights reserved. Amazo', '66.64896392822266'), ('7', '92.72494506835938'), ('INFRASTRUCTURE TO BUILD AND TRAIN AI MODELS', '97.16600036621094'), ('Amazon SageMaker', '97.67601776123047'), ('AWS', '98.4103775024414')]

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

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

graph_builder=StateGraph(State)

image_path="aws-event/assets/2.jpg"

def user_input_node(state:State) -> State:
    state['user_input'] = input("User: ")
    return state

def get_texts_and_labels_node(state:State) -> State:

    # image_bytes = load_image(image_path)
    detected_text = detect_text(image_bytes)
    detected_label = detect_label(image_bytes)

    state['detected_text'] = detected_text
    state['detected_labels'] = detected_label

    return state


def combine_content_node(state:State) -> State:
    state['combined_content'] = f"Detected Texts: {state['detected_text']}\nDetected Labels: {state['detected_labels']}\Event Description: {state['user_input']}\n"
    return state

def content_generation_node(state:State) -> State:
    llm_prompt = f""" Given the notes, detected text and labels, write a short,crisp linkedin post of 75 words making sense of the text
    combined_text : {state['combined_content']}"""

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

result = graph.invoke({})
print("\n🔵 Final Content:")
print(result["result"])


print(State)


# llm_prompt = f""" Given a list of detected texts and their confidence scores, write a short paragraph of 50 words making sense of the text
# detected_text : {detected_text}"""

# response = llm.invoke(llm_prompt)
# print("LLM Response:", response.content)