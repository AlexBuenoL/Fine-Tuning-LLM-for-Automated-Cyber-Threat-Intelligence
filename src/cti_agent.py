import operator
import json
import torch
from typing import Annotated, TypedDict, Union, List, Optional
from transformers import pipeline
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# Ensure the model is loaded only one time
MODEL_PATH = "./models/best_cti_finetuned"
cti_pipeline = pipeline(
    "text-generation",
    model=MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

# Custom LLM Wrapper
class MistralCTI(LLM):
    @property
    def _llm_type(self) -> str:
        return "mistral-cti-finetuned"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = cti_pipeline(prompt, max_new_tokens=256)
        return output[0]["generated_text"].split("### Response:")[1].strip()

model_cti_instance = MistralCTI()
    

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# Node Tool definition
@tool
def extract_cyber_entities(text: str):
    prompt = (
        f"### Instruction: Extract cyber threat entities in JSON format.\n"
        f"### Input: {text}\n"
        f"### Response:"
    )

    json_entities = model_cti_instance.invoke(prompt)

    return json_entities

tools = [extract_cyber_entities]
tool_node = ToolNode(tools)


# Node Agent definition
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Node Control logic
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    return END


# Graph construction
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue) # Agent -> Action | End
graph.add_edge("action", "agent")                     # Action -> Agent (Always make the model observe and reason after an action)

app = graph.compile()


if __name__ == '__main__':
    final_state = app.invoke({
        "messages": [HumanMessage(content="Analyze this log and extract entities: 'Malicious traffic was detected from 192.168.1.10 to the domain evil.com.'")]
    })
    
    print(final_state["messages"][-1].content)