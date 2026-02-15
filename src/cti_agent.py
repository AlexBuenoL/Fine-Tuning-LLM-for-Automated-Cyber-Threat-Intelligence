import os
import operator
import torch
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Union, List, Optional
from transformers import pipeline, BitsAndBytesConfig
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.language_models.llms import LLM
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# Load environment variable
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Ensure the model is loaded only one time
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

MODEL_PATH = "./models/best_cti_finetuned"
cti_pipeline = pipeline(
    "text-generation",
    model=MODEL_PATH,
    model_kwargs={
        "quantization_config": bnb_config,
        "low_cpu_mem_usage": True
    },
    device_map="auto"
)

# Custom LLM Wrapper
class MistralCTI(LLM):
    @property
    def _llm_type(self) -> str:
        return "mistral-cti-finetuned"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        device = cti_pipeline.model.device
        with torch.no_grad():
            output = cti_pipeline(
                prompt, 
                max_new_tokens=256,
                pad_token_id=cti_pipeline.tokenizer.eos_token_id,
                eos_token_id=cti_pipeline.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1
            )
        return output[0]["generated_text"].split("### Response:")[1].strip()

model_cti_instance = MistralCTI()
    

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# Node Tool definition
@tool
def extract_cyber_entities(text: str):
    """Extract cybersecurity entities (IPs, malware, etc.) from a given text."""
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
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0,
    groq_api_key=GROQ_API_KEY
).bind_tools(tools)

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
    inputs = {"messages": [HumanMessage(content="Analyze this log: 'Malicious traffic was detected from 192.168.1.10 to the domain evil.com.'")]}
    
    print("\n--- INICIANDO AGENTE CTI ---\n")
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"\nNodo ejecutado: {key}")
            if "messages" in value:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, "content"):
                    print(f"Respuesta: {last_msg.content}")
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"Llamada a herramienta: {last_msg.tool_calls[0]['name']}")