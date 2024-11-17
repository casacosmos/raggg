from typing import Literal, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langgraph.prebuilt import ToolNode
from langchain.tools.retriever import create_retriever_tool

class ConfigSchema(TypedDict):
    model: Optional[str]
    system_message: Optional[str]

embedding = HuggingFaceEmbeddings(model_name="microsoft/graphcodebert-base")


memory = MemorySaver()

# Add to vectorDB
vector_store = Chroma(
    collection_name="langgraph_docs",
    embedding_function=embedding,
    persist_directory="./chroma_store",  # Directory to save the Chroma data
)
retriever = vector_store.as_retriever(k=5)


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)


tools = [retriever_tool]
tool_node = ToolNode(tools)
model = ChatOpenAI(model_name="gpt-4o-mini")
bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    return "action"


def filter_messages(messages: list):
    # This function returns the last 5 messages
    return messages[-5:]


config = {"configurable": {"thread_id": "123", "system_message": "You are a helpful coding assistant. With a retriever tool to search for information in a vector database that contains langgraph documentation."}}


# Define the function that calls the model
def call_model(state: MessagesState):
   messages = filter_messages(state["messages"])
   if "system_message" in config["configurable"]:
      messages = [
         SystemMessage(content=config["configurable"]["system_message"])
      ] + messages
   response = bound_model.invoke(messages)
   # We return a list, because this will get added to the existing list
   return {"messages": response}



 
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["action", END],
)

workflow.add_edge("action", "agent")

graph = workflow.compile(checkpointer=memory)
 
def main():
   while True:
      query = input("Enter a query: ")
      if query == "exit":
         break

      for s in graph.stream({"messages": [HumanMessage(content=query)]}, config=config):
          print(s)
          print("---")

if __name__ == "__main__":
    main()
