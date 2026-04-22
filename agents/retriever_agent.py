from agents.tools import fault_search_tool
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from agents.tools import fault_search_tool
from config import config


# 1. LLM
llm = ChatOllama(
    model=config["llm"]["model_name"],
    base_url=config["llm"]["base_url"],
    temperature=0
)


# 2. Tools (imported, NOT defined here)
tools = [fault_search_tool]


# 3. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a 5G network fault diagnosis assistant. "
     "Use tools when needed to retrieve fault information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# 4. Tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


# 5. Retrieve function
def retrieve(query: str) -> str:
    response = agent_executor.invoke({"input": query})
    return response["output"]


# 6. Test
if __name__ == "__main__":
    print(retrieve("Why is my 5G handover failing?"))