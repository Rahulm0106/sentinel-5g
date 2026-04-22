from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.retriever_agent import retrieve
from agents.diagnostician_agent import diagnose, IncidentReport


# 1. Define AgentState
class AgentState(TypedDict):
    query: str
    retrieved_context: str
    incident_report: IncidentReport


# 2. Node functions

def retriever_node(state: AgentState) -> dict:
    result = retrieve(state["query"])
    return {"retrieved_context": result}


def diagnostician_node(state: AgentState) -> dict:
    result = diagnose(state["query"], state["retrieved_context"])
    return {"incident_report": result}


# 3. Build graph
graph = StateGraph(AgentState)

graph.add_node("retriever", retriever_node)
graph.add_node("diagnostician", diagnostician_node)

graph.set_entry_point("retriever")

graph.add_edge("retriever", "diagnostician")
graph.add_edge("diagnostician", END)

app = graph.compile()


# 4. Run function
def run(query: str) -> IncidentReport:
    state = app.invoke({"query": query})
    return state["incident_report"]


# 5. Test
if __name__ == "__main__":
    query = "Why is my 5G handover failing?"

    result = run(query)

    print("\n=== Incident Report ===")
    print(result.model_dump_json(indent=2))