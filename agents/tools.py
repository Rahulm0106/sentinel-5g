from langchain.tools import tool
from retrieval.search import search

@tool
def fault_search_tool(query: str) -> str:
    """Use this tool when diagnosing 5G network faults, investigating symptoms,
    finding root causes, or looking up recommended actions for network issues.
    Input should be a natural language description of the fault or symptom.
    Returns relevant fault scenarios with symptoms, causes, and actions."""
    
    results = search(query)
    if not results:
        return "No relevant 5G fault information found."
    
    formatted_output = []
    for i, res in enumerate(results, 1):
        tag = res.metadata.get("tag", "unknown")
        scenario_id = res.metadata.get("scenario_id", "N/A")
        score = res.metadata.get("score", "N/A")
        content = res.page_content.strip()
        formatted_output.append(
            f"{i}. [Tag: {tag} | Scenario: {scenario_id} | Score: {score}]\n"
            f"{content}\n"
        )
    return "\n".join(formatted_output)

if __name__ == "__main__":
    print(fault_search_tool.invoke({"query": "handover failure between gNodeBs"}))