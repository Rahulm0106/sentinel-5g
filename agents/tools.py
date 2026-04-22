from langchain.tools import Tool
from retrieval.search import search


# 1. Function: search_faults
def search_faults(query: str) -> str:
    """
    Searches 5G fault-related knowledge base and returns formatted results.
    """
    results = search(query)

    if not results:
        return "No relevant 5G fault information found."

    formatted_output = []

    for i, res in enumerate(results, 1):
        # Extract metadata fields
        tag = res.metadata.get("tag", "unknown")
        scenario_id = res.metadata.get("scenario_id", "N/A")
        score = res.metadata.get("score", "N/A")

        content = res.page_content.strip()

        formatted_output.append(
            f"{i}. [Tag: {tag} | Scenario: {scenario_id} | Score: {score}]\n"
            f"{content}\n"
        )

    return "\n".join(formatted_output)


# 2. Tool object: fault_search_tool
fault_search_tool = Tool(
    name="search_faults",
    description=(
        "Use this tool when diagnosing 5G network faults, investigating symptoms, "
        "finding root causes, or looking up recommended actions for network issues. "
        "Input should be a natural language description of the fault or symptom. "
        "Returns relevant fault scenarios with symptoms, causes, and actions."
    ),
    func=search_faults,
)

if __name__ == "__main__":
    print(search_faults("handover failure between gNodeBs"))