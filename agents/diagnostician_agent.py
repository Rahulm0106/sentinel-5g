from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List

from config import config


# 1. Define output schema
class IncidentReport(BaseModel):
    symptoms: List[str] = Field(description="Observed network symptoms")
    possible_causes: List[str] = Field(description="Potential root causes")
    recommended_fix: List[str] = Field(description="Steps to resolve the issue")
    severity: str = Field(description="Severity level: Low, Medium, High, Critical")
    escalate: bool = Field(description="Whether escalation is required")


# 2. Initialize LLM (FIXED: uses config + correct import)
llm = ChatOllama(
    model=config["llm"]["model_name"],
    temperature=0
)


# 3. Create parser
parser = PydanticOutputParser(pydantic_object=IncidentReport)


# 4. Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a senior 5G network engineer diagnosing telecom issues.

Analyze the given context and user query carefully.
Return a structured diagnosis strictly in JSON format.

{format_instructions}
"""),
    ("human",
     """Context:
{retrieved_context}

Query:
{query}
""")
]).partial(format_instructions=parser.get_format_instructions())


# 5. Diagnose function
def diagnose(query: str, retrieved_context: str) -> IncidentReport:
    chain = prompt | llm | parser
    return chain.invoke({
        "query": query,
        "retrieved_context": retrieved_context
    })


# 6. Test
if __name__ == "__main__":
    context = """
    Multiple users in a 5G NSA deployment are experiencing intermittent call drops.
    gNB logs show RRC re-establishment failures.
    High interference detected in the 3.5 GHz band.
    Neighbor cell handover attempts are failing.
    """

    query = "Why are users facing call drops and how to fix it?"

    result = diagnose(query, context)

    print("\n=== Incident Report ===")
    print(result.model_dump_json(indent=2))