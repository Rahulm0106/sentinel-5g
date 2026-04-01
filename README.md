# SENTINEL-5g 🛰️
> Self-correcting 5G Network Fault Intelligence Platform powered by Multi-Agent RAG

## What it does
- 🔍 Autonomously diagnoses 5G network faults using a multi-agent RAG pipeline
- 🧠 Self-corrects bad retrievals using Corrective RAG and confidence scoring
- 📡 Monitors KPIs and detects anomalies across RAN and Core network domains
- 🔌 Exposes diagnostic tools via MCP for integration with any AI client
- 📊 Live Streamlit dashboard for real-time fault investigation

## Tech Stack
| Layer | Tool |
|---|---|
| Vector DB | Qdrant |
| Embeddings | all-MiniLM-L6-v2 |
| LLM | Ollama + llama3.2 (local) |
| Agent Framework | LangGraph |
| MCP | MCP Python SDK |
| Observability | LangSmith |
| UI | Streamlit |

## Dataset
[telco-5G-data-faults](https://huggingface.co/datasets/greenwich157/telco-5G-data-faults) — 470 real 5G fault scenarios across RAN and Core domains

## Setup
```bash
# Coming soon
```