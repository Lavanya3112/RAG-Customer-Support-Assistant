# RAG-Based Customer Support Assistant

## A production-grade Retrieval-Augmented Generation (RAG) system with LangGraph workflow orchestration and Human-in-the-Loop (HITL) escalation for customer support automation.

This project was developed as the **Final Evaluation Project** for the Agentic AI Internship at Innomatics Research Labs.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

> Without an API key, the system runs in **demo mode** with mock responses.

---

### 3. Add Your Knowledge Base

Place PDF files in the `./data/` directory:

```text
data/
  HLD_RAG_Customer_Support.pdf
  LLD_RAG_Customer_Support.pdf
  Technical_Documentation_RAG.pdf
```

---

### 4. Run the Assistant

```bash
python main.py
```

---

## Architecture

```text
PDF Files → Load → Chunk → Embed → ChromaDB
                                      ↓
User Query → LangGraph Workflow → Retrieve → Generate → Route
                                                          ↓
                                              Auto-Answer OR HITL Escalation
```

---

## LangGraph Workflow Nodes

| Node             | Responsibility                      |
| ---------------- | ----------------------------------- |
| input_node       | Query validation & preprocessing    |
| retrieve_node    | ChromaDB similarity search          |
| generate_node    | LLM answer generation               |
| routing_function | Conditional routing: output or HITL |
| output_node      | Format response with citations      |
| hitl_node        | Human agent escalation              |

---

## Routing Logic

### Auto-answer

* Confidence ≥ 0.40
* Average similarity score ≥ 0.35

### HITL Escalation

* Confidence < 0.40
* No chunks retrieved
* Sensitive keywords detected (refund, legal complaint, escalation, etc.)

---

## Project Structure

```text
RAG-Customer-Support-Assistant/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── HLD_RAG_Customer_Support.pdf
│   ├── LLD_RAG_Customer_Support.pdf
│   └── Technical_Documentation_RAG.pdf
│
├── screenshots/
│   ├── startup_success.png
│   ├── human_agent_response.png
│   ├── normal_query_response.png
│   ├── project_structure.png
│   └── documentation_preview.png
│
└── chroma_db/
```

---

## Key Technology Choices

### ChromaDB

Used as a persistent vector database for semantic retrieval with minimal setup.

### LangGraph

Used for stateful workflow orchestration and Human-in-the-Loop routing.

### Sentence Transformers

Provides local embedding generation without requiring paid APIs.

### GPT-4o-mini / Mock LLM

Used for response generation with fallback demo mode support.

---

## Example Queries

* What is your return policy?
* I want refund for my cancelled order
* I want to file a legal complaint

---

## Evaluation Metrics Target

| Metric              | Target |
| ------------------- | ------ |
| Faithfulness        | > 90%  |
| Answer Relevance    | > 85%  |
| P95 Latency         | < 3s   |
| Escalation Accuracy | > 95%  |

---

## Documentation Included

* High Level Design (HLD)
* Low Level Design (LLD)
* Technical Documentation
* End-to-End Implementation
* Demo Presentation

---

## Internship Details

* Agentic AI Internship
* Innomatics Research Labs
* Intern ID: IN226104602

---

## Author

Lavanya Dive
