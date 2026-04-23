"""
RAG-Based Customer Support Assistant
With LangGraph Workflow & Human-in-the-Loop (HITL)
Author: Innomatics Research Labs Internship Project
"""

import os
import hashlib
import uuid
import time
from datetime import datetime
from typing import List, Dict, Optional, Literal, Tuple
from typing_extensions import TypedDict

# ─── LangChain Imports ───────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ─── LangGraph Imports ───────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

# ─── LLM Imports ─────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ─── Environment ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.40
SCORE_THRESHOLD = 0.35
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "customer_support_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Escalation keywords (pre-classification)
ESCALATION_KEYWORDS = [
    "refund", "lawsuit", "legal", "sue", "complaint", "fraud",
    "scam", "urgent", "emergency", "escalate", "manager", "supervisor"
]


# =============================================================================
# STATE DEFINITION
# =============================================================================

class RAGState(TypedDict):
    """Shared state object that flows through the LangGraph workflow."""
    query: str
    processed_query: str
    retrieved_chunks: List[Document]
    retrieval_scores: List[float]
    llm_response: str
    confidence: float
    final_answer: str
    sources: List[str]
    escalated: bool
    human_answer: Optional[str]
    error: Optional[str]
    latency_ms: int
    start_time: float


# =============================================================================
# MODULE 1: DOCUMENT PROCESSOR
# =============================================================================

class DocumentProcessor:
    """Loads and extracts text from PDF files."""

    def load_documents(self, pdf_paths: List[str]) -> List[Document]:
        """Load multiple PDFs and return list of Document objects."""
        all_docs = []
        for path in pdf_paths:
            try:
                print(f"  [+] Loading: {path}")
                loader = PyPDFLoader(path)
                docs = loader.load()
                # Enrich metadata
                for doc in docs:
                    doc.metadata["filename"] = os.path.basename(path)
                all_docs.extend(docs)
                print(f"      Loaded {len(docs)} pages from {os.path.basename(path)}")
            except FileNotFoundError:
                print(f"  [!] WARNING: File not found: {path}. Skipping.")
            except Exception as e:
                print(f"  [!] WARNING: Failed to load {path}: {e}. Skipping.")
        return all_docs


# =============================================================================
# MODULE 2: CHUNKING ENGINE
# =============================================================================

class ChunkingEngine:
    """Splits documents into overlapping chunks."""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks with enriched metadata."""
        chunks = self.splitter.split_documents(docs)
        # Add chunk index and doc_id to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["doc_id"] = hashlib.sha256(
                chunk.page_content.encode()
            ).hexdigest()[:16]
            chunk.metadata["char_count"] = len(chunk.page_content)
        print(f"  [+] Created {len(chunks)} chunks from {len(docs)} pages")
        return chunks


# =============================================================================
# MODULE 3: EMBEDDING + VECTOR STORE
# =============================================================================

class VectorStoreManager:
    """Manages ChromaDB vector store — ingestion and retrieval."""

    def __init__(self):
        print(f"  [+] Loading embedding model: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = None

    def build_store(self, chunks: List[Document]) -> None:
        """Embed chunks and persist to ChromaDB."""
        print(f"  [+] Embedding {len(chunks)} chunks into ChromaDB...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR
        )
        self.vectorstore.persist()
        print(f"  [+] Vector store built and persisted at {CHROMA_PERSIST_DIR}")

    def load_store(self) -> bool:
        """Load existing ChromaDB store from disk."""
        if os.path.exists(CHROMA_PERSIST_DIR):
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            count = self.vectorstore._collection.count()
            if count > 0:
                print(f"  [+] Loaded existing vector store ({count} chunks)")
                return True
        return False

    def retrieve(self, query: str, k: int = TOP_K) -> Tuple[List[Document], List[float]]:
        """Retrieve top-k chunks with similarity scores."""
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized. Run ingest() first.")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        chunks = [r[0] for r in results]
        # ChromaDB returns L2 distance; convert to similarity score
        scores = [max(0.0, 1.0 - r[1]) for r in results]
        return chunks, scores


# =============================================================================
# MODULE 4: QUERY PROCESSOR (LLM)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful and accurate customer support assistant.
Answer ONLY based on the provided context documents.
Do NOT use prior knowledge or make assumptions beyond the context.
If the context does not contain enough information, respond with exactly:
"INSUFFICIENT_CONTEXT"
Always be concise, friendly, and professional.
At the end of your answer, provide a confidence score as: CONFIDENCE: 0.XX"""

USER_PROMPT_TEMPLATE = """Context Documents:
{context}

Customer Question: {query}

Please provide a helpful answer based only on the above context:"""


class QueryProcessor:
    """Handles LLM call and response parsing."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=api_key
            )
        else:
            print("  [!] No OPENAI_API_KEY found. Using mock LLM for demonstration.")
            self.llm = None

    def generate(self, query: str, chunks: List[Document]) -> Tuple[str, float]:
        """Generate answer from retrieved chunks. Returns (answer, confidence)."""
        if not chunks:
            return "INSUFFICIENT_CONTEXT", 0.0

        # Format context
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("filename", "document")
            page = chunk.metadata.get("page", "?")
            context_parts.append(
                f"[Source {i+1}: {source}, Page {page}]\n{chunk.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        if self.llm is None:
            # Mock response for demonstration
            return self._mock_response(query, chunks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE)
        ])
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": query})
        raw_answer = response.content

        # Parse confidence from response
        confidence = self._extract_confidence(raw_answer)
        answer = self._clean_answer(raw_answer)
        return answer, confidence

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from LLM output."""
        import re
        match = re.search(r"CONFIDENCE:\s*(0?\.\d+)", text)
        if match:
            return float(match.group(1))
        if "INSUFFICIENT_CONTEXT" in text:
            return 0.0
        return 0.6  # Default moderate confidence

    def _clean_answer(self, text: str) -> str:
        """Remove confidence marker from answer."""
        import re
        cleaned = re.sub(r"\nCONFIDENCE:\s*0?\.\d+", "", text).strip()
        return cleaned

    def _mock_response(self, query: str, chunks: List[Document]) -> Tuple[str, float]:
        """Mock LLM response for demo without API key."""
        if any(kw in query.lower() for kw in ["return", "refund", "policy"]):
            return (
                f"Based on the provided documentation, our return policy allows "
                f"customers to return products within 30 days of purchase in original "
                f"condition for a full refund. Please retain your receipt.\n"
                f"Source: {chunks[0].metadata.get('filename', 'policy.pdf')}",
                0.82
            )
        return (
            f"I found relevant information in our knowledge base. "
            f"Based on the documentation: {chunks[0].page_content[:200]}...",
            0.65
        )


# =============================================================================
# LANGGRAPH NODES
# =============================================================================

def input_node(state: RAGState) -> RAGState:
    """Validate and preprocess the user query."""
    query = state["query"].strip()
    # Basic sanitization
    processed = " ".join(query.split())  # normalize whitespace
    state["processed_query"] = processed
    state["start_time"] = time.time()
    state["escalated"] = False
    state["error"] = None
    print(f"\n  [Graph] input_node: Query preprocessed ({len(processed)} chars)")
    return state


def retrieve_node(state: RAGState, vsm: VectorStoreManager) -> RAGState:
    """Retrieve relevant chunks from vector store."""
    # Pre-classification: keyword-based escalation
    query_lower = state["processed_query"].lower()
    for kw in ESCALATION_KEYWORDS:
        if kw in query_lower:
            print(f"  [Graph] retrieve_node: Escalation keyword '{kw}' detected")
            state["retrieved_chunks"] = []
            state["retrieval_scores"] = [0.0]
            state["confidence"] = 0.0
            return state

    try:
        chunks, scores = vsm.retrieve(state["processed_query"])
        state["retrieved_chunks"] = chunks
        state["retrieval_scores"] = scores
        avg_score = sum(scores) / max(len(scores), 1)
        print(f"  [Graph] retrieve_node: Retrieved {len(chunks)} chunks (avg score: {avg_score:.3f})")
    except Exception as e:
        state["retrieved_chunks"] = []
        state["retrieval_scores"] = [0.0]
        state["error"] = str(e)
        print(f"  [Graph] retrieve_node: ERROR - {e}")
    return state


def generate_node(state: RAGState, qp: QueryProcessor) -> RAGState:
    """Generate answer using LLM."""
    if not state["retrieved_chunks"]:
        state["llm_response"] = "INSUFFICIENT_CONTEXT"
        state["confidence"] = 0.0
        return state

    answer, confidence = qp.generate(
        state["processed_query"],
        state["retrieved_chunks"]
    )
    state["llm_response"] = answer
    state["confidence"] = confidence

    # Extract source citations
    sources = list(set([
        f"{c.metadata.get('filename', 'doc')} (p.{c.metadata.get('page', '?')})"
        for c in state["retrieved_chunks"]
    ]))
    state["sources"] = sources
    print(f"  [Graph] generate_node: Generated answer (confidence: {confidence:.2f})")
    return state


def routing_function(state: RAGState) -> Literal["output", "hitl"]:
    """Conditional routing based on confidence and retrieval quality."""
    confidence = state.get("confidence", 0.0)
    scores = state.get("retrieval_scores", [0.0])
    chunks = state.get("retrieved_chunks", [])
    avg_score = sum(scores) / max(len(scores), 1)

    if not chunks:
        print("  [Graph] route_node: → HITL (no chunks retrieved)")
        return "hitl"
    if avg_score < SCORE_THRESHOLD:
        print(f"  [Graph] route_node: → HITL (avg_score {avg_score:.3f} < {SCORE_THRESHOLD})")
        return "hitl"
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"  [Graph] route_node: → HITL (confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD})")
        return "hitl"

    print(f"  [Graph] route_node: → OUTPUT (confidence {confidence:.2f}, score {avg_score:.3f})")
    return "output"


def output_node(state: RAGState) -> RAGState:
    """Format and finalize the response."""
    answer = state["llm_response"]
    sources = state.get("sources", [])

    if sources:
        answer += f"\n\n📄 Sources: {', '.join(sources)}"

    state["final_answer"] = answer
    state["escalated"] = False
    state["latency_ms"] = int((time.time() - state["start_time"]) * 1000)
    print(f"  [Graph] output_node: Response ready (latency: {state['latency_ms']}ms)")
    return state


def hitl_node(state: RAGState) -> RAGState:
    """Human-in-the-Loop escalation handler."""
    print("\n" + "="*60)
    print("  🚨 ESCALATION TO HUMAN AGENT")
    print("="*60)
    print(f"  Query: {state['query']}")
    print(f"  Confidence: {state.get('confidence', 0.0):.2f}")
    print(f"  Escalation ID: {uuid.uuid4().hex[:8].upper()}")

    if state["retrieved_chunks"]:
        print(f"\n  AI Partial Answer: {state.get('llm_response', 'N/A')[:200]}...")

    print("\n  [HITL] Please provide a verified response for this query:")
    print("  (In production: this goes to Slack/dashboard queue)")

    # In production this would be async; for demo, we prompt inline
    human_answer = input("  Agent Response: ").strip()
    if not human_answer:
        human_answer = (
            "Thank you for reaching out. A support ticket has been created and "
            "a human agent will respond within 2 business hours."
        )

    state["human_answer"] = human_answer
    state["final_answer"] = f"🧑‍💼 [Human Agent]: {human_answer}"
    state["escalated"] = True
    state["latency_ms"] = int((time.time() - state["start_time"]) * 1000)
    print(f"  [Graph] hitl_node: Human response captured")
    return state


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_graph(vsm: VectorStoreManager, qp: QueryProcessor):
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(RAGState)

    # Add nodes (wrapping functions that need dependencies)
    graph.add_node("input", input_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, vsm))
    graph.add_node("generate", lambda s: generate_node(s, qp))
    graph.add_node("output", output_node)
    graph.add_node("hitl", hitl_node)

    # Define edges
    graph.set_entry_point("input")
    graph.add_edge("input", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges(
        "generate",
        routing_function,
        {"output": "output", "hitl": "hitl"}
    )
    graph.add_edge("output", END)
    graph.add_edge("hitl", END)

    return graph.compile()


# =============================================================================
# INGESTION PIPELINE
# =============================================================================

def ingest_knowledge_base(pdf_paths: List[str], vsm: VectorStoreManager) -> bool:
    """Full ingestion pipeline: PDF → Chunks → Embeddings → ChromaDB."""
    print("\n" + "="*60)
    print("  KNOWLEDGE BASE INGESTION")
    print("="*60)

    # Step 1: Load documents
    dp = DocumentProcessor()
    docs = dp.load_documents(pdf_paths)
    if not docs:
        print("  [!] No documents loaded. Ingestion failed.")
        return False

    # Step 2: Chunk
    ce = ChunkingEngine()
    chunks = ce.chunk_documents(docs)

    # Step 3: Embed & store
    vsm.build_store(chunks)

    print(f"\n  ✅ Ingestion complete: {len(chunks)} chunks indexed")
    return True


# =============================================================================
# MAIN CLI APPLICATION
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  RAG CUSTOMER SUPPORT ASSISTANT")
    print("  Innomatics Research Labs | Internship Project")
    print("="*60)

    # Initialize components
    print("\n[1/3] Initializing components...")
    vsm = VectorStoreManager()
    qp = QueryProcessor()

    # Load or build knowledge base
    print("\n[2/3] Loading knowledge base...")
    if not vsm.load_store():
        # Demo: create a sample PDF path list
        pdf_paths = []
        data_dir = "./data"
        if os.path.exists(data_dir):
            pdf_paths = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".pdf")
            ]

        if pdf_paths:
            ingest_knowledge_base(pdf_paths, vsm)
        else:
            print("  [!] No PDFs found in ./data/ directory.")
            print("  [!] Please add PDF files to ./data/ and restart.")
            print("  [!] Running in DEMO mode (mock responses).")
            # For demo: create a minimal vector store with sample data
            sample_chunks = [
                Document(
                    page_content="Our return policy allows returns within 30 days of purchase. "
                                 "Items must be in original condition with receipt.",
                    metadata={"filename": "policy.pdf", "page": 1, "chunk_index": 0, "doc_id": "demo001"}
                ),
                Document(
                    page_content="To reset your password, click 'Forgot Password' on the login page. "
                                 "You will receive an email with reset instructions within 5 minutes.",
                    metadata={"filename": "faq.pdf", "page": 2, "chunk_index": 1, "doc_id": "demo002"}
                ),
                Document(
                    page_content="Our customer support hours are Monday to Friday, 9 AM to 6 PM IST. "
                                 "You can reach us at support@company.com or call 1800-XXX-XXXX.",
                    metadata={"filename": "contact.pdf", "page": 1, "chunk_index": 2, "doc_id": "demo003"}
                ),
                Document(
                    page_content="Shipping typically takes 3-5 business days for standard delivery "
                                 "and 1-2 business days for express delivery. Free shipping on orders above Rs. 500.",
                    metadata={"filename": "shipping.pdf", "page": 3, "chunk_index": 3, "doc_id": "demo004"}
                ),
            ]
            vsm.build_store(sample_chunks)

    # Build LangGraph
    print("\n[3/3] Compiling LangGraph workflow...")
    compiled_graph = build_graph(vsm, qp)
    print("  ✅ Graph compiled successfully\n")

    # Interactive loop
    print("="*60)
    print("  Customer Support Assistant Ready!")
    print("  Type 'quit' to exit | Type 'help' for tips")
    print("="*60)

    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! Thank you for using our support assistant.")
            break
        if user_input.lower() == "help":
            print("\nTips:")
            print("  - Ask factual questions about products, policies, or services")
            print("  - Complex or sensitive queries are escalated to human agents")
            print("  - Type 'quit' to exit")
            continue

        # Initialize state
        initial_state: RAGState = {
            "query": user_input,
            "processed_query": "",
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "llm_response": "",
            "confidence": 0.0,
            "final_answer": "",
            "sources": [],
            "escalated": False,
            "human_answer": None,
            "error": None,
            "latency_ms": 0,
            "start_time": time.time()
        }

        print("\n[Processing...]")
        try:
            result = compiled_graph.invoke(initial_state)
            print(f"\nAssistant: {result['final_answer']}")
            print(f"\n[Confidence: {result['confidence']:.2f} | "
                  f"Latency: {result['latency_ms']}ms | "
                  f"Escalated: {result['escalated']}]")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Assistant: I encountered an error. Please try again.")


if __name__ == "__main__":
    main()
