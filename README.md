# Create Nexus - Neural MySQL Table Creation Engine & Enterprise ERP RAG

Dedicated MySQL 8.0 DDL table creation and schema architecture engine powered by a local **Qwen2.5-Coder-0.5B ONNX** model paired with an intelligent **Groq Gateway** and an Enterprise **PostgreSQL Primary Knowledge Base with ChromaDB Vector Engine**.

---

## Architecture Overview

```
Llm_gateway/
├── pyproject.toml           # Poetry package definition & dependency specification
├── Dockerfile               # Production multi-stage container
├── docker-compose.yml       # Orchestrates FastAPI app + PostgreSQL 16
├── .dockerignore            # Docker ignore filters
├── .env                     # Environment credentials & configuration
├── .env.example             # Documented template for environment setup
│
├── src/
│   ├── main.py              # FastAPI application root & middleware
│   ├── config.py            # Global settings, paths, logger, env loader
│   │
│   ├── gateway/             # Groq Gateway & Query Enhancer Subsystem
│   │   ├── api.py           # Endpoints: /api/v1/gateway, /api/v1/routed-generate
│   │   ├── schema.py        # GatewayRequest, GatewayData, StandardGatewayResponse
│   │   └── service.py       # groq_gateway_router, capability analyzer & dispatcher
│   │
│   ├── generator/           # Local DDL Generation Engine
│   │   ├── api.py           # Endpoints: /api/v1/generate, /api/v1/chat
│   │   ├── schema.py        # GenerateRequest, StandardGenerateResponse, GenerateData
│   │   └── service.py       # NexusCreationService (ONNX Causal LM & DDL Sanitizer)
│   │
│   |
│   ├── rag/                 # Enterprise RAG & Knowledge Base Engine
│   │   ├── config.py        # Unified RAG & PostgreSQL connection configuration, prompts & synonyms
│   │   ├── database.py      # SQLAlchemy engine, session management, DAL repository & JSON migrator
│   │   ├── models.py        # Declarative PostgreSQL schema models (UUID PKs)
│   │   ├── schema.py        # Pydantic RAG query and response schemas
│   │   ├── service.py       # Core retrieval, ChromaDB vector engine & embeddings
│   │   └── data/
│   │       ├── domains/     # 5 ERP Domains (105 master tables)
│   │       └── chroma_db/   # Persistent ChromaDB vector store
│   │
│   └── model/               # Model Weights & Runtime
│       └── onnx/
│           ├── model_q4.onnx        # 4-bit quantized ONNX graph
│           └── model_q4.onnx_data   # Quantized model weights (~390 MB)
│
├── utils/
│   ├── domain_priority.py              # Domain & table primary entity priorities
│   └── schema_synonyms.py              # Natural language synonyms & anti-hallucination lists
│
└── tests/
    ├── run_comprehensive_test_matrix.py# 24-scenario enterprise test runner
    └── unit/                # 122 automated unit & regression tests
        ├── test_chromadb_integration.py
        ├── test_creation_matrix.py
        ├── test_eleven_acceptance_criteria.py
        ├── test_postgres_kb_integration.py
        ├── test_rag_regression.py
        └── test_user_spec_verification.py
```

---

## Quickstart

### 1. Environment Setup
Copy the template and configure your parameters:
```bash
cp .env.example .env
```

Ensure your `.env` has:
```env
PORT=8000
HOST=0.0.0.0
GROQ_API_KEY=gsk_your_actual_key_here
DEFAULT_GROQ_MODEL=openai/gpt-oss-120b

# PostgreSQL Primary Knowledge Base
DB_USER=postgres
DB_PASSWORD=your_postgres_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=db_nexus_rag
DATABASE_URL=postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
USE_POSTGRES_KB=true

# ChromaDB Vector Store
CHROMA_DATA_FOLDER=chroma_db
CHROMA_COLLECTION_NAME=nexus_erp_knowledge
CHROMA_TOP_K=5
```

---

### 2. Run with Poetry

Install dependencies:
```bash
poetry install
```

Start the application:
```bash
poetry run python -m src.main
```
Or directly with uvicorn:
```bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 3. Run with Docker & Docker Compose

The Docker setup seamlessly connects to your PostgreSQL database using the `DATABASE_URL` specified in `.env`:
```bash
docker compose up -d --build
```

To view logs:
```bash
docker compose logs -f app
```

---

### 4. Running the Test Suite

Run all **122 unit and integration tests**:
```bash
poetry run python -m unittest discover -s tests -p "test_*.py"
```

Run the **24-scenario enterprise benchmark matrix**:
```bash
poetry run python tests/run_comprehensive_test_matrix.py
```

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| **`/api/v1/generate`** | `POST` | Core Table Creation & ERP RAG schema generation pipeline. |
| **`/api/v1/gateway`** | `POST` | Groq Gateway decision router & prompt enhancement. |
| **`/api/v1/routed-generate`** | `POST` | Hybrid routing across local SLM, RAG, and Groq. |
| **`/health_check`** | `GET` | Service health status probe. |
