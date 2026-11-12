# Create Nexus - Neural MySQL Table Creation Engine

Dedicated MySQL 8.0 DDL table creation and schema architecture engine powered by a local **Qwen2.5-Coder-0.5B ONNX** model paired with an intelligent **Groq Gateway** for prompt enhancement and dynamic capability-based routing.

---

## Architecture Overview

```
create_nexus_11/
├── src/
│   ├── main.py              # FastAPI application root & middleware
│   ├── config.py            # Global settings, paths, logger, env loader
│   ├── api.py               # Backward-compatibility re-exports
│   ├── service.py           # Backward-compatibility re-exports
│   ├── schema.py            # Backward-compatibility re-exports
│   │
│   ├── gateway/             # Groq Gateway & Query Enhancer Subsystem
│   │   ├── __init__.py
│   │   ├── api.py           # Endpoints: /api/v1/gateway, /api/v1/routed-generate
│   │   ├── schema.py        # GatewayRequest, GatewayData, StandardGatewayResponse
│   │   └── service.py       # groq_gateway_router, capability analyzer & dispatcher
│   │
│   ├── generator/           # Local DDL Generation Engine
│   │   ├── __init__.py
│   │   ├── api.py           # Endpoints: /api/v1/generate, /api/v1/chat
│   │   ├── schema.py        # GenerateRequest, StandardGenerateResponse, GenerateData
│   │   └── service.py       # NexusCreationService (ONNX Causal LM & DDL Sanitizer)
│   │
│   └── model/               # Model Weights & Runtime
│       └── onnx/
│           ├── model_q4.onnx        # 4-bit quantized ONNX graph
│           └── model_q4.onnx_data   # Quantized model weights (~390 MB)
│
├── tests/                   # Integration and sanity tests
│   ├── test_routed_flow.py
│   └── quick_test_fixes.py
│
├── sample_tests/            # 21-Level Standardized Benchmark Suite
│   ├── run_21_level_benchmark.py
│   ├── benchmark_21_levels_cases.json
│   ├── scoring_engine.py
│   ├── sql_evaluator.py
│   └── results/
│
├── run_server.py            # Server startup CLI
├── .env                     # Environment credentials
└── requirements.txt         # Dependencies
```

---

## API Endpoints

| Endpoint | Method | Response Contract | Description |
| :--- | :---: | :--- | :--- |
| **`/api/v1/gateway`** | `POST` | `StandardGatewayResponse` | **Gateway Decision Only**: Cleans typos/grammar, determines `model_name` (`create-nexus-slm` vs `groq`), handles greetings/QA directly, and tracks only gateway tokens. |
| **`/api/v1/routed-generate`** | `POST` | **Native Model Response** | **End-to-End Hybrid Flow**: Groq enhances the query. If within SLM capability (≤ 3 tables), local model executes and returns native output. If over capability (4+ tables), Groq generates the full schema. |
| **`/api/v1/generate`** | `POST` | `StandardGenerateResponse` | **Direct Local Model**: Bypasses the gateway and queries the ONNX model directly. |
| **`/api/v1/chat`** | `POST` | `StandardGenerateResponse` | **Multi-Turn Session**: Retains declarative schema state across turns via `session_id`. |
| **`/health_check`** | `GET` | `{"status": "healthy"}` | Health status probe. |

---

## Quickstart

### 1. Environment Setup
Configure your Groq API key in `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
DEFAULT_GROQ_MODEL=openai/gpt-oss-120b
HOST=0.0.0.0
PORT=8000
```

### 2. Start the Server
```bash
python run_server.py
```
Or with reload:
```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Run Verification Tests
```bash
python tests/test_routed_flow.py
```

### 4. Run the 21-Level Benchmark
```bash
python sample_tests/run_21_level_benchmark.py --quick-sample
```

