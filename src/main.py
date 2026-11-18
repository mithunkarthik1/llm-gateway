from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.config import get_logger
from src.generator.api import router as generator_router
from src.gateway.api import router as gateway_router

logger = get_logger(__name__)

app = FastAPI(
    title="Create Nexus - Neural MySQL Table Creation Engine",
    version="1.0.0",
    description="Dedicated MySQL 8.0 DDL table creation service powered by Qwen2.5-Coder-0.5B ONNX and Groq Gateway."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health_check")
def health_check():
    try:
        logger.info("Health check endpoint requested")
        return {
            "message": "Create Nexus table creation service is running",
            "status_code": 200,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error during health check: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Health check failed: {str(e)}",
                "status_code": 500,
                "status": "unhealthy"
            }
        )

app.include_router(generator_router)
app.include_router(gateway_router)

if __name__ == "__main__":
    import uvicorn
    from src.config import HOST, PORT
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=True)
