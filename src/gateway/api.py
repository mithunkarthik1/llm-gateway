import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import get_logger
from src.gateway.schema import (
    GatewayRequest,
    GatewayData,
    StandardGatewayResponse
)
from src.gateway.service import groq_gateway_router

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Gateway"])


@router.post("/gateway", response_model=StandardGatewayResponse)
def gateway_endpoint(payload: GatewayRequest):
    """
    Initial Groq Gateway:
    - Enhances user prompt (fixes typos, grammar, jumbled words)
    - Detects operation: create, alter, drop, greetings, general_qa, out_of_scope
    - Assigns preferred_model:
        * 'create-nexus-slm' for greetings and 1-5 tables schema operations
        * 'groq' for technical questions (general_qa) and 6+ tables/enterprise operations
    - Makes decision only (message remains null)
    - Returns standardized envelope with status_code 200
    """
    logger.info(f">>> [GATEWAY REQUEST]: prompt='{payload.prompt}'")
    try:
        res = groq_gateway_router(
            user_prompt=payload.prompt,
            api_key=payload.api_key,
            context=None
        )

        gw_data = GatewayData(
            intent=res.get("intent", "general_qa"),
            suggested_model=res.get("suggested_model", "create-nexus-slm"),
            token_used=res.get("token_used", 0),
            enhanced_prompt=res.get("enhanced_prompt", payload.prompt or "")
        )

        response = StandardGatewayResponse(
            message="decision made successfully",
            status_code=200,
            data=gw_data
        )

        logger.info(f"<<< [GATEWAY RESPONSE]: intent='{gw_data.intent}' | suggested_model='{gw_data.suggested_model}' | tokens={gw_data.token_used}")
        return response

    except Exception as e:
        logger.error(f"Error in gateway_endpoint: {e}", exc_info=True)
        fallback_data = GatewayData(
            intent="create",
            suggested_model="create-nexus-slm",
            token_used=0,
            enhanced_prompt=payload.prompt or ""
        )
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Gateway error: {str(e)}",
                "status_code": 500,
                "data": fallback_data.model_dump()
            }
        )
