from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, model_validator


class GatewayRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="User prompt to enhance and classify")
    user_query: Optional[str] = Field(None, description="Alias for prompt")
    api_key: Optional[str] = Field(None, description="Optional Groq API Key")
    groq_api_key: Optional[str] = Field(None, description="Alias for api_key")
    context: Optional[str] = Field(None, description="Active database context / summary")
    summary: Optional[str] = Field(None, description="Alias for context")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn context retention")
    enable_summary: bool = Field(default=True, description="Whether summary tracking is enabled")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for local SLM generation")

    @model_validator(mode="before")
    @classmethod
    def resolve_aliases(cls, values: dict):
        if not isinstance(values, dict):
            return values
        p = values.get("prompt") or values.get("user_query") or values.get("message") or ""
        values["prompt"] = p
        values["user_query"] = p
        ak = values.get("api_key") or values.get("groq_api_key") or values.get("apiKey")
        values["api_key"] = ak
        values["groq_api_key"] = ak
        c = values.get("context") or values.get("summary") or ""
        values["context"] = c
        values["summary"] = c
        sid = values.get("session_id") or values.get("sessionId") or values.get("conversation_id")
        values["session_id"] = sid
        return values


class GatewayData(BaseModel):
    intent: str = Field(..., description="Classified intent: create, alter, drop, greetings, general_qa, out_of_scope")
    suggested_model: str = Field(default="create-nexus-slm", description="Designated execution engine: create-nexus-slm or groq")
    token_used: int = Field(default=0, description="Token consumption for gateway decision turn")
    enhanced_prompt: str = Field(..., description="Grammar-fixed, typo-corrected user prompt")

    class Config:
        populate_by_name = True
        by_alias = False


class StandardGatewayResponse(BaseModel):
    message: str = Field(default="decision made successfully", description="Status message")
    status_code: int = Field(default=200, description="HTTP status code")
    data: GatewayData


# Backward compatibility alias
GatewayResponse = StandardGatewayResponse
