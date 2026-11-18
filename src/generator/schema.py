from typing import Optional
from pydantic import BaseModel, Field, model_validator


class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="User prompt describing database tables or fields")
    user_query: Optional[str] = Field(None, description="Alternative field for user prompt")

    max_tokens: Optional[int] = Field(None, description="Max new tokens to generate")
    max_token: Optional[int] = Field(None, description="Alternative field for max_tokens")

    model: Optional[str] = Field(
        default="llama-3.3-70b-versatile",
        description="Model name"
    )

    summary: Optional[str] = Field(default="", description="Existing schema summary from previous turns")
    SQL: Optional[str] = Field(default=None, description="Active SQL schema from previous turns")
    sql: Optional[str] = Field(default=None, description="Alternative field for active SQL schema")
    sql_query: Optional[str] = Field(default=None, description="Alternative field for active SQL schema")
    dbml: Optional[str] = Field(default=None, description="Legacy schema field")
    enable_summary: bool = Field(default=True, description="Whether summary tracking is enabled")
    session_id: Optional[str] = Field(default=None, description="Session ID for multi-turn context retention")
    api_key: Optional[str] = Field(default=None, description="Optional Groq API Key")
    groq_api_key: Optional[str] = Field(default=None, description="Alias for api_key")

    @model_validator(mode="before")
    @classmethod
    def resolve_aliases(cls, values: dict):
        if not isinstance(values, dict):
            return values

        # Resolve api_key
        ak = values.get("api_key") or values.get("groq_api_key") or values.get("apiKey")
        values["api_key"] = ak
        values["groq_api_key"] = ak

        # Resolve prompt
        p = (
            values.get("prompt")
            or values.get("user_query")
            or values.get("query")
            or values.get("requirement")
            or values.get("message")
            or values.get("text")
            or values.get("input")
            or ""
        )
        values["prompt"] = p
        values["user_query"] = p

        # Resolve summary input
        s = (
            values.get("summary")
            or values.get("previous_summary")
            or values.get("schema_summary")
            or ""
        )
        values["summary"] = s

        # Resolve SQL / schema input
        active_sql = (
            values.get("SQL")
            or values.get("sql")
            or values.get("sql_query")
            or values.get("dbml")
            or values.get("schema")
            or ""
        )
        values["SQL"] = active_sql
        values["sql"] = active_sql
        values["sql_query"] = active_sql
        values["dbml"] = active_sql

        # Resolve max_tokens
        if values.get("max_tokens") is None and values.get("max_token") is not None:
            values["max_tokens"] = values.get("max_token")
        elif values.get("max_tokens") is None:
            values["max_tokens"] = 550

        if values["max_tokens"] > 2048:
            values["max_tokens"] = 2048
        elif values["max_tokens"] < 20:
            values["max_tokens"] = 150

        # Resolve session_id
        values["session_id"] = (
            values.get("session_id")
            or values.get("sessionId")
            or values.get("conversation_id")
            or None
        )

        return values


class GenerateData(BaseModel):
    intent: Optional[str] = Field(default="RELATED", description="Detected intent: SCHEMA or RELATED")
    operation: Optional[str] = Field(default="CREATE_TABLE", description="Code-level operation type: CREATE_TABLE, ALTER_TABLE, RENAME_TABLE, DROP_TABLE, CLARIFICATION, TECHNICAL_QA, DML_RESTRICTED, GENERAL_CHAT")
    sql: str = Field(default="", description="Generated MySQL 8.0 CREATE TABLE statements")
    updated_summary: str = Field(default="", description="Updated schema ledger summary")
    explanation: str = Field(default="", description="Step-by-step schema explanation")

    # Fields hidden/excluded from client response JSON (retained internally in Python code):
    debug: dict = Field(default_factory=dict, exclude=True, description="Debugging trace and structured query plan")
    columns_metadata: list[dict] = Field(default_factory=list, exclude=True, description="Granular structured per-column metadata")
    warnings: list[str] = Field(default_factory=list, exclude=True, description="Validation or relationship warnings")
    needs_clarification: bool = Field(default=False, exclude=True, description="Whether clarification is required")
    suggestions: list[str] = Field(default_factory=list, exclude=True, description="Suggestions for ambiguous entities")
    sql_query: Optional[str] = Field(default="", exclude=True, description="Internal alias for sql")
    dbml: Optional[str] = Field(default="", exclude=True, description="Internal DBML schema representation")
    schema_plan: list[dict] = Field(default_factory=list, exclude=True, description="Structured column definitions and relationships array (hidden to prevent response bloat)")
    renamed_tables: list[dict] = Field(default_factory=list, exclude=True, description="Lineage mapping for renamed tables")
    dropped_tables: list[str] = Field(default_factory=list, exclude=True, description="List of dropped table names")
    altered_columns: list[dict] = Field(default_factory=list, exclude=True, description="List of altered column changes")
    token_used: int = Field(default=0, exclude=True, description="Tokens used for generation")


class StandardGenerateResponse(BaseModel):
    message: str = Field(default="Query processed successfully")
    status_code: int = Field(default=200)
    status: Optional[str] = Field(default="accepted", exclude=True, description="Status string: accepted, rejected, clarification_required")
    data: GenerateData


class GatewayRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="User prompt to enhance and classify")
    user_query: Optional[str] = Field(None, description="Alias for prompt")
    api_key: Optional[str] = Field(None, description="Optional Groq API Key")

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
        return values


class GatewayResponse(BaseModel):
    message: Optional[str] = Field(None, description="Direct result if greetings/general_qa/out_of_scope, or null for DDL")
    operation: str = Field(..., description="Detected operation: greetings, create, alter, drop, general_qa, out_of_scope")
    model_name: str = Field(default="llama-3.3-70b-versatile", description="Model name that processed this gateway turn")
    token_used: int = Field(default=0, description="Token count for this gateway turn only")
    enhanced_prompt: str = Field(..., description="Typo-fixed, grammatically cleaned user prompt")


