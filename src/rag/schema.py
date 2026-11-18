from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class TableSchema(BaseModel):
    table_name: str = Field(..., description="Unique name of the relational database table")
    category: Optional[str] = Field(None, description="Module or functional domain category")
    description: Optional[str] = Field(None, description="Detailed explanation of the table purpose")
    dependencies: List[str] = Field(default_factory=list, description="Parent tables required by foreign keys")
    sql: str = Field(..., description="DDL statement to create the table and its constraints")
    columns: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional column specifications")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query requesting ERP schema tables")
    domain: Optional[str] = Field(None, description="Specific ERP domain ID (e.g. erp_finance, erp_supply_chain)")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversational context")

    @model_validator(mode="before")
    @classmethod
    def resolve_aliases(cls, values: dict):
        if not isinstance(values, dict):
            return values
        q = values.get("query") or values.get("prompt") or values.get("user_query") or ""
        values["query"] = q
        return values


class RAGSchemaResponse(BaseModel):
    domain: str = Field(..., description="Domain identifier (e.g., erp_finance)")
    domain_name: str = Field(..., description="Human-readable domain title")
    tables: List[str] = Field(default_factory=list, description="Names of all retrieved tables in dependency order")
    sql: str = Field(..., description="Combined valid DDL statements for the retrieved tables")
    count: int = Field(..., description="Total count of tables retrieved")
    summary: str = Field(..., description="Dynamically generated schema summary ledger matching retrieved tables")
    explanation: str = Field(..., description="Brief architectural explanation of the retrieved schema")
    warnings: List[str] = Field(default_factory=list, description="List of warnings such as missing columns or excluded tables")
    structured_metadata: Optional[List[Dict[str, Any]]] = Field(default=None, description="Granular structured JSON metadata for tables and columns")
    columns_metadata: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Granular structured per-column metadata")
    suggestions: List[str] = Field(default_factory=list, description="List of suggestions for ambiguous requests")
    debug: Optional[Dict[str, Any]] = Field(default=None, description="Debugging trace and structured query plan")
    status: str = Field(default="accepted", description="Status outcome: accepted, partial, needs_clarification, rejected")
    intent: str = Field(default="SCHEMA", description="Detected intent: SCHEMA or RELATED")
    needs_clarification: bool = Field(default=False, description="Whether clarification is required")

    def __getitem__(self, item: str):
        return getattr(self, item)

    def get(self, item: str, default: Any = None):
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
