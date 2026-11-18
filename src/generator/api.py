import os
import json
import re
from typing import Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.config import get_logger
from src.generator.schema import (
    GenerateRequest,
    StandardGenerateResponse,
    GenerateData,
)
from src.generator.service import get_creation_service
from src.gateway.service import groq_gateway_router, query_groq_llm

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Creation"])

SESSION_CACHE: dict[str, dict] = {}


def classify_rag_operation_and_intent(query: str, sql: str, rag_result: Any) -> tuple[str, str]:
    q_low = (query or "").lower().strip()
    clean_sql = (sql or "").strip()

    # 1. Clarification needed
    if (
        getattr(rag_result, "domain", "") == "ambiguous"
        or getattr(rag_result, "needs_clarification", False)
        or getattr(rag_result, "status", "") == "needs_clarification"
    ):
        return "CLARIFICATION", "SCHEMA"

    # 2. General conversational chat / greetings
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings", "how are you", "who are you", "thanks", "thank you", "bye"}
    if q_low in greetings or q_low.rstrip("!.? ") in greetings:
        return "GENERAL_CHAT", "RELATED"

    # 3. DML operations
    if re.search(r"\b(?:insert\s+into|update\s+[a-zA-Z0-9_]+\s+set|delete\s+from)\b", q_low):
        return "DML_RESTRICTED", "RELATED"

    # 4. RENAME table
    if re.search(r"\b(?:rename\s+table|rename)\b", q_low):
        return "RENAME_TABLE", "SCHEMA"

    # 5. DROP table
    if re.search(r"\b(?:drop\s+table|delete\s+table|remove\s+table)\b", q_low):
        return "DROP_TABLE", "SCHEMA"

    # 6. ALTER table / column
    if re.search(r"\b(?:alter\s+table|add\s+column|drop\s+column|modify\s+column|change\s+column|rename\s+column)\b", q_low):
        return "ALTER_TABLE", "SCHEMA"

    # 7. Entity not found or rejected
    if getattr(rag_result, "domain", "") == "not_found" or getattr(rag_result, "status", "") == "rejected" or not clean_sql:
        return "TECHNICAL_QA", "SCHEMA"

    # 8. Check question vs creation intent
    is_pure_question = bool(re.search(r"\b(?:which\s+column|what\s+column|what\s+is|explain\s+what|why\s+is|how\s+does|how\s+do|tell\s+me\s+about)\b", q_low))
    has_create_intent = bool(re.search(r"\b(?:create|generate|make|build|setup|set\s+up|construct|define|add|schemas?\s+for|tables?\s+for|database\s+for|db\s+for)\b", q_low))

    if is_pure_question and not has_create_intent:
        return "TECHNICAL_QA", "SCHEMA"

    if has_create_intent or "CREATE TABLE" in clean_sql.upper():
        return "CREATE_TABLE", "SCHEMA"

    return "CREATE_TABLE" if "CREATE TABLE" in clean_sql.upper() else "TECHNICAL_QA", "SCHEMA"


def extract_pure_sql_query(sql_text: str) -> str:
    """
    Extracts only executable SQL DDL statements (e.g. CREATE TABLE statements),
    stripping commentary headers and non-SQL metadata lines.
    """
    if not sql_text or not sql_text.strip():
        return ""
    matches = re.findall(r"((?:CREATE|ALTER|DROP|RENAME)\s+TABLE\s+[^;]+;)", sql_text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    lines = [l for l in sql_text.splitlines() if not l.strip().startswith("--")]
    return "\n".join(lines).strip()


@router.post("/chat", response_model=StandardGenerateResponse)
def generate_chat(payload: GenerateRequest):
    return generate_table_endpoint(payload)


@router.post("/generate", response_model=StandardGenerateResponse)
def generate_table_endpoint(payload: GenerateRequest):
    try:
        # Log incoming request payload
        payload_dict = payload.model_dump(exclude_none=True)
        payload_json = json.dumps(payload_dict, indent=2, ensure_ascii=True)
        print("\n" + "=" * 80, flush=True)
        print(">>> [INCOMING CREATE_NEXUS REQUEST PAYLOAD]:", flush=True)
        print("=" * 80, flush=True)
        print(payload_json, flush=True)
        print("=" * 80, flush=True)
        logger.info(f"Incoming payload:\n{payload_json}")

        active_summary = payload.summary or ""
        active_sql = payload.SQL or payload.dbml or ""
        if payload.session_id and payload.session_id in SESSION_CACHE:
            sess = SESSION_CACHE[payload.session_id]
            if not active_summary:
                active_summary = sess.get("summary", "")
            if not active_sql:
                active_sql = sess.get("sql", "")

        # ---------------------------------------------------------------------
        # Step 1: Initial Gateway Decision & Prompt Enhancement (Approach 1: Prompt-Only)
        # We pass only payload.prompt (no summary) to minimize token consumption and maximize speed
        # ---------------------------------------------------------------------
        gw_decision = groq_gateway_router(
            user_prompt=payload.prompt,
            api_key=payload.api_key,
            context=None
        )
        op_gw = (gw_decision.get("intent") or gw_decision.get("intent_type") or gw_decision.get("operation", "create")).lower()
        target_model = gw_decision.get("suggested_model") or gw_decision.get("recommended_engine") or gw_decision.get("model_name", "rag-enterprise")
        enhanced = gw_decision.get("enhanced_prompt") or gw_decision.get("refined_query") or payload.prompt
        gw_tokens = gw_decision.get("token_used") or gw_decision.get("gateway_tokens", 0)

        logger.info(f"Gateway routed turn: op='{op_gw}' | target_model='{target_model}' | enhanced='{enhanced}'")

        # ---------------------------------------------------------------------
        # Step 2: Route Based on Decision (RAG-Only Mode: All Queries Route to RAG)
        # ---------------------------------------------------------------------
        rag_only_mode = os.getenv("RAG_ONLY_MODE", "true").lower() in ("1", "true", "yes")

        # Enterprise / ERP RAG Schema Pipeline
        if target_model in ("rag-model", "rag-enterprise") or rag_only_mode:
            logger.info(f"Routing query exclusively to Enterprise RAG Schema Pipeline: '{enhanced}' (SLM & Groq stopped)")
            from src.rag import retrieve_erp_schema
            sess_tbl = None
            if payload.session_id and payload.session_id in SESSION_CACHE:
                sess_sql = SESSION_CACHE[payload.session_id].get("sql", "")
                m_tbl = re.search(r"CREATE TABLE ([a-zA-Z0-9_]+)", sess_sql, re.IGNORECASE) or re.search(r"-- Table: ([a-zA-Z0-9_]+)", sess_sql, re.IGNORECASE)
                if m_tbl:
                    sess_tbl = m_tbl.group(1)

            rag_result = retrieve_erp_schema(enhanced, session_context=sess_tbl, summary=active_summary)
            rag_sql = rag_result.sql
            rag_summary = rag_result.summary
            explanation = rag_result.explanation
            total_tokens = 0

            # Only call Groq LLM if RAG_ONLY_MODE is disabled and query needs custom reasoning
            if not rag_only_mode:
                needs_llm_reasoning = (
                    op_gw in ("general_qa", "alter", "modify", "custom")
                    or any(w in enhanced.lower() for w in ["explain", "why", "how", "alter", "modify", "custom", "add "])
                )
                if needs_llm_reasoning:
                    from src.rag import generate_rag_llm
                    logger.info(f"Invoking optimized RAG LLM for query: '{enhanced}'")
                    llm_res = generate_rag_llm(query=enhanced, api_key=payload.api_key)
                    if llm_res.get("explanation"):
                        explanation = llm_res["explanation"]
                        total_tokens += llm_res.get("token_used", 0)

            # Intent & Operation Classification for read-only schema queries
            if rag_result.domain == "ambiguous" or getattr(rag_result, "needs_clarification", False) or getattr(rag_result, "status", "") == "needs_clarification":
                status_str = "needs_clarification"
                msg_str = "Clarification needed"
                clean_sql = ""
            elif rag_result.domain == "not_found" or getattr(rag_result, "status", "") == "rejected" or (not rag_result.tables and not rag_sql.strip()):
                status_str = "rejected"
                msg_str = "Schema not found"
                clean_sql = ""
            elif getattr(rag_result, "status", "") == "partial":
                status_str = "partial"
                msg_str = "Schema retrieved with warnings"
                clean_sql = extract_pure_sql_query(rag_sql)
            else:
                status_str = getattr(rag_result, "status", "accepted")
                msg_str = "Schema retrieved successfully"
                clean_sql = extract_pure_sql_query(rag_sql)

            op_str, intent_str = classify_rag_operation_and_intent(
                query=enhanced,
                sql=clean_sql,
                rag_result=rag_result
            )

            gen_data = GenerateData(
                intent=intent_str,
                operation=op_str,
                sql=clean_sql,
                sql_query=clean_sql,
                dbml="",
                schema_plan=[],
                renamed_tables=[],
                dropped_tables=[],
                altered_columns=[],
                updated_summary=rag_summary,
                explanation=explanation,
                warnings=getattr(rag_result, "warnings", []),
                needs_clarification=(status_str == "needs_clarification"),
                suggestions=getattr(rag_result, "suggestions", []),
                debug=getattr(rag_result, "debug", {}) or {},
                columns_metadata=getattr(rag_result, "columns_metadata", []) or [],
                token_used=total_tokens,
            )

            if payload.session_id:
                SESSION_CACHE[payload.session_id] = {
                    "summary": rag_summary,
                    "sql": clean_sql,
                }

            response = StandardGenerateResponse(
                message=msg_str,
                status_code=200,
                status=status_str,
                data=gen_data
            )
            return response

        op = contract_dict.get("operation", "CREATE_TABLE")
        sql_out = contract_dict.get("sql_query", "")
        if op == "RENAME_TABLE":
            resp_msg = "Table renamed successfully"
        elif op == "DROP_TABLE":
            resp_msg = "Table dropped successfully"
        elif op == "ALTER_TABLE":
            resp_msg = "Table altered successfully"
        elif op == "GENERAL_CHAT" or op_gw == "greetings":
            resp_msg = "Conversational response generated"
        elif op == "TECHNICAL_QA":
            resp_msg = "Technical explanation generated"
        elif op == "DML_RESTRICTED":
            resp_msg = "DML operations are restricted"
        elif op == "CLARIFICATION":
            resp_msg = "Clarification needed"
        elif sql_out.strip():
            resp_msg = "Table(s) created successfully"
        elif contract_dict.get("intent") == "RELATED":
            resp_msg = "Conversational response generated" if op_gw == "greetings" else "Clarification needed"
        else:
            resp_msg = "Request processed successfully"

        gen_data = GenerateData(
            intent=contract_dict.get("intent", "SCHEMA" if sql_out.strip() else "RELATED"),
            operation=op,
            sql_query=sql_out,
            sql=sql_out,
            dbml=contract_dict.get("dbml", ""),
            schema_plan=contract_dict.get("schema_plan", []),
            renamed_tables=contract_dict.get("renamed_tables", []),
            dropped_tables=contract_dict.get("dropped_tables", []),
            altered_columns=contract_dict.get("altered_columns", []),
            updated_summary=contract_dict.get("updated_summary", ""),
            explanation=contract_dict.get("explanation", ""),
            warnings=contract_dict.get("warnings", []),
            needs_clarification=contract_dict.get("needs_clarification", False),
            token_used=gw_tokens + slm_tokens,
        )

        status_str = (
            "clarification_required"
            if gen_data.needs_clarification
            else (
                "accepted"
                if sql_out.strip() or op in ["DROP_TABLE", "RENAME_TABLE", "TECHNICAL_QA", "GENERAL_CHAT"]
                else ("rejected" if op == "DML_RESTRICTED" else "accepted")
            )
        )

        response = StandardGenerateResponse(
            message=resp_msg,
            status_code=200,
            status=status_str,
            data=gen_data,
        )

        if payload.session_id:
            prev_sess_sql = SESSION_CACHE.get(payload.session_id, {}).get("sql", "")
            cum_sql = gen_data.sql_query
            if prev_sess_sql:
                if op == "DROP_TABLE" and gen_data.dropped_tables:
                    dt_set = {d.lower() for d in gen_data.dropped_tables}
                    p_stmts = [s.strip() for s in prev_sess_sql.split(";\n\n") if not any(f"CREATE TABLE {d}" in s.upper() for d in dt_set)]
                    cum_sql = ";\n\n".join(p_stmts)
                elif op == "ALTER_TABLE" and gen_data.sql_query:
                    m_alt = re.search(r"CREATE\s+TABLE\s+([a-zA-Z0-9_`\"\[\]]+)", gen_data.sql_query, re.I)
                    if m_alt:
                        alt_name = m_alt.group(1).lower()
                        p_stmts = [s.strip() for s in prev_sess_sql.split(";\n\n") if not re.search(r"CREATE\s+TABLE\s+" + re.escape(alt_name) + r"\b", s, re.I)]
                        p_stmts.append(gen_data.sql_query.strip(";"))
                        cum_sql = ";\n\n".join(p_stmts) + ";"
                elif gen_data.sql_query:
                    cum_sql = prev_sess_sql.strip() + "\n\n" + gen_data.sql_query.strip()
                else:
                    cum_sql = prev_sess_sql
            SESSION_CACHE[payload.session_id] = {
                "summary": gen_data.updated_summary or active_summary,
                "sql": cum_sql
            }

        # Log outgoing response
        resp_dict = response.model_dump()
        resp_json = json.dumps(resp_dict, indent=2, ensure_ascii=True)
        print("\n" + "=" * 80, flush=True)
        print("<<< [OUTGOING CREATE_NEXUS RESPONSE]:", flush=True)
        print("=" * 80, flush=True)
        print(resp_json, flush=True)
        print("=" * 80 + "\n", flush=True)
        logger.info(f"Outgoing response:\n{resp_json}")

        return response

    except Exception as e:
        logger.error(f"Error in generate_table_endpoint: {e}", exc_info=True)
        err_data = GenerateData(
            intent="ERROR",
            operation="ERROR",
            sql="",
            sql_query="",
            updated_summary="",
            explanation=str(e),
            warnings=[str(e)],
            needs_clarification=False,
            token_used=0,
        )
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Generation error: {str(e)}",
                "status_code": 500,
                "status": "error",
                "data": err_data.model_dump()
            }
        )

