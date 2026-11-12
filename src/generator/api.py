import json
import re
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
        target_model = gw_decision.get("suggested_model") or gw_decision.get("recommended_engine") or gw_decision.get("model_name", "create-nexus-slm")
        enhanced = gw_decision.get("enhanced_prompt") or gw_decision.get("refined_query") or payload.prompt
        gw_tokens = gw_decision.get("token_used") or gw_decision.get("gateway_tokens", 0)

        logger.info(f"Gateway routed turn: op='{op_gw}' | target_model='{target_model}' | enhanced='{enhanced}'")

        # ---------------------------------------------------------------------
        # Step 2: Route Based on Decision (Summary is supplied to the chosen model)
        # ---------------------------------------------------------------------
        # Case A: Questions / General QA / Out-of-scope -> Groq model
        if target_model == "groq" or op_gw in ("general_qa", "out_of_scope"):
            logger.info(f"Routing question directly to Groq LLM: '{enhanced}' with context summary")
            user_msg = f"Database Context:\n{active_summary}\n\nQuestion:\n{enhanced}" if active_summary.strip() else enhanced
            groq_res = query_groq_llm(prompt=user_msg, api_key=payload.api_key)
            answer_text = groq_res.get("content", "Question answered.")
            total_tokens = gw_tokens + groq_res.get("token_used", 0)

            gen_data = GenerateData(
                intent="RELATED",
                operation="TECHNICAL_QA",
                sql="",
                sql_query="",
                dbml="",
                schema_plan=[],
                renamed_tables=[],
                dropped_tables=[],
                altered_columns=[],
                updated_summary=active_summary,
                explanation=answer_text,
                warnings=[],
                needs_clarification=False,
                token_used=total_tokens,
            )
            response = StandardGenerateResponse(
                message="Technical explanation generated",
                status_code=200,
                status="accepted",
                data=gen_data
            )
            return response

        # Case B: Greetings OR DDL (Create, Alter, Drop) -> Our Local Model (create-nexus-slm)
        logger.info(f"Routing '{op_gw}' to local model (create-nexus-slm) with enhanced_prompt='{enhanced}'")
        service = get_creation_service()
        contract_dict, slm_tokens = service.generate(
            user_query=enhanced,
            summary=active_summary,
            dbml=active_sql,
            enable_summary=payload.enable_summary,
            max_tokens=payload.max_tokens,
        )

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

