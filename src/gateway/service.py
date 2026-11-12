import os
import json
import logging
from typing import Any, Dict, Optional
import requests

from src.config import get_logger, GROQ_API_KEY, DEFAULT_GROQ_MODEL, normalize_groq_model

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GATEWAY_SYSTEM_PROMPT = """Route db request:
- suggested_model: "create-nexus-slm" (greetings, 1-5 tables schema) or "groq" (questions, 6+ tables).
- intent: "create"|"alter"|"drop"|"greetings"|"general_qa"|"out_of_scope".
- enhanced_prompt: clean user prompt.
JSON only: {"enhanced_prompt":"...","intent":"...","suggested_model":"..."}"""


def groq_gateway_router(
    user_prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[str] = None,
    timeout: int = 15
) -> Dict[str, Any]:
    """
    Initial Gateway step:
    - Fast-path for simple greetings (0 tokens, instant)
    - Analyzes user prompt using Groq with streamlined prompt and max_tokens constraint
    - Detects operation and assigns preferred_model
    - Tracks gateway token count only
    """
    cleaned_prompt = (user_prompt or "").strip()

    # Fast-path optimization: simple greetings consume 0 tokens and return instantly
    common_greetings = {"hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings"}
    if cleaned_prompt.lower().rstrip("!.? ") in common_greetings:
        logger.info(f"Fast-path greeting matched for '{cleaned_prompt}' -> 0 tokens.")
        return {
            "intent": "greetings",
            "suggested_model": "create-nexus-slm",
            "token_used": 0,
            "enhanced_prompt": cleaned_prompt.capitalize()
        }

    model_name = normalize_groq_model(model or DEFAULT_GROQ_MODEL)
    key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

    if not key:
        logger.warning("GROQ_API_KEY is not set. Gateway falling back to raw prompt passthrough.")
        return {
            "message": None,
            "operation": "create",
            "model_name": "create-nexus-slm",
            "token_used": 0,
            "enhanced_prompt": user_prompt
        }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    user_content = (
        f"Existing Database Context:\n{context}\n\nUser Query:\n{user_prompt}"
        if (context and context.strip())
        else user_prompt
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": GATEWAY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 300
    }

    try:
        logger.info(f"Sending prompt to Groq Gateway model '{model_name}'...")
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        
        if resp.status_code != 200:
            logger.error(f"Groq API returned HTTP {resp.status_code}: {resp.text}")
            return {
                "message": None,
                "operation": "create",
                "model_name": model_name,
                "token_used": 0,
                "enhanced_prompt": user_prompt
            }

        data = resp.json()
        gateway_tokens = data.get("usage", {}).get("total_tokens", 0)
        content_str = data["choices"][0]["message"]["content"]
        parsed = json.loads(content_str)

        enhanced_prompt = (
            parsed.get("enhanced_prompt")
            or parsed.get("refined_query")
            or user_prompt
        )
        intent = (
            parsed.get("intent")
            or parsed.get("intent_type")
            or parsed.get("operation")
            or "general_qa"
        ).lower()

        # Engine recommendation resolution
        target_model = (
            parsed.get("suggested_model")
            or parsed.get("suggested_model")
            or parsed.get("recommended_engine")
            or parsed.get("preferred_model")
            or ("create-nexus-slm" if intent in ("greetings", "create", "alter", "drop") else "groq")
        ).strip().lower()

        # Strict routing rules:
        if intent == "greetings":
            target_model = "create-nexus-slm"
        elif intent in ("general_qa", "out_of_scope"):
            target_model = "groq"
        elif "create-nexus" in target_model or "slm" in target_model:
            target_model = "create-nexus-slm"
        else:
            target_model = "groq"

        logger.info(f"Groq Gateway decision: intent='{intent}' | suggested_model='{target_model}' | tokens={gateway_tokens} | enhanced='{enhanced_prompt}'")

        return {
            "intent": intent,
            "suggested_model": target_model,
            "token_used": gateway_tokens,
            "enhanced_prompt": enhanced_prompt
        }

    except Exception as e:
        logger.error(f"Exception during Groq Gateway execution: {e}", exc_info=True)
        return {
            "intent": "create",
            "suggested_model": "create-nexus-slm",
            "token_used": 0,
            "enhanced_prompt": user_prompt
        }


def query_groq_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout: int = 20
) -> Dict[str, Any]:
    """Directly query the Groq LLM for technical QA, world knowledge, or complex tasks."""
    model_name = normalize_groq_model(model or DEFAULT_GROQ_MODEL)
    key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not key:
        return {
            "content": "Groq API key not configured.",
            "token_used": 0
        }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    sys_content = system_prompt or "You are an expert database architect and computer science assistant. Provide concise, clear, accurate technical answers."
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            d = resp.json()
            content = d["choices"][0]["message"]["content"]
            tokens = d.get("usage", {}).get("total_tokens", 0)
            return {"content": content, "token_used": tokens}
        else:
            return {"content": f"Groq API error ({resp.status_code}): {resp.text}", "token_used": 0}
    except Exception as e:
        logger.error(f"Error querying Groq LLM: {e}", exc_info=True)
        return {"content": f"Error querying Groq: {str(e)}", "token_used": 0}


def route_to_model(
    gateway_decision: Dict[str, Any],
    service: Any = None,
    summary: str = "",
    dbml: str = "",
    enable_summary: bool = True,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Dedicated routing function:
    - Greetings: Handled by our model itself (create-nexus-slm).
    - Questions / general_qa / out_of_scope: Routed to Groq LLM.
    - DDL tasks within capability (create, alter, drop): Routed to our local model (create-nexus-slm).
    - Over-capacity DDL tasks: Routed to Groq model.
    """
    op = gateway_decision.get("operation", "general_qa").lower()
    enhanced = gateway_decision.get("enhanced_prompt", "")
    gw_tokens = gateway_decision.get("token_used", 0)
    target_model = gateway_decision.get("model_name", "create-nexus-slm")

    try:
        # Route 1: Questions / General QA / Out-of-scope -> Groq model
        if target_model == "groq" or op in ("general_qa", "out_of_scope"):
            logger.info(f"Routing question/query to Groq model: operation='{op}', query='{enhanced}'")
            groq_resp = query_groq_llm(prompt=enhanced, api_key=api_key)
            total_tokens = gw_tokens + groq_resp.get("token_used", 0)
            return {
                "message": groq_resp.get("content", "Answered by Groq"),
                "operation": op,
                "model_name": "groq",
                "token_used": total_tokens,
                "enhanced_prompt": enhanced,
                "details": None
            }

        # Route 2: Greetings or DDL within capability -> Handled by our local model (create-nexus-slm)
        logger.info(f"Routing to local model (create-nexus-slm): operation='{op}', prompt='{enhanced}'")
        if service is None:
            from src.generator.service import get_creation_service
            service = get_creation_service()

        contract_dict, slm_tokens = service.generate(
            user_query=enhanced,
            summary=summary,
            dbml=dbml,
            enable_summary=enable_summary,
            max_tokens=max_tokens
        )

        sql_out = contract_dict.get("sql_query") or contract_dict.get("sql") or ""
        msg = sql_out.strip() or contract_dict.get("explanation", "Processed successfully.")

        logger.info(f"Local SLM completed: op='{op}', generated {len(sql_out)} chars of SQL.")

        return {
            "message": msg,
            "operation": op,
            "model_name": "create-nexus-slm",
            "token_used": gw_tokens + slm_tokens,
            "enhanced_prompt": enhanced,
            "details": contract_dict
        }

    except Exception as e:
        logger.error(f"Error in route_to_model for operation '{op}': {e}", exc_info=True)
        return {
            "message": f"Execution error: {str(e)}",
            "operation": op,
            "model_name": "create-nexus-slm",
            "token_used": gw_tokens,
            "enhanced_prompt": enhanced,
            "details": {"error": str(e)}
        }


# Alias for backward compatibility
dispatch_to_model = route_to_model
