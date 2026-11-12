import requests
import json

URL = "http://127.0.0.1:8000/api/v1/generate"

def test_query(title, prompt, summary="", expected_intent=None, expected_op=None):
    print(f"\n{'='*75}")
    print(f"TEST: {title}")
    print(f"Prompt: {prompt}")
    if summary:
        print(f"Incoming Summary (snippet): {summary[:120]}...")
    payload = {"prompt": prompt, "summary": summary, "enable_summary": True}
    res = requests.post(URL, json=payload).json()
    status_code = res.get("status_code")
    status = res.get("status")
    data = res.get("data", {})
    intent = data.get("intent")
    op = data.get("operation")
    sql = data.get("sql", "")
    updated_sum = data.get("updated_summary", "")
    exp = data.get("explanation", "")
    clarif = data.get("needs_clarification", False)

    print(f"Result -> Status: {status_code} ({status}) | Intent: {intent} | Op: {op} | Clarif: {clarif}")
    if sql:
        print(f"SQL Snippet:\n{sql[:250]}...")
    if updated_sum:
        req_sum_part = updated_sum.split("Available Tables:")[0].strip()
        print(f"Summary Ledger Header:\n{req_sum_part}")
    print(f"Explanation:\n{exp[:180]}...")
    return data

print("Starting Comprehensive DB Nexus Multi-Turn & Multi-Category Test Suite...")

# 1. Sanity Test
t1 = test_query("Sanity Single Table", "create a books table with id, title, and price")

# 2. General Chat
t2 = test_query("General Chat", "hello, who are you?")

# 3. Technical QA
t3 = test_query("Technical QA", "What is third normal form (3NF) in database design?")

# 4. Guardrail: Ambiguity
t4 = test_query("Guardrail Ambiguity", "create a table")

# 5. Guardrail: DML
t5 = test_query("Guardrail DML", "insert into users (name) values ('Alice')")

# 6. Guardrail: Invalid Datatype
t6 = test_query("Guardrail Invalid Type", "create table accounts with id int and balance moneytext")

# 7. Misspelled / Jumbled (Monkey Test)
t7 = test_query("Misspelled / Jumbled Prompt", "mak a ecomerce app with usrs and prodcts")

# 8. Multi-Turn Sequence with Declarative Evolved Schema
print("\n" + "#"*75)
print("### MULTI-TURN SCHEMA EVOLUTION SEQUENCE ###")
print("#"*75)

turn1 = test_query("Turn 1: Initial Schema", "create customers and products tables")
sum1 = turn1.get("updated_summary", "")

turn2 = test_query("Turn 2: Follow-up Add Table", "now add orders table with fk to customers and total amount", summary=sum1)
sum2 = turn2.get("updated_summary", "")

turn3 = test_query("Turn 3: Follow-up Alteration (Add Column with Typo)", "add is_actve colmn to customers", summary=sum2)
sum3 = turn3.get("updated_summary", "")

turn4 = test_query("Turn 4: Follow-up Drop Table", "drop products table", summary=sum3)
sum4 = turn4.get("updated_summary", "")

print("\n" + "="*75)
print("TEST SUITE COMPLETED!")
print("="*75)

