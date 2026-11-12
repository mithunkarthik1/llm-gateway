import requests
import json

def test_q(q):
    r = requests.post("http://127.0.0.1:8000/api/v1/generate", json={"user_query": q, "enable_summary": True})
    data = r.json().get("data", {})
    print(f"Q: {q}")
    print(f"  Intent: {data.get('intent')} | Op: {data.get('operation')} | NeedsClarif: {data.get('needs_clarification')}")
    print(f"  SQL:\n{data.get('sql_query')}")
    print(f"  Exp: {data.get('explanation')}")
    print("-" * 50)

if __name__ == "__main__":
    test_q("Hi")
    test_q("What is a database?")
    test_q("Create two independent tables: employees with id and name, and departments with id and department_name without any relationships.")
    test_q("What is the capital city of Australia?")
    test_q("Create an order_items table with a composite primary key consisting of (order_id, item_id).")
    test_q("Create a database for our system.")

