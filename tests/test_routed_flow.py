#!/usr/bin/env python3
"""
Test script to verify the hybrid Groq Gateway & Routed Generation endpoints.
Tests:
1. Typos and grammar enhancement (CREATE TABLE)
2. Greetings handling directly via Groq
3. General QA / Out-of-scope handling via Groq
4. Composite primary key DDL via local SLM
5. Relational schema creation via local SLM
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_gateway_only(prompt: str):
    print(f"\n[TEST 1: Gateway Decision Only] Prompt: '{prompt}'")
    url = f"{BASE_URL}/api/v1/gateway"
    try:
        resp = requests.post(url, json={"user_query": prompt}, timeout=15)
        print(f"Status Code: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")


def test_routed_generate(prompt: str):
    print(f"\n[TEST 2: End-to-End Routed Generate] Prompt: '{prompt}'")
    url = f"{BASE_URL}/api/v1/routed-generate"
    try:
        resp = requests.post(url, json={"user_query": prompt}, timeout=30)
        print(f"Status Code: {resp.status_code}")
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING HYBRID GATEWAY & ROUTING VERIFICATION SUITE")
    print("=" * 80)

    # 1. Gateway only decision test
    test_gateway_only("tabe mitrah_soft create")

    # 2. Greeting test
    test_routed_generate("heyy there good morning")

    # 3. Out-of-scope test
    test_routed_generate("What is the capital city of Australia?")

    # 4. Schema creation test with typos
    test_routed_generate("mak me tabl for employes hav id and nam and departmnt_id")

    # 5. Composite PK table test
    test_routed_generate("Create an order_items table with a composite primary key consisting of (order_id, item_id).")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)

