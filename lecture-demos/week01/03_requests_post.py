#!/usr/bin/env python3
"""
Week 01: POST Requests
Sending data to APIs
"""

import requests

# =============================================================================
# PART 1: POST with JSON Body
# =============================================================================

new_item = {
    "name": "Mango",
    "price": 2.50,
    "quantity": 50
}

response = requests.post(
    "https://nipun-api-testing.hf.space/items",
    json=new_item  # Automatically sets Content-Type: application/json
)

print("Created item:")
print(response.json())

# =============================================================================
# PART 2: POST with Form Data
# =============================================================================

form_data = {
    "name": "Alice",
    "email": "alice@example.com",
    "message": "Hello from Python!"
}

response = requests.post(
    "https://nipun-api-testing.hf.space/form/contact",
    data=form_data  # Sends as application/x-www-form-urlencoded
)

print("\nForm submission result:")
print(response.json())

# =============================================================================
# PART 3: Echo Endpoint - See What You Sent
# =============================================================================

# Useful for debugging - API echoes back everything
test_data = {
    "test": "data",
    "numbers": [1, 2, 3],
    "nested": {"key": "value"}
}

response = requests.post(
    "https://nipun-api-testing.hf.space/echo",
    json=test_data
)

print("\nEcho response (what the server received):")
print(response.json())

# =============================================================================
# PART 4: json= vs data= Parameter
# =============================================================================

# json= : Sends JSON, sets Content-Type: application/json
# data= : Sends form data, sets Content-Type: application/x-www-form-urlencoded

# These are different!
print("\n--- json= parameter ---")
r1 = requests.post("https://nipun-api-testing.hf.space/echo", json={"key": "value"})
print(f"Content-Type sent: {r1.json().get('content_type', 'N/A')}")

print("\n--- data= parameter ---")
r2 = requests.post("https://nipun-api-testing.hf.space/echo", data={"key": "value"})
print(f"Content-Type sent: {r2.json().get('content_type', 'N/A')}")
