#!/usr/bin/env python3
"""
Week 01: HTTP Headers with requests
Understanding request and response headers
"""

import requests

# =============================================================================
# PART 1: View Response Headers
# =============================================================================

response = requests.get("https://nipun-api-testing.hf.space/items")

print("Response Headers:")
for key, value in response.headers.items():
    print(f"  {key}: {value}")

# =============================================================================
# PART 2: Check What Headers YOU Send
# =============================================================================

# Our API has an endpoint that echoes back your headers
response = requests.get("https://nipun-api-testing.hf.space/headers")
my_headers = response.json()

print("\nHeaders your request sent:")
for key, value in my_headers["your_headers"].items():
    print(f"  {key}: {value}")

# =============================================================================
# PART 3: Custom Headers
# =============================================================================

# Add custom headers to your request
custom_headers = {
    "User-Agent": "MyPythonApp/1.0",
    "Accept": "application/json",
    "X-Custom-Header": "Hello from Python!"
}

response = requests.get(
    "https://nipun-api-testing.hf.space/headers",
    headers=custom_headers
)

print("\nWith custom headers:")
print(response.json()["your_headers"])

# =============================================================================
# PART 4: Content-Type Header
# =============================================================================

# The Content-Type tells the server what format you're sending
# Accept header tells server what format you want back

response = requests.get(
    "https://nipun-api-testing.hf.space/items",
    headers={"Accept": "application/json"}
)

print(f"\nContent-Type received: {response.headers.get('Content-Type')}")
